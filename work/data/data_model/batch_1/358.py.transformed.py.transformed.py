import ctypes
import traceback
import comtypes
import comtypes.hresult
import comtypes.automation
import comtypes.typeinfo
import comtypes.connectionpoints
from comtypes.client._generate import GetModule
import logging
logger = logging.getLogger(__name__)
class _AdviseConnection(object):
    def __init__(self, source, interface, receiver):
        self.cp = None
        self.cookie = None
        self.receiver = None
        self._connect(source, interface, receiver)
    def _connect(self, source, interface, receiver):
        cpc = source.QueryInterface(comtypes.connectionpoints.IConnectionPointContainer)
        self.cp = cpc.FindConnectionPoint(ctypes.byref(interface._iid_))
        logger.debug("Start advise %s", interface)
        self.cookie = self.cp.Advise(receiver)
        self.receiver = receiver
    def disconnect(self):
        if self.cookie:
            self.cp.Unadvise(self.cookie)
            logger.debug("Unadvised %s", self.cp)
            self.cp = None
            self.cookie = None
            del self.receiver
    def __del__(self):
        try:
            if self.cookie is not None:
                self.cp.Unadvise(self.cookie)
        except (comtypes.COMError, WindowsError):
            pass
def FindOutgoingInterface(source):
    try:
        pci = source.QueryInterface(comtypes.typeinfo.IProvideClassInfo2)
        guid = pci.GetGUID(1)
    except comtypes.COMError:
        pass
    else:
        try:
            interface = comtypes.com_interface_registry[str(guid)]
        except KeyError:
            tinfo = pci.GetClassInfo()
            tlib, index = tinfo.GetContainingTypeLib()
            GetModule(tlib)
            interface = comtypes.com_interface_registry[str(guid)]
        logger.debug("%s using sinkinterface %s", source, interface)
        return interface
    clsid = source.__dict__.get('__clsid')
    try:
        interface = comtypes.com_coclass_registry[clsid]._outgoing_interfaces_[0]
    except KeyError:
        pass
    else:
        logger.debug("%s using sinkinterface from clsid %s", source, interface)
        return interface
    raise TypeError("cannot determine source interface")
def find_single_connection_interface(source):
    cpc = source.QueryInterface(comtypes.connectionpoints.IConnectionPointContainer)
    enum = cpc.EnumConnectionPoints()
    iid = enum.next().GetConnectionInterface()
    try:
        next(enum)
    except StopIteration:
        try:
            interface = comtypes.com_interface_registry[str(iid)]
        except KeyError:
            return None
        else:
            logger.debug("%s using sinkinterface from iid %s", source, interface)
            return interface
    else:
        logger.debug("%s has more than one connection point", source)
    return None
def report_errors(func):
    if func.__code__.co_varnames[:2] == ('self', 'this'):
        def error_printer(self, this, *args, **kw):
            try:
                return func(self, this, *args, **kw)
            except:
                traceback.print_exc()
                raise
    else:
        def error_printer(*args, **kw):
            try:
                return func(*args, **kw)
            except:
                traceback.print_exc()
                raise
    return error_printer
from comtypes._comobject import _MethodFinder
class _SinkMethodFinder(_MethodFinder):
    def __init__(self, inst, sink):
        super(_SinkMethodFinder, self).__init__(inst)
        self.sink = sink
    def find_method(self, fq_name, mthname):
        impl = self._find_method(fq_name, mthname)
        try:
            im_self, im_func = impl.__self__, impl.__func__
            method = report_errors(im_func)
            return comtypes.instancemethod(method,
                                           im_self,
                                           type(im_self))
        except AttributeError as details:
            raise RuntimeError(details)
    def _find_method(self, fq_name, mthname):
        try:
            return super(_SinkMethodFinder, self).find_method(fq_name, mthname)
        except AttributeError:
            try:
                return getattr(self.sink, fq_name)
            except AttributeError:
                return getattr(self.sink, mthname)
def CreateEventReceiver(interface, handler):
    class Sink(comtypes.COMObject):
        _com_interfaces_ = [interface]
        def _get_method_finder_(self, itf):
            return _SinkMethodFinder(self, handler)
    sink = Sink()
    if issubclass(interface, comtypes.automation.IDispatch)           and not hasattr(sink, "_dispimpl_"):
        finder = sink._get_method_finder_(interface)
        dispimpl = sink._dispimpl_ = {}
        for m in interface._methods_:
            restype, mthname, argtypes, paramflags, idlflags, helptext = m
            dispid = idlflags[0]
            impl = finder.get_impl(interface, mthname, paramflags, idlflags)
            dispimpl[(dispid, comtypes.automation.DISPATCH_METHOD)] = impl
    return sink
def GetEvents(source, sink, interface=None):
    if interface is None:
        interface = FindOutgoingInterface(source)
    rcv = CreateEventReceiver(interface, sink)
    return _AdviseConnection(source, interface, rcv)
class EventDumper(object):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        def handler(self, this, *args, **kw):
            args = (None,) + args
            print("Event %s(%s)" % (name, ", ".join([repr(a) for a in args])))
        return comtypes.instancemethod(handler, self, EventDumper)
def ShowEvents(source, interface=None):
    return comtypes.client.GetEvents(source, sink=EventDumper(), interface=interface)
_handles_type = ctypes.c_void_p * 1
def PumpEvents(timeout):
    hevt = ctypes.windll.kernel32.CreateEventA(None, True, False, None)
    handles = _handles_type(hevt)
    RPC_S_CALLPENDING = -2147417835
    def HandlerRoutine(dwCtrlType):
        if dwCtrlType == 0:
            ctypes.windll.kernel32.SetEvent(hevt)
            return 1
        return 0
    HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(HandlerRoutine)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(HandlerRoutine, 1)
    try:
        try:
            res = ctypes.oledll.ole32.CoWaitForMultipleHandles(0,
                                                               int(timeout * 1000),
                                                               len(handles), handles,
                                                               ctypes.byref(ctypes.c_ulong()))
        except WindowsError as details:
            if details.winerror != RPC_S_CALLPENDING:
                raise
        else:
            raise KeyboardInterrupt
    finally:
        ctypes.windll.kernel32.CloseHandle(hevt)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(HandlerRoutine, 0)
