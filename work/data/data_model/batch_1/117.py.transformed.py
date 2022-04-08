from ctypes import *
from comtypes import IUnknown, COMObject, COMError
from comtypes.hresult import *
from comtypes.typeinfo import LoadRegTypeLib
from comtypes.connectionpoints import IConnectionPoint
from comtypes.automation import IDispatch
import logging
logger = logging.getLogger(__name__)
__all__ = ["ConnectableObjectMixin"]
class ConnectionPointImpl(COMObject):
    _com_interfaces_ = [IConnectionPoint]
    def __init__(self, sink_interface, sink_typeinfo):
        super(ConnectionPointImpl, self).__init__()
        self._connections = {}
        self._cookie = 0
        self._sink_interface = sink_interface
        self._typeinfo = sink_typeinfo
    def IConnectionPoint_Advise(self, this, pUnk, pdwCookie):
        if not pUnk or not pdwCookie:
            return E_POINTER
        logger.debug("Advise")
        try:
            ptr = pUnk.QueryInterface(self._sink_interface)
        except COMError:
            return CONNECT_E_CANNOTCONNECT
        pdwCookie[0] = self._cookie = self._cookie + 1
        self._connections[self._cookie] = ptr
        return S_OK
    def IConnectionPoint_Unadvise(self, this, dwCookie):
        logger.debug("Unadvise %s", dwCookie)
        try:
            del self._connections[dwCookie]
        except KeyError:
            return CONNECT_E_NOCONNECTION
        return S_OK
    def IConnectionPoint_GetConnectionPointContainer(self, this, ppCPC):
        return E_NOTIMPL
    def IConnectionPoint_GetConnectionInterface(self, this, pIID):
        return E_NOTIMPL
    def _call_sinks(self, name, *args, **kw):
        results = []
        logger.debug("_call_sinks(%s, %s, *%s, **%s)", self, name, args, kw)
        if hasattr(self._sink_interface, "Invoke"):
            dispid = self._typeinfo.GetIDsOfNames(name)[0]
            for key, p in list(self._connections.items()):
                try:
                    result = p.Invoke(dispid, *args, **kw)
                except COMError as details:
                    if details.hresult == -2147023174:
                        logger.warning("_call_sinks(%s, %s, *%s, **%s) failed; removing connection",
                                       self, name, args, kw,
                                       exc_info=True)
                        try:
                            del self._connections[key]
                        except KeyError:
                            pass
                    else:
                        logger.warning("_call_sinks(%s, %s, *%s, **%s)", self, name, args, kw,
                                       exc_info=True)
                else:
                    results.append(result)
        else:
            for p in list(self._connections.values()):
                try:
                    result = getattr(p, name)(*args, **kw)
                except COMError as details:
                    if details.hresult == -2147023174:
                        logger.warning("_call_sinks(%s, %s, *%s, **%s) failed; removing connection",
                                       self, name, args, kw,
                                       exc_info=True)
                        del self._connections[key]
                    else:
                        logger.warning("_call_sinks(%s, %s, *%s, **%s)", self, name, args, kw,
                                       exc_info=True)
                else:
                    results.append(result)
        return results
class ConnectableObjectMixin(object):
    def __init__(self):
        super(ConnectableObjectMixin, self).__init__()
        self.__connections = {}
        tlib = LoadRegTypeLib(*self._reg_typelib_)
        for itf in self._outgoing_interfaces_:
            typeinfo = tlib.GetTypeInfoOfGuid(itf._iid_)
            self.__connections[itf] = ConnectionPointImpl(itf, typeinfo)
    def IConnectionPointContainer_EnumConnectionPoints(self, this, ppEnum):
        return E_NOTIMPL
    def IConnectionPointContainer_FindConnectionPoint(self, this, refiid, ppcp):
        iid = refiid[0]
        logger.debug("FindConnectionPoint %s", iid)
        if not ppcp:
            return E_POINTER
        for itf in self._outgoing_interfaces_:
            if itf._iid_ == iid:
                conn = self.__connections[itf]
                result = conn.IUnknown_QueryInterface(None, pointer(IConnectionPoint._iid_), ppcp)
                logger.debug("connectionpoint found, QI() -> %s", result)
                return result
        logger.debug("No connectionpoint found")
        return CONNECT_E_NOCONNECTION
    def Fire_Event(self, itf, name, *args, **kw):
        logger.debug("Fire_Event(%s, %s, *%s, **%s)", itf, name, args, kw)
        if isinstance(itf, int):
            itf = self._outgoing_interfaces_[itf]
        return self.__connections[itf]._call_sinks(name, *args, **kw)
