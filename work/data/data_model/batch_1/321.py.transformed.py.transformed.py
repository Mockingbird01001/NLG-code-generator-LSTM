import ctypes
import comtypes.automation
import comtypes.typeinfo
import comtypes.client
import comtypes.client.lazybind
from comtypes import COMError, IUnknown, _is_object
import comtypes.hresult as hres
ERRORS_BAD_CONTEXT = [
    hres.DISP_E_MEMBERNOTFOUND,
    hres.DISP_E_BADPARAMCOUNT,
    hres.DISP_E_PARAMNOTOPTIONAL,
    hres.DISP_E_TYPEMISMATCH,
    hres.E_INVALIDARG,
]
def Dispatch(obj):
    if isinstance(obj, _Dispatch):
        return obj
    if isinstance(obj, ctypes.POINTER(comtypes.automation.IDispatch)):
        try:
            tinfo = obj.GetTypeInfo(0)
        except (comtypes.COMError, WindowsError):
            return _Dispatch(obj)
        return comtypes.client.lazybind.Dispatch(obj, tinfo)
    return obj
class MethodCaller:
    def __init__(self, _id, _obj):
        self._id = _id
        self._obj = _obj
    def __call__(self, *args):
        return self._obj._comobj.Invoke(self._id, *args)
    def __getitem__(self, *args):
        return self._obj._comobj.Invoke(self._id, *args,
                                        **dict(_invkind=comtypes.automation.DISPATCH_PROPERTYGET))
    def __setitem__(self, *args):
        if _is_object(args[-1]):
            self._obj._comobj.Invoke(self._id, *args,
                                        **dict(_invkind=comtypes.automation.DISPATCH_PROPERTYPUTREF))
        else:
            self._obj._comobj.Invoke(self._id, *args,
                                        **dict(_invkind=comtypes.automation.DISPATCH_PROPERTYPUT))
class _Dispatch(object):
    def __init__(self, comobj):
        self.__dict__["_comobj"] = comobj
        self.__dict__["_ids"] = {}
        self.__dict__["_methods"] = set()
    def __enum(self):
        e = self._comobj.Invoke(-4)
        return e.QueryInterface(comtypes.automation.IEnumVARIANT)
    def __cmp__(self, other):
        if not isinstance(other, _Dispatch):
            return 1
        return cmp(self._comobj, other._comobj)
    def __hash__(self):
        return hash(self._comobj)
    def __getitem__(self, index):
        enum = self.__enum()
        if index > 0:
            if 0 != enum.Skip(index):
                raise IndexError("index out of range")
        item, fetched = enum.Next(1)
        if not fetched:
            raise IndexError("index out of range")
        return item
    def QueryInterface(self, *args):
        return self._comobj.QueryInterface(*args)
    def _FlagAsMethod(self, *names):
        self._methods.update(names)
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dispid = self._ids.get(name)
        if not dispid:
            dispid = self._comobj.GetIDsOfNames(name)[0]
            self._ids[name] = dispid
        if name in self._methods:
            result = MethodCaller(dispid, self)
            self.__dict__[name] = result
            return result
        flags = comtypes.automation.DISPATCH_PROPERTYGET
        try:
            result = self._comobj.Invoke(dispid, _invkind=flags)
        except COMError as err:
            (hresult, text, details) = err.args
            if hresult in ERRORS_BAD_CONTEXT:
                result = MethodCaller(dispid, self)
                self.__dict__[name] = result
            else:
                raise
        except:
            raise
        return result
    def __setattr__(self, name, value):
        dispid = self._ids.get(name)
        if not dispid:
            dispid = self._comobj.GetIDsOfNames(name)[0]
            self._ids[name] = dispid
        flags = 8 if _is_object(value) else 4
        return self._comobj.Invoke(dispid, value, _invkind=flags)
    def __iter__(self):
        return _Collection(self.__enum())
class _Collection(object):
    def __init__(self, enum):
        self.enum = enum
    def __next__(self):
        item, fetched = self.enum.Next(1)
        if fetched:
            return item
        raise StopIteration
    def __iter__(self):
        return self
__all__ = ["Dispatch"]
