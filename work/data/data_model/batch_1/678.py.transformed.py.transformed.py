
from ctypes import POINTER, c_void_p, cast
import comtypes
def _wrap_coclass(self):
    itf = self._com_interfaces_[0]
    punk = cast(self, POINTER(itf))
    result = punk.QueryInterface(itf)
    result.__dict__["__clsid"] = str(self._reg_clsid_)
    return result
def _coclass_from_param(cls, obj):
    if isinstance(obj, (cls._com_interfaces_[0], cls)):
        return obj
    raise TypeError(obj)
class _coclass_meta(type):
    def __new__(cls, name, bases, namespace):
        klass = type.__new__(cls, name, bases, namespace)
        if bases == (object,):
            return klass
        if "_reg_clsid_" in namespace:
            clsid = namespace["_reg_clsid_"]
            comtypes.com_coclass_registry[str(clsid)] = klass
        PTR = _coclass_pointer_meta("POINTER(%s)" % klass.__name__,
                                    (klass, c_void_p),
                                    {"__ctypes_from_outparam__": _wrap_coclass,
                                     "from_param": classmethod(_coclass_from_param),
                                     })
        from ctypes import _pointer_type_cache
        _pointer_type_cache[klass] = PTR
        return klass
class _coclass_pointer_meta(type(c_void_p), _coclass_meta):
    pass
