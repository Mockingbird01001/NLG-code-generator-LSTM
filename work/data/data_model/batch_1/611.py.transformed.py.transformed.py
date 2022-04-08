import comtypes
import comtypes.automation
from comtypes.automation import IEnumVARIANT
from comtypes.automation import DISPATCH_METHOD
from comtypes.automation import DISPATCH_PROPERTYGET
from comtypes.automation import DISPATCH_PROPERTYPUT
from comtypes.automation import DISPATCH_PROPERTYPUTREF
from comtypes.automation import DISPID_VALUE
from comtypes.automation import DISPID_NEWENUM
from comtypes.typeinfo import FUNC_PUREVIRTUAL, FUNC_DISPATCH
class FuncDesc(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
_all_slice = slice(None, None, None)
class NamedProperty(object):
    def __init__(self, disp, get, put, putref):
        self.get = get
        self.put = put
        self.putref = putref
        self.disp = disp
    def __getitem__(self, arg):
        if self.get is None:
            raise TypeError("unsubscriptable object")
        if isinstance(arg, tuple):
            return self.disp._comobj._invoke(self.get.memid,
                                             self.get.invkind,
                                             0,
                                             *arg)
        elif arg == _all_slice:
            return self.disp._comobj._invoke(self.get.memid,
                                             self.get.invkind,
                                             0)
        return self.disp._comobj._invoke(self.get.memid,
                                         self.get.invkind,
                                         0,
                                         *[arg])
    def __call__(self, *args):
        if self.get is None:
            raise TypeError("object is not callable")
        return self.disp._comobj._invoke(self.get.memid,
                                            self.get.invkind,
                                            0,
                                            *args)
    def __setitem__(self, name, value):
        if self.put is None and self.putref is None:
            raise TypeError("object does not support item assignment")
        if comtypes._is_object(value):
            descr = self.putref or self.put
        else:
            descr = self.put or self.putref
        if isinstance(name, tuple):
            self.disp._comobj._invoke(descr.memid,
                                      descr.invkind,
                                      0,
                                      *(name + (value,)))
        elif name == _all_slice:
            self.disp._comobj._invoke(descr.memid,
                                      descr.invkind,
                                      0,
                                      value)
        else:
            self.disp._comobj._invoke(descr.memid,
                                      descr.invkind,
                                      0,
                                      name,
                                      value)
    def __iter__(self):
        msg = "%r is not iterable" % self.disp
        raise TypeError(msg)
class Dispatch(object):
    def __init__(self, comobj, tinfo):
        self.__dict__["_comobj"] = comobj
        self.__dict__["_tinfo"] = tinfo
        self.__dict__["_tcomp"] = tinfo.GetTypeComp()
        self.__dict__["_tdesc"] = {}
    def __bind(self, name, invkind):
        try:
            return self._tdesc[(name, invkind)]
        except KeyError:
            try:
                descr = self._tcomp.Bind(name, invkind)[1]
            except comtypes.COMError:
                info = None
            else:
                info = FuncDesc(memid=descr.memid,
                                invkind=descr.invkind,
                                cParams=descr.cParams,
                                funckind=descr.funckind)
            self._tdesc[(name, invkind)] = info
            return info
    def QueryInterface(self, *args):
        return self._comobj.QueryInterface(*args)
    def __cmp__(self, other):
        if not isinstance(other, Dispatch):
            return 1
        return cmp(self._comobj, other._comobj)
    def __eq__(self, other):
        return isinstance(other, Dispatch) and               self._comobj == other._comobj
    def __hash__(self):
        return hash(self._comobj)
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        descr = self.__bind(name, DISPATCH_METHOD | DISPATCH_PROPERTYGET)
        if descr is None:
            raise AttributeError(name)
        if descr.invkind == DISPATCH_PROPERTYGET:
            if descr.funckind == FUNC_DISPATCH:
                if descr.cParams == 0:
                    return self._comobj._invoke(descr.memid, descr.invkind, 0)
            elif descr.funckind == FUNC_PUREVIRTUAL:
                if descr.cParams == 1:
                    return self._comobj._invoke(descr.memid, descr.invkind, 0)
            else:
                raise RuntimeError("funckind %d not yet implemented" % descr.funckind)
            put = self.__bind(name, DISPATCH_PROPERTYPUT)
            putref = self.__bind(name, DISPATCH_PROPERTYPUTREF)
            return NamedProperty(self, descr, put, putref)
        else:
            def caller(*args):
                return self._comobj._invoke(descr.memid, descr.invkind, 0, *args)
            try:
                caller.__name__ = name
            except TypeError:
                pass
            return caller
    def __setattr__(self, name, value):
        put = self.__bind(name, DISPATCH_PROPERTYPUT)
        putref = self.__bind(name, DISPATCH_PROPERTYPUTREF)
        if not put and not putref:
            raise AttributeError(name)
        if comtypes._is_object(value):
            descr = putref or put
        else:
            descr = put or putref
        if descr.cParams == 1:
            self._comobj._invoke(descr.memid, descr.invkind, 0, value)
            return
        raise AttributeError(name)
    def __call__(self, *args):
        return self._comobj._invoke(DISPID_VALUE,
                                    DISPATCH_METHOD | DISPATCH_PROPERTYGET,
                                    0,
                                    *args)
    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            args = arg
        elif arg == _all_slice:
            args = ()
        else:
            args = (arg,)
        try:
            return self._comobj._invoke(DISPID_VALUE,
                                        DISPATCH_METHOD | DISPATCH_PROPERTYGET,
                                        0,
                                        *args)
        except comtypes.COMError:
            return iter(self)[arg]
    def __setitem__(self, name, value):
        if comtypes._is_object(value):
            invkind = DISPATCH_PROPERTYPUTREF
        else:
            invkind = DISPATCH_PROPERTYPUT
        if isinstance(name, tuple):
            args = name + (value,)
        elif name == _all_slice:
            args = (value,)
        else:
            args = (name, value)
        return self._comobj._invoke(DISPID_VALUE,
                                    invkind,
                                    0,
                                    *args)
    def __iter__(self):
        punk = self._comobj._invoke(DISPID_NEWENUM,
                                    DISPATCH_METHOD | DISPATCH_PROPERTYGET,
                                    0)
        enum = punk.QueryInterface(IEnumVARIANT)
        enum._dynamic = True
        return enum
