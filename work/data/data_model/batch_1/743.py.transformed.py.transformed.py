
from __future__ import absolute_import
import sys
from types import FunctionType
from future.utils import PY3, PY26
_builtin_super = super
_SENTINEL = object()
def newsuper(typ=_SENTINEL, type_or_obj=_SENTINEL, framedepth=1):
    if typ is _SENTINEL:
        f = sys._getframe(framedepth)
        try:
            type_or_obj = f.f_locals[f.f_code.co_varnames[0]]
        except (IndexError, KeyError,):
            raise RuntimeError('super() used in a function with no args')
        try:
            mro = type_or_obj.__mro__
        except (AttributeError, RuntimeError):
            try:
                mro = type_or_obj.__class__.__mro__
            except AttributeError:
                raise RuntimeError('super() used with a non-newstyle class')
        for typ in mro:
            for meth in typ.__dict__.values():
                try:
                    while not isinstance(meth,FunctionType):
                        if isinstance(meth, property):
                            meth = meth.fget
                        else:
                            try:
                                meth = meth.__func__
                            except AttributeError:
                                meth = meth.__get__(type_or_obj, typ)
                except (AttributeError, TypeError):
                    continue
                if meth.func_code is f.f_code:
                    break
            else:
                continue
            break
        else:
            raise RuntimeError('super() called outside a method')
    if type_or_obj is not _SENTINEL:
        return _builtin_super(typ, type_or_obj)
    return _builtin_super(typ)
def superm(*args, **kwds):
    f = sys._getframe(1)
    nm = f.f_code.co_name
    return getattr(newsuper(framedepth=2),nm)(*args, **kwds)
__all__ = ['newsuper']
