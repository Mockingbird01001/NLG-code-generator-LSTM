
import collections.abc
import contextlib
from .overrides import set_module
from .umath import (
    UFUNC_BUFSIZE_DEFAULT,
    ERR_IGNORE, ERR_WARN, ERR_RAISE, ERR_CALL, ERR_PRINT, ERR_LOG, ERR_DEFAULT,
    SHIFT_DIVIDEBYZERO, SHIFT_OVERFLOW, SHIFT_UNDERFLOW, SHIFT_INVALID,
)
from . import umath
__all__ = [
    "seterr", "geterr", "setbufsize", "getbufsize", "seterrcall", "geterrcall",
    "errstate",
]
_errdict = {"ignore": ERR_IGNORE,
            "warn": ERR_WARN,
            "raise": ERR_RAISE,
            "call": ERR_CALL,
            "print": ERR_PRINT,
            "log": ERR_LOG}
_errdict_rev = {value: key for key, value in _errdict.items()}
@set_module('numpy')
def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    pyvals = umath.geterrobj()
    old = geterr()
    if divide is None:
        divide = all or old['divide']
    if over is None:
        over = all or old['over']
    if under is None:
        under = all or old['under']
    if invalid is None:
        invalid = all or old['invalid']
    maskvalue = ((_errdict[divide] << SHIFT_DIVIDEBYZERO) +
                 (_errdict[over] << SHIFT_OVERFLOW) +
                 (_errdict[under] << SHIFT_UNDERFLOW) +
                 (_errdict[invalid] << SHIFT_INVALID))
    pyvals[1] = maskvalue
    umath.seterrobj(pyvals)
    return old
@set_module('numpy')
def geterr():
    maskvalue = umath.geterrobj()[1]
    mask = 7
    res = {}
    val = (maskvalue >> SHIFT_DIVIDEBYZERO) & mask
    res['divide'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_OVERFLOW) & mask
    res['over'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_UNDERFLOW) & mask
    res['under'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_INVALID) & mask
    res['invalid'] = _errdict_rev[val]
    return res
@set_module('numpy')
def setbufsize(size):
    if size > 10e6:
        raise ValueError("Buffer size, %s, is too big." % size)
    if size < 5:
        raise ValueError("Buffer size, %s, is too small." % size)
    if size % 16 != 0:
        raise ValueError("Buffer size, %s, is not a multiple of 16." % size)
    pyvals = umath.geterrobj()
    old = getbufsize()
    pyvals[0] = size
    umath.seterrobj(pyvals)
    return old
@set_module('numpy')
def getbufsize():
    return umath.geterrobj()[0]
@set_module('numpy')
def seterrcall(func):
    if func is not None and not isinstance(func, collections.abc.Callable):
        if (not hasattr(func, 'write') or
                not isinstance(func.write, collections.abc.Callable)):
            raise ValueError("Only callable can be used as callback")
    pyvals = umath.geterrobj()
    old = geterrcall()
    pyvals[2] = func
    umath.seterrobj(pyvals)
    return old
@set_module('numpy')
def geterrcall():
    return umath.geterrobj()[2]
class _unspecified:
    pass
_Unspecified = _unspecified()
@set_module('numpy')
class errstate(contextlib.ContextDecorator):
    def __init__(self, *, call=_Unspecified, **kwargs):
        self.call = call
        self.kwargs = kwargs
    def __enter__(self):
        self.oldstate = seterr(**self.kwargs)
        if self.call is not _Unspecified:
            self.oldcall = seterrcall(self.call)
    def __exit__(self, *exc_info):
        seterr(**self.oldstate)
        if self.call is not _Unspecified:
            seterrcall(self.oldcall)
def _setdef():
    defval = [UFUNC_BUFSIZE_DEFAULT, ERR_DEFAULT, None]
    umath.seterrobj(defval)
_setdef()
