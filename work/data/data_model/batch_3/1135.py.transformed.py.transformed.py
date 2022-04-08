
import functools
import warnings
__all__ = ['iscomplexobj', 'isrealobj', 'imag', 'iscomplex',
           'isreal', 'nan_to_num', 'real', 'real_if_close',
           'typename', 'asfarray', 'mintypecode', 'asscalar',
           'common_type']
import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, asanyarray, isnan, zeros
from numpy.core.overrides import set_module
from numpy.core import overrides
from .ufunclike import isneginf, isposinf
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')
_typecodes_by_elsize = 'GDFgdfQqLlIiHhBb?'
@set_module('numpy')
def mintypecode(typechars, typeset='GDFgdf', default='d'):
    typecodes = ((isinstance(t, str) and t) or asarray(t).dtype.char
                 for t in typechars)
    intersection = set(t for t in typecodes if t in typeset)
    if not intersection:
        return default
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    return min(intersection, key=_typecodes_by_elsize.index)
def _asfarray_dispatcher(a, dtype=None):
    return (a,)
@array_function_dispatch(_asfarray_dispatcher)
def asfarray(a, dtype=_nx.float_):
    if not _nx.issubdtype(dtype, _nx.inexact):
        dtype = _nx.float_
    return asarray(a, dtype=dtype)
def _real_dispatcher(val):
    return (val,)
@array_function_dispatch(_real_dispatcher)
def real(val):
    try:
        return val.real
    except AttributeError:
        return asanyarray(val).real
def _imag_dispatcher(val):
    return (val,)
@array_function_dispatch(_imag_dispatcher)
def imag(val):
    try:
        return val.imag
    except AttributeError:
        return asanyarray(val).imag
def _is_type_dispatcher(x):
    return (x,)
@array_function_dispatch(_is_type_dispatcher)
def iscomplex(x):
    ax = asanyarray(x)
    if issubclass(ax.dtype.type, _nx.complexfloating):
        return ax.imag != 0
    res = zeros(ax.shape, bool)
    return res[()]
@array_function_dispatch(_is_type_dispatcher)
def isreal(x):
    return imag(x) == 0
@array_function_dispatch(_is_type_dispatcher)
def iscomplexobj(x):
    try:
        dtype = x.dtype
        type_ = dtype.type
    except AttributeError:
        type_ = asarray(x).dtype.type
    return issubclass(type_, _nx.complexfloating)
@array_function_dispatch(_is_type_dispatcher)
def isrealobj(x):
    return not iscomplexobj(x)
def _getmaxmin(t):
    from numpy.core import getlimits
    f = getlimits.finfo(t)
    return f.max, f.min
def _nan_to_num_dispatcher(x, copy=None, nan=None, posinf=None, neginf=None):
    return (x,)
@array_function_dispatch(_nan_to_num_dispatcher)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    x = _nx.array(x, subok=True, copy=copy)
    xtype = x.dtype.type
    isscalar = (x.ndim == 0)
    if not issubclass(xtype, _nx.inexact):
        return x[()] if isscalar else x
    iscomplex = issubclass(xtype, _nx.complexfloating)
    dest = (x.real, x.imag) if iscomplex else (x,)
    maxf, minf = _getmaxmin(x.real.dtype)
    if posinf is not None:
        maxf = posinf
    if neginf is not None:
        minf = neginf
    for d in dest:
        idx_nan = isnan(d)
        idx_posinf = isposinf(d)
        idx_neginf = isneginf(d)
        _nx.copyto(d, nan, where=idx_nan)
        _nx.copyto(d, maxf, where=idx_posinf)
        _nx.copyto(d, minf, where=idx_neginf)
    return x[()] if isscalar else x
def _real_if_close_dispatcher(a, tol=None):
    return (a,)
@array_function_dispatch(_real_if_close_dispatcher)
def real_if_close(a, tol=100):
    a = asanyarray(a)
    if not issubclass(a.dtype.type, _nx.complexfloating):
        return a
    if tol > 1:
        from numpy.core import getlimits
        f = getlimits.finfo(a.dtype.type)
        tol = f.eps * tol
    if _nx.all(_nx.absolute(a.imag) < tol):
        a = a.real
    return a
def _asscalar_dispatcher(a):
    warnings.warn('np.asscalar(a) is deprecated since NumPy v1.16, use '
                  'a.item() instead', DeprecationWarning, stacklevel=3)
    return (a,)
@array_function_dispatch(_asscalar_dispatcher)
def asscalar(a):
    return a.item()
_namefromtype = {'S1': 'character',
                 '?': 'bool',
                 'b': 'signed char',
                 'B': 'unsigned char',
                 'h': 'short',
                 'H': 'unsigned short',
                 'i': 'integer',
                 'I': 'unsigned integer',
                 'l': 'long integer',
                 'L': 'unsigned long integer',
                 'q': 'long long integer',
                 'Q': 'unsigned long long integer',
                 'f': 'single precision',
                 'd': 'double precision',
                 'g': 'long precision',
                 'F': 'complex single precision',
                 'D': 'complex double precision',
                 'G': 'complex long double precision',
                 'S': 'string',
                 'U': 'unicode',
                 'V': 'void',
                 'O': 'object'
                 }
@set_module('numpy')
def typename(char):
    return _namefromtype[char]
array_type = [[_nx.half, _nx.single, _nx.double, _nx.longdouble],
              [None, _nx.csingle, _nx.cdouble, _nx.clongdouble]]
array_precision = {_nx.half: 0,
                   _nx.single: 1,
                   _nx.double: 2,
                   _nx.longdouble: 3,
                   _nx.csingle: 1,
                   _nx.cdouble: 2,
                   _nx.clongdouble: 3}
def _common_type_dispatcher(*arrays):
    return arrays
@array_function_dispatch(_common_type_dispatcher)
def common_type(*arrays):
    is_complex = False
    precision = 0
    for a in arrays:
        t = a.dtype.type
        if iscomplexobj(a):
            is_complex = True
        if issubclass(t, _nx.integer):
            p = 2
        else:
            p = array_precision.get(t, None)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]
