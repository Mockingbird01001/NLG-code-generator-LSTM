
import numpy.core.numeric as nx
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, any
from numpy.core.overrides import array_function_dispatch
from numpy.lib.type_check import isreal
__all__ = [
    'sqrt', 'log', 'log2', 'logn', 'log10', 'power', 'arccos', 'arcsin',
    'arctanh'
    ]
_ln2 = nx.log(2.0)
def _tocomplex(arr):
    if issubclass(arr.dtype.type, (nt.single, nt.byte, nt.short, nt.ubyte,
                                   nt.ushort, nt.csingle)):
        return arr.astype(nt.csingle)
    else:
        return arr.astype(nt.cdouble)
def _fix_real_lt_zero(x):
    x = asarray(x)
    if any(isreal(x) & (x < 0)):
        x = _tocomplex(x)
    return x
def _fix_int_lt_zero(x):
    x = asarray(x)
    if any(isreal(x) & (x < 0)):
        x = x * 1.0
    return x
def _fix_real_abs_gt_1(x):
    x = asarray(x)
    if any(isreal(x) & (abs(x) > 1)):
        x = _tocomplex(x)
    return x
def _unary_dispatcher(x):
    return (x,)
@array_function_dispatch(_unary_dispatcher)
def sqrt(x):
    x = _fix_real_lt_zero(x)
    return nx.sqrt(x)
@array_function_dispatch(_unary_dispatcher)
def log(x):
    x = _fix_real_lt_zero(x)
    return nx.log(x)
@array_function_dispatch(_unary_dispatcher)
def log10(x):
    x = _fix_real_lt_zero(x)
    return nx.log10(x)
def _logn_dispatcher(n, x):
    return (n, x,)
@array_function_dispatch(_logn_dispatcher)
def logn(n, x):
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return nx.log(x)/nx.log(n)
@array_function_dispatch(_unary_dispatcher)
def log2(x):
    x = _fix_real_lt_zero(x)
    return nx.log2(x)
def _power_dispatcher(x, p):
    return (x, p)
@array_function_dispatch(_power_dispatcher)
def power(x, p):
    x = _fix_real_lt_zero(x)
    p = _fix_int_lt_zero(p)
    return nx.power(x, p)
@array_function_dispatch(_unary_dispatcher)
def arccos(x):
    x = _fix_real_abs_gt_1(x)
    return nx.arccos(x)
@array_function_dispatch(_unary_dispatcher)
def arcsin(x):
    x = _fix_real_abs_gt_1(x)
    return nx.arcsin(x)
@array_function_dispatch(_unary_dispatcher)
def arctanh(x):
    x = _fix_real_abs_gt_1(x)
    return nx.arctanh(x)
