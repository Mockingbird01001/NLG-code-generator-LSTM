
__all__ = ['fix', 'isneginf', 'isposinf']
import numpy.core.numeric as nx
from numpy.core.overrides import (
    array_function_dispatch, ARRAY_FUNCTION_ENABLED,
)
import warnings
import functools
def _deprecate_out_named_y(f):
    @functools.wraps(f)
    def func(x, out=None, **kwargs):
        if 'y' in kwargs:
            if 'out' in kwargs:
                raise TypeError(
                    "{} got multiple values for argument 'out'/'y'"
                    .format(f.__name__)
                )
            out = kwargs.pop('y')
            warnings.warn(
                "The name of the out argument to {} has changed from `y` to "
                "`out`, to match other ufuncs.".format(f.__name__),
                DeprecationWarning, stacklevel=3)
        return f(x, out=out, **kwargs)
    return func
def _fix_out_named_y(f):
    @functools.wraps(f)
    def func(x, out=None, **kwargs):
        if 'y' in kwargs:
            out = kwargs.pop('y')
        return f(x, out=out, **kwargs)
    return func
def _fix_and_maybe_deprecate_out_named_y(f):
    if ARRAY_FUNCTION_ENABLED:
        return _fix_out_named_y(f)
    else:
        return _deprecate_out_named_y(f)
@_deprecate_out_named_y
def _dispatcher(x, out=None):
    return (x, out)
@array_function_dispatch(_dispatcher, verify=False, module='numpy')
@_fix_and_maybe_deprecate_out_named_y
def fix(x, out=None):
    res = nx.asanyarray(nx.ceil(x, out=out))
    res = nx.floor(x, out=res, where=nx.greater_equal(x, 0))
    if out is None and type(res) is nx.ndarray:
        res = res[()]
    return res
@array_function_dispatch(_dispatcher, verify=False, module='numpy')
@_fix_and_maybe_deprecate_out_named_y
def isposinf(x, out=None):
    is_inf = nx.isinf(x)
    try:
        signbit = ~nx.signbit(x)
    except TypeError as e:
        raise TypeError('This operation is not supported for complex values '
                        'because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)
@array_function_dispatch(_dispatcher, verify=False, module='numpy')
@_fix_and_maybe_deprecate_out_named_y
def isneginf(x, out=None):
    is_inf = nx.isinf(x)
    try:
        signbit = nx.signbit(x)
    except TypeError as e:
        raise TypeError('This operation is not supported for complex values '
                        'because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)
