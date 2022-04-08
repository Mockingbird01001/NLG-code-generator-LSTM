
import functools
import warnings
from . import overrides
from . import _multiarray_umath
from ._multiarray_umath import *
from ._multiarray_umath import (
    _fastCopyAndTranspose, _flagdict, _insert, _reconstruct, _vec_string,
    _ARRAY_API, _monotonicity, _get_ndarray_c_version, _set_madvise_hugepage,
    )
__all__ = [
    '_ARRAY_API', 'ALLOW_THREADS', 'BUFSIZE', 'CLIP', 'DATETIMEUNITS',
    'ITEM_HASOBJECT', 'ITEM_IS_POINTER', 'LIST_PICKLE', 'MAXDIMS',
    'MAY_SHARE_BOUNDS', 'MAY_SHARE_EXACT', 'NEEDS_INIT', 'NEEDS_PYAPI',
    'RAISE', 'USE_GETITEM', 'USE_SETITEM', 'WRAP', '_fastCopyAndTranspose',
    '_flagdict', '_insert', '_reconstruct', '_vec_string', '_monotonicity',
    'add_docstring', 'arange', 'array', 'bincount', 'broadcast',
    'busday_count', 'busday_offset', 'busdaycalendar', 'can_cast',
    'compare_chararrays', 'concatenate', 'copyto', 'correlate', 'correlate2',
    'count_nonzero', 'c_einsum', 'datetime_as_string', 'datetime_data',
    'digitize', 'dot', 'dragon4_positional', 'dragon4_scientific', 'dtype',
    'empty', 'empty_like', 'error', 'flagsobj', 'flatiter', 'format_longfloat',
    'frombuffer', 'fromfile', 'fromiter', 'fromstring', 'inner',
    'interp', 'interp_complex', 'is_busday', 'lexsort',
    'matmul', 'may_share_memory', 'min_scalar_type', 'ndarray', 'nditer',
    'nested_iters', 'normalize_axis_index', 'packbits',
    'promote_types', 'putmask', 'ravel_multi_index', 'result_type', 'scalar',
    'set_datetimeparse_function', 'set_legacy_print_mode', 'set_numeric_ops',
    'set_string_function', 'set_typeDict', 'shares_memory',
    'tracemalloc_domain', 'typeinfo', 'unpackbits', 'unravel_index', 'vdot',
    'where', 'zeros']
_reconstruct.__module__ = 'numpy.core.multiarray'
scalar.__module__ = 'numpy.core.multiarray'
arange.__module__ = 'numpy'
array.__module__ = 'numpy'
datetime_data.__module__ = 'numpy'
empty.__module__ = 'numpy'
frombuffer.__module__ = 'numpy'
fromfile.__module__ = 'numpy'
fromiter.__module__ = 'numpy'
frompyfunc.__module__ = 'numpy'
fromstring.__module__ = 'numpy'
geterrobj.__module__ = 'numpy'
may_share_memory.__module__ = 'numpy'
nested_iters.__module__ = 'numpy'
promote_types.__module__ = 'numpy'
set_numeric_ops.__module__ = 'numpy'
seterrobj.__module__ = 'numpy'
zeros.__module__ = 'numpy'
array_function_from_c_func_and_dispatcher = functools.partial(
    overrides.array_function_from_dispatcher,
    module='numpy', docs_from_dispatcher=True, verify=False)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.empty_like)
def empty_like(prototype, dtype=None, order=None, subok=None, shape=None):
    return (prototype,)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.concatenate)
def concatenate(arrays, axis=None, out=None, *, dtype=None, casting=None):
    if out is not None:
        arrays = list(arrays)
        arrays.append(out)
    return arrays
@array_function_from_c_func_and_dispatcher(_multiarray_umath.inner)
def inner(a, b):
    return (a, b)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.where)
def where(condition, x=None, y=None):
    return (condition, x, y)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.lexsort)
def lexsort(keys, axis=None):
    if isinstance(keys, tuple):
        return keys
    else:
        return (keys,)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.can_cast)
def can_cast(from_, to, casting=None):
    return (from_,)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.min_scalar_type)
def min_scalar_type(a):
    return (a,)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.result_type)
def result_type(*arrays_and_dtypes):
    return arrays_and_dtypes
@array_function_from_c_func_and_dispatcher(_multiarray_umath.dot)
def dot(a, b, out=None):
    return (a, b, out)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.vdot)
def vdot(a, b):
    return (a, b)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.bincount)
def bincount(x, weights=None, minlength=None):
    return (x, weights)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.ravel_multi_index)
def ravel_multi_index(multi_index, dims, mode=None, order=None):
    return multi_index
@array_function_from_c_func_and_dispatcher(_multiarray_umath.unravel_index)
def unravel_index(indices, shape=None, order=None, dims=None):
    if dims is not None:
        warnings.warn("'shape' argument should be used instead of 'dims'",
                      DeprecationWarning, stacklevel=3)
    return (indices,)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.copyto)
def copyto(dst, src, casting=None, where=None):
    return (dst, src, where)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.putmask)
def putmask(a, mask, values):
    return (a, mask, values)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.packbits)
def packbits(a, axis=None, bitorder='big'):
    return (a,)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.unpackbits)
def unpackbits(a, axis=None, count=None, bitorder='big'):
    return (a,)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.shares_memory)
def shares_memory(a, b, max_work=None):
    return (a, b)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.may_share_memory)
def may_share_memory(a, b, max_work=None):
    return (a, b)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.is_busday)
def is_busday(dates, weekmask=None, holidays=None, busdaycal=None, out=None):
    return (dates, weekmask, holidays, out)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_offset)
def busday_offset(dates, offsets, roll=None, weekmask=None, holidays=None,
                  busdaycal=None, out=None):
    return (dates, offsets, weekmask, holidays, out)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_count)
def busday_count(begindates, enddates, weekmask=None, holidays=None,
                 busdaycal=None, out=None):
    return (begindates, enddates, weekmask, holidays, out)
@array_function_from_c_func_and_dispatcher(
    _multiarray_umath.datetime_as_string)
def datetime_as_string(arr, unit=None, timezone=None, casting=None):
    return (arr,)
