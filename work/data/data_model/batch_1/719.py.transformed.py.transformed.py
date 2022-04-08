
import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')
__all__ = [
    'nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean',
    'nanmedian', 'nanpercentile', 'nanvar', 'nanstd', 'nanprod',
    'nancumsum', 'nancumprod', 'nanquantile'
    ]
def _nan_mask(a, out=None):
    if a.dtype.kind not in 'fc':
        return True
    y = np.isnan(a, out=out)
    y = np.invert(y, out=y)
    return y
def _replace_nan(a, val):
    a = np.asanyarray(a)
    if a.dtype == np.object_:
        mask = np.not_equal(a, a, dtype=bool)
    elif issubclass(a.dtype.type, np.inexact):
        mask = np.isnan(a)
    else:
        mask = None
    if mask is not None:
        a = np.array(a, subok=True, copy=True)
        np.copyto(a, val, where=mask)
    return a, mask
def _copyto(a, val, mask):
    if isinstance(a, np.ndarray):
        np.copyto(a, val, where=mask, casting='unsafe')
    else:
        a = a.dtype.type(val)
    return a
def _remove_nan_1d(arr1d, overwrite_input=False):
    c = np.isnan(arr1d)
    s = np.nonzero(c)[0]
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=5)
        return arr1d[:0], True
    elif s.size == 0:
        return arr1d, overwrite_input
    else:
        if not overwrite_input:
            arr1d = arr1d.copy()
        enonan = arr1d[-s.size:][~c[-s.size:]]
        arr1d[s[:enonan.size]] = enonan
        return arr1d[:-s.size], True
def _divide_by_count(a, b, out=None):
    with np.errstate(invalid='ignore', divide='ignore'):
        if isinstance(a, np.ndarray):
            if out is None:
                return np.divide(a, b, out=a, casting='unsafe')
            else:
                return np.divide(a, b, out=out, casting='unsafe')
        else:
            if out is None:
                return a.dtype.type(a / b)
            else:
                return np.divide(a, b, out=out, casting='unsafe')
def _nanmin_dispatcher(a, axis=None, out=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_nanmin_dispatcher)
def nanmin(a, axis=None, out=None, keepdims=np._NoValue):
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if type(a) is np.ndarray and a.dtype != np.object_:
        res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning,
                          stacklevel=3)
    else:
        a, mask = _replace_nan(a, +np.inf)
        res = np.amin(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning,
                          stacklevel=3)
    return res
def _nanmax_dispatcher(a, axis=None, out=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_nanmax_dispatcher)
def nanmax(a, axis=None, out=None, keepdims=np._NoValue):
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if type(a) is np.ndarray and a.dtype != np.object_:
        res = np.fmax.reduce(a, axis=axis, out=out, **kwargs)
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning,
                          stacklevel=3)
    else:
        a, mask = _replace_nan(a, -np.inf)
        res = np.amax(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning,
                          stacklevel=3)
    return res
def _nanargmin_dispatcher(a, axis=None):
    return (a,)
@array_function_dispatch(_nanargmin_dispatcher)
def nanargmin(a, axis=None):
    a, mask = _replace_nan(a, np.inf)
    res = np.argmin(a, axis=axis)
    if mask is not None:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    return res
def _nanargmax_dispatcher(a, axis=None):
    return (a,)
@array_function_dispatch(_nanargmax_dispatcher)
def nanargmax(a, axis=None):
    a, mask = _replace_nan(a, -np.inf)
    res = np.argmax(a, axis=axis)
    if mask is not None:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    return res
def _nansum_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_nansum_dispatcher)
def nansum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    a, mask = _replace_nan(a, 0)
    return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
def _nanprod_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_nanprod_dispatcher)
def nanprod(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    a, mask = _replace_nan(a, 1)
    return np.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
def _nancumsum_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)
@array_function_dispatch(_nancumsum_dispatcher)
def nancumsum(a, axis=None, dtype=None, out=None):
    a, mask = _replace_nan(a, 0)
    return np.cumsum(a, axis=axis, dtype=dtype, out=out)
def _nancumprod_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)
@array_function_dispatch(_nancumprod_dispatcher)
def nancumprod(a, axis=None, dtype=None, out=None):
    a, mask = _replace_nan(a, 1)
    return np.cumprod(a, axis=axis, dtype=dtype, out=out)
def _nanmean_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_nanmean_dispatcher)
def nanmean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")
    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims)
    tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    avg = _divide_by_count(tot, cnt, out=out)
    isbad = (cnt == 0)
    if isbad.any():
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=3)
    return avg
def _nanmedian1d(arr1d, overwrite_input=False):
    arr1d, overwrite_input = _remove_nan_1d(arr1d,
                                            overwrite_input=overwrite_input)
    if arr1d.size == 0:
        return np.nan
    return np.median(arr1d, overwrite_input=overwrite_input)
def _nanmedian(a, axis=None, out=None, overwrite_input=False):
    if axis is None or a.ndim == 1:
        part = a.ravel()
        if out is None:
            return _nanmedian1d(part, overwrite_input)
        else:
            out[...] = _nanmedian1d(part, overwrite_input)
            return out
    else:
        if a.shape[axis] < 600:
            return _nanmedian_small(a, axis, out, overwrite_input)
        result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
        if out is not None:
            out[...] = result
        return result
def _nanmedian_small(a, axis=None, out=None, overwrite_input=False):
    a = np.ma.masked_array(a, np.isnan(a))
    m = np.ma.median(a, axis=axis, overwrite_input=overwrite_input)
    for i in range(np.count_nonzero(m.mask.ravel())):
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=4)
    if out is not None:
        out[...] = m.filled(np.nan)
        return out
    return m.filled(np.nan)
def _nanmedian_dispatcher(
        a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_nanmedian_dispatcher)
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=np._NoValue):
    a = np.asanyarray(a)
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)
    r, k = function_base._ureduce(a, func=_nanmedian, axis=axis, out=out,
                                  overwrite_input=overwrite_input)
    if keepdims and keepdims is not np._NoValue:
        return r.reshape(k)
    else:
        return r
def _nanpercentile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                              interpolation=None, keepdims=None):
    return (a, q, out)
@array_function_dispatch(_nanpercentile_dispatcher)
def nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
                  interpolation='linear', keepdims=np._NoValue):
    a = np.asanyarray(a)
    q = np.true_divide(q, 100.0)
    if not function_base._quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")
    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, interpolation, keepdims)
def _nanquantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                            interpolation=None, keepdims=None):
    return (a, q, out)
@array_function_dispatch(_nanquantile_dispatcher)
def nanquantile(a, q, axis=None, out=None, overwrite_input=False,
                interpolation='linear', keepdims=np._NoValue):
    a = np.asanyarray(a)
    q = np.asanyarray(q)
    if not function_base._quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, interpolation, keepdims)
def _nanquantile_unchecked(a, q, axis=None, out=None, overwrite_input=False,
                           interpolation='linear', keepdims=np._NoValue):
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)
    r, k = function_base._ureduce(
        a, func=_nanquantile_ureduce_func, q=q, axis=axis, out=out,
        overwrite_input=overwrite_input, interpolation=interpolation
    )
    if keepdims and keepdims is not np._NoValue:
        return r.reshape(q.shape + k)
    else:
        return r
def _nanquantile_ureduce_func(a, q, axis=None, out=None, overwrite_input=False,
                              interpolation='linear'):
    if axis is None or a.ndim == 1:
        part = a.ravel()
        result = _nanquantile_1d(part, q, overwrite_input, interpolation)
    else:
        result = np.apply_along_axis(_nanquantile_1d, axis, a, q,
                                     overwrite_input, interpolation)
        if q.ndim != 0:
            result = np.moveaxis(result, axis, 0)
    if out is not None:
        out[...] = result
    return result
def _nanquantile_1d(arr1d, q, overwrite_input=False, interpolation='linear'):
    arr1d, overwrite_input = _remove_nan_1d(arr1d,
        overwrite_input=overwrite_input)
    if arr1d.size == 0:
        return np.full(q.shape, np.nan)[()]
    return function_base._quantile_unchecked(
        arr1d, q, overwrite_input=overwrite_input, interpolation=interpolation)
def _nanvar_dispatcher(
        a, axis=None, dtype=None, out=None, ddof=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_nanvar_dispatcher)
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.var(arr, axis=axis, dtype=dtype, out=out, ddof=ddof,
                      keepdims=keepdims)
    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")
    if type(arr) is np.matrix:
        _keepdims = np._NoValue
    else:
        _keepdims = True
    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=_keepdims)
    avg = np.sum(arr, axis=axis, dtype=dtype, keepdims=_keepdims)
    avg = _divide_by_count(avg, cnt)
    np.subtract(arr, avg, out=arr, casting='unsafe')
    arr = _copyto(arr, 0, mask)
    if issubclass(arr.dtype.type, np.complexfloating):
        sqr = np.multiply(arr, arr.conj(), out=arr).real
    else:
        sqr = np.multiply(arr, arr, out=arr)
    var = np.sum(sqr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if var.ndim < cnt.ndim:
        cnt = cnt.squeeze(axis)
    dof = cnt - ddof
    var = _divide_by_count(var, dof)
    isbad = (dof <= 0)
    if np.any(isbad):
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning,
                      stacklevel=3)
        var = _copyto(var, np.nan, isbad)
    return var
def _nanstd_dispatcher(
        a, axis=None, dtype=None, out=None, ddof=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_nanstd_dispatcher)
def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                 keepdims=keepdims)
    if isinstance(var, np.ndarray):
        std = np.sqrt(var, out=var)
    else:
        std = var.dtype.type(np.sqrt(var))
    return std
