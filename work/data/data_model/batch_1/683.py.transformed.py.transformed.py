
import warnings
from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core._asarray import asanyarray
from numpy.core import numerictypes as nt
from numpy.core import _exceptions
from numpy._globals import _NoValue
from numpy.compat import pickle, os_fspath, contextlib_nullcontext
umr_maximum = um.maximum.reduce
umr_minimum = um.minimum.reduce
umr_sum = um.add.reduce
umr_prod = um.multiply.reduce
umr_any = um.logical_or.reduce
umr_all = um.logical_and.reduce
_complex_to_float = {
    nt.dtype(nt.csingle) : nt.dtype(nt.single),
    nt.dtype(nt.cdouble) : nt.dtype(nt.double),
}
if nt.dtype(nt.longdouble) != nt.dtype(nt.double):
    _complex_to_float.update({
        nt.dtype(nt.clongdouble) : nt.dtype(nt.longdouble),
    })
def _amax(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_maximum(a, axis, None, out, keepdims, initial, where)
def _amin(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
         initial=_NoValue, where=True):
    return umr_sum(a, axis, dtype, out, keepdims, initial, where)
def _prod(a, axis=None, dtype=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_prod(a, axis, dtype, out, keepdims, initial, where)
def _any(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    if where is True:
        return umr_any(a, axis, dtype, out, keepdims)
    return umr_any(a, axis, dtype, out, keepdims, where=where)
def _all(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    if where is True:
        return umr_all(a, axis, dtype, out, keepdims)
    return umr_all(a, axis, dtype, out, keepdims, where=where)
def _count_reduce_items(arr, axis, keepdims=False, where=True):
    if where is True:
        if axis is None:
            axis = tuple(range(arr.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)
        items = nt.intp(1)
        for ax in axis:
            items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
    else:
        from numpy.lib.stride_tricks import broadcast_to
        items = umr_sum(broadcast_to(where, arr.shape), axis, nt.intp, None,
                        keepdims)
    return items
def _clip_dep_is_scalar_nan(a):
    from numpy.core.fromnumeric import ndim
    if ndim(a) != 0:
        return False
    try:
        return um.isnan(a)
    except TypeError:
        return False
def _clip_dep_is_byte_swapped(a):
    if isinstance(a, mu.ndarray):
        return not a.dtype.isnative
    return False
def _clip_dep_invoke_with_casting(ufunc, *args, out=None, casting=None, **kwargs):
    if casting is not None:
        return ufunc(*args, out=out, casting=casting, **kwargs)
    try:
        return ufunc(*args, out=out, **kwargs)
    except _exceptions._UFuncOutputCastingError as e:
        warnings.warn(
            "Converting the output of clip from {!r} to {!r} is deprecated. "
            "Pass `casting=\"unsafe\"` explicitly to silence this warning, or "
            "correct the type of the variables.".format(e.from_, e.to),
            DeprecationWarning,
            stacklevel=2
        )
        return ufunc(*args, out=out, casting="unsafe", **kwargs)
def _clip(a, min=None, max=None, out=None, *, casting=None, **kwargs):
    if min is None and max is None:
        raise ValueError("One of max or min must be given")
    if not _clip_dep_is_byte_swapped(a) and not _clip_dep_is_byte_swapped(out):
        using_deprecated_nan = False
        if _clip_dep_is_scalar_nan(min):
            min = -float('inf')
            using_deprecated_nan = True
        if _clip_dep_is_scalar_nan(max):
            max = float('inf')
            using_deprecated_nan = True
        if using_deprecated_nan:
            warnings.warn(
                "Passing `np.nan` to mean no clipping in np.clip has always "
                "been unreliable, and is now deprecated. "
                "In future, this will always return nan, like it already does "
                "when min or max are arrays that contain nan. "
                "To skip a bound, pass either None or an np.inf of an "
                "appropriate sign.",
                DeprecationWarning,
                stacklevel=2
            )
    if min is None:
        return _clip_dep_invoke_with_casting(
            um.minimum, a, max, out=out, casting=casting, **kwargs)
    elif max is None:
        return _clip_dep_invoke_with_casting(
            um.maximum, a, min, out=out, casting=casting, **kwargs)
    else:
        return _clip_dep_invoke_with_casting(
            um.clip, a, min, max, out=out, casting=casting, **kwargs)
def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    arr = asanyarray(a)
    is_float16_result = False
    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    if rcount == 0 if where is True else umr_any(rcount == 0, axis=None):
        warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)
    if dtype is None:
        if issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
            dtype = mu.dtype('f8')
        elif issubclass(arr.dtype.type, nt.float16):
            dtype = mu.dtype('f4')
            is_float16_result = True
    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
        if is_float16_result and out is None:
            ret = arr.dtype.type(ret)
    elif hasattr(ret, 'dtype'):
        if is_float16_result:
            ret = arr.dtype.type(ret / rcount)
        else:
            ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount
    return ret
def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True):
    arr = asanyarray(a)
    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    if ddof >= rcount if where is True else umr_any(ddof >= rcount, axis=None):
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                      stacklevel=2)
    if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
        dtype = mu.dtype('f8')
    arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where)
    if rcount.ndim == 0:
        div = rcount
    else:
        div = rcount.reshape(arrmean.shape)
    if isinstance(arrmean, mu.ndarray):
        arrmean = um.true_divide(arrmean, div, out=arrmean, casting='unsafe',
                                 subok=False)
    else:
        arrmean = arrmean.dtype.type(arrmean / rcount)
    x = asanyarray(arr - arrmean)
    if issubclass(arr.dtype.type, (nt.floating, nt.integer)):
        x = um.multiply(x, x, out=x)
    elif x.dtype in _complex_to_float:
        xv = x.view(dtype=(_complex_to_float[x.dtype], (2,)))
        um.multiply(xv, xv, out=xv)
        x = um.add(xv[..., 0], xv[..., 1], out=x.real).real
    else:
        x = um.multiply(x, um.conjugate(x), out=x).real
    ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where)
    rcount = um.maximum(rcount - ddof, 0)
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount
    return ret
def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
               keepdims=keepdims, where=where)
    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        ret = um.sqrt(ret)
    return ret
def _ptp(a, axis=None, out=None, keepdims=False):
    return um.subtract(
        umr_maximum(a, axis, None, out, keepdims),
        umr_minimum(a, axis, None, None, keepdims),
        out
    )
def _dump(self, file, protocol=2):
    if hasattr(file, 'write'):
        ctx = contextlib_nullcontext(file)
    else:
        ctx = open(os_fspath(file), "wb")
    with ctx as f:
        pickle.dump(self, f, protocol=protocol)
def _dumps(self, protocol=2):
    return pickle.dumps(self, protocol=protocol)
