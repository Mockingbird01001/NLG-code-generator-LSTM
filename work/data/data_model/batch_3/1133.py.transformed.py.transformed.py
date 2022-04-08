
import functools
from numpy.core.numeric import (
    asanyarray, arange, zeros, greater_equal, multiply, ones,
    asarray, where, int8, int16, int32, int64, intp, empty, promote_types,
    diagonal, nonzero, indices
    )
from numpy.core.overrides import set_array_function_like_doc, set_module
from numpy.core import overrides
from numpy.core import iinfo
__all__ = [
    'diag', 'diagflat', 'eye', 'fliplr', 'flipud', 'tri', 'triu',
    'tril', 'vander', 'histogram2d', 'mask_indices', 'tril_indices',
    'tril_indices_from', 'triu_indices', 'triu_indices_from', ]
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')
i1 = iinfo(int8)
i2 = iinfo(int16)
i4 = iinfo(int32)
def _min_int(low, high):
    if high <= i1.max and low >= i1.min:
        return int8
    if high <= i2.max and low >= i2.min:
        return int16
    if high <= i4.max and low >= i4.min:
        return int32
    return int64
def _flip_dispatcher(m):
    return (m,)
@array_function_dispatch(_flip_dispatcher)
def fliplr(m):
    m = asanyarray(m)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return m[:, ::-1]
@array_function_dispatch(_flip_dispatcher)
def flipud(m):
    m = asanyarray(m)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return m[::-1, ...]
def _eye_dispatcher(N, M=None, k=None, dtype=None, order=None, *, like=None):
    return (like,)
@set_array_function_like_doc
@set_module('numpy')
def eye(N, M=None, k=0, dtype=float, order='C', *, like=None):
    if like is not None:
        return _eye_with_like(N, M=M, k=k, dtype=dtype, order=order, like=like)
    if M is None:
        M = N
    m = zeros((N, M), dtype=dtype, order=order)
    if k >= M:
        return m
    if k >= 0:
        i = k
    else:
        i = (-k) * M
    m[:M-k].flat[i::M+1] = 1
    return m
_eye_with_like = array_function_dispatch(
    _eye_dispatcher
)(eye)
def _diag_dispatcher(v, k=None):
    return (v,)
@array_function_dispatch(_diag_dispatcher)
def diag(v, k=0):
    v = asanyarray(v)
    s = v.shape
    if len(s) == 1:
        n = s[0]+abs(k)
        res = zeros((n, n), v.dtype)
        if k >= 0:
            i = k
        else:
            i = (-k) * n
        res[:n-k].flat[i::n+1] = v
        return res
    elif len(s) == 2:
        return diagonal(v, k)
    else:
        raise ValueError("Input must be 1- or 2-d.")
@array_function_dispatch(_diag_dispatcher)
def diagflat(v, k=0):
    try:
        wrap = v.__array_wrap__
    except AttributeError:
        wrap = None
    v = asarray(v).ravel()
    s = len(v)
    n = s + abs(k)
    res = zeros((n, n), v.dtype)
    if (k >= 0):
        i = arange(0, n-k, dtype=intp)
        fi = i+k+i*n
    else:
        i = arange(0, n+k, dtype=intp)
        fi = i+(i-k)*n
    res.flat[fi] = v
    if not wrap:
        return res
    return wrap(res)
def _tri_dispatcher(N, M=None, k=None, dtype=None, *, like=None):
    return (like,)
@set_array_function_like_doc
@set_module('numpy')
def tri(N, M=None, k=0, dtype=float, *, like=None):
    if like is not None:
        return _tri_with_like(N, M=M, k=k, dtype=dtype, like=like)
    if M is None:
        M = N
    m = greater_equal.outer(arange(N, dtype=_min_int(0, N)),
                            arange(-k, M-k, dtype=_min_int(-k, M - k)))
    m = m.astype(dtype, copy=False)
    return m
_tri_with_like = array_function_dispatch(
    _tri_dispatcher
)(tri)
def _trilu_dispatcher(m, k=None):
    return (m,)
@array_function_dispatch(_trilu_dispatcher)
def tril(m, k=0):
    m = asanyarray(m)
    mask = tri(*m.shape[-2:], k=k, dtype=bool)
    return where(mask, m, zeros(1, m.dtype))
@array_function_dispatch(_trilu_dispatcher)
def triu(m, k=0):
    m = asanyarray(m)
    mask = tri(*m.shape[-2:], k=k-1, dtype=bool)
    return where(mask, zeros(1, m.dtype), m)
def _vander_dispatcher(x, N=None, increasing=None):
    return (x,)
@array_function_dispatch(_vander_dispatcher)
def vander(x, N=None, increasing=False):
    x = asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array or sequence.")
    if N is None:
        N = len(x)
    v = empty((len(x), N), dtype=promote_types(x.dtype, int))
    tmp = v[:, ::-1] if not increasing else v
    if N > 0:
        tmp[:, 0] = 1
    if N > 1:
        tmp[:, 1:] = x[:, None]
        multiply.accumulate(tmp[:, 1:], out=tmp[:, 1:], axis=1)
    return v
def _histogram2d_dispatcher(x, y, bins=None, range=None, normed=None,
                            weights=None, density=None):
    yield x
    yield y
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N == 2:
        yield from bins
    else:
        yield bins
    yield weights
@array_function_dispatch(_histogram2d_dispatcher)
def histogram2d(x, y, bins=10, range=None, normed=None, weights=None,
                density=None):
    from numpy import histogramdd
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N != 1 and N != 2:
        xedges = yedges = asarray(bins)
        bins = [xedges, yedges]
    hist, edges = histogramdd([x, y], bins, range, normed, weights, density)
    return hist, edges[0], edges[1]
@set_module('numpy')
def mask_indices(n, mask_func, k=0):
    m = ones((n, n), int)
    a = mask_func(m, k)
    return nonzero(a != 0)
@set_module('numpy')
def tril_indices(n, k=0, m=None):
    return nonzero(tri(n, m, k=k, dtype=bool))
def _trilu_indices_form_dispatcher(arr, k=None):
    return (arr,)
@array_function_dispatch(_trilu_indices_form_dispatcher)
def tril_indices_from(arr, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])
@set_module('numpy')
def triu_indices(n, k=0, m=None):
    return nonzero(~tri(n, m, k=k-1, dtype=bool))
@array_function_dispatch(_trilu_indices_form_dispatcher)
def triu_indices_from(arr, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])
