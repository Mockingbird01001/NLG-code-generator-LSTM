
__all__ = [
    'apply_along_axis', 'apply_over_axes', 'atleast_1d', 'atleast_2d',
    'atleast_3d', 'average', 'clump_masked', 'clump_unmasked',
    'column_stack', 'compress_cols', 'compress_nd', 'compress_rowcols',
    'compress_rows', 'count_masked', 'corrcoef', 'cov', 'diagflat', 'dot',
    'dstack', 'ediff1d', 'flatnotmasked_contiguous', 'flatnotmasked_edges',
    'hsplit', 'hstack', 'isin', 'in1d', 'intersect1d', 'mask_cols', 'mask_rowcols',
    'mask_rows', 'masked_all', 'masked_all_like', 'median', 'mr_',
    'notmasked_contiguous', 'notmasked_edges', 'polyfit', 'row_stack',
    'setdiff1d', 'setxor1d', 'stack', 'unique', 'union1d', 'vander', 'vstack',
    ]
import itertools
import warnings
from . import core as ma
from .core import (
    MaskedArray, MAError, add, array, asarray, concatenate, filled, count,
    getmask, getmaskarray, make_mask_descr, masked, masked_array, mask_or,
    nomask, ones, sort, zeros, getdata, get_masked_subclass, dot,
    mask_rowcols
    )
import numpy as np
from numpy import ndarray, array as nxarray
import numpy.core.umath as umath
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
def issequence(seq):
    return isinstance(seq, (ndarray, tuple, list))
def count_masked(arr, axis=None):
    m = getmaskarray(arr)
    return m.sum(axis)
def masked_all(shape, dtype=float):
    a = masked_array(np.empty(shape, dtype),
                     mask=np.ones(shape, make_mask_descr(dtype)))
    return a
def masked_all_like(arr):
    a = np.empty_like(arr).view(MaskedArray)
    a._mask = np.ones(a.shape, dtype=make_mask_descr(a.dtype))
    return a
class _fromnxfunction:
    def __init__(self, funcname):
        self.__name__ = funcname
        self.__doc__ = self.getdoc()
    def getdoc(self):
        npfunc = getattr(np, self.__name__, None)
        doc = getattr(npfunc, '__doc__', None)
        if doc:
            sig = self.__name__ + ma.get_object_signature(npfunc)
            doc = ma.doc_note(doc, "The function is applied to both the _data "
                                   "and the _mask, if any.")
            return '\n\n'.join((sig, doc))
        return
    def __call__(self, *args, **params):
        pass
class _fromnxfunction_single(_fromnxfunction):
    def __call__(self, x, *args, **params):
        func = getattr(np, self.__name__)
        if isinstance(x, ndarray):
            _d = func(x.__array__(), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            return masked_array(_d, mask=_m)
        else:
            _d = func(np.asarray(x), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            return masked_array(_d, mask=_m)
class _fromnxfunction_seq(_fromnxfunction):
    def __call__(self, x, *args, **params):
        func = getattr(np, self.__name__)
        _d = func(tuple([np.asarray(a) for a in x]), *args, **params)
        _m = func(tuple([getmaskarray(a) for a in x]), *args, **params)
        return masked_array(_d, mask=_m)
class _fromnxfunction_args(_fromnxfunction):
    def __call__(self, *args, **params):
        func = getattr(np, self.__name__)
        arrays = []
        args = list(args)
        while len(args) > 0 and issequence(args[0]):
            arrays.append(args.pop(0))
        res = []
        for x in arrays:
            _d = func(np.asarray(x), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            res.append(masked_array(_d, mask=_m))
        if len(arrays) == 1:
            return res[0]
        return res
class _fromnxfunction_allargs(_fromnxfunction):
    def __call__(self, *args, **params):
        func = getattr(np, self.__name__)
        res = []
        for x in args:
            _d = func(np.asarray(x), **params)
            _m = func(getmaskarray(x), **params)
            res.append(masked_array(_d, mask=_m))
        if len(args) == 1:
            return res[0]
        return res
atleast_1d = _fromnxfunction_allargs('atleast_1d')
atleast_2d = _fromnxfunction_allargs('atleast_2d')
atleast_3d = _fromnxfunction_allargs('atleast_3d')
vstack = row_stack = _fromnxfunction_seq('vstack')
hstack = _fromnxfunction_seq('hstack')
column_stack = _fromnxfunction_seq('column_stack')
dstack = _fromnxfunction_seq('dstack')
stack = _fromnxfunction_seq('stack')
hsplit = _fromnxfunction_single('hsplit')
diagflat = _fromnxfunction_single('diagflat')
def flatten_inplace(seq):
    k = 0
    while (k != len(seq)):
        while hasattr(seq[k], '__iter__'):
            seq[k:(k + 1)] = seq[k]
        k += 1
    return seq
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    arr = array(arr, copy=False, subok=True)
    nd = arr.ndim
    axis = normalize_axis_index(axis, nd)
    ind = [0] * (nd - 1)
    i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    indlist.remove(axis)
    i[axis] = slice(None, None)
    outshape = np.asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
    asscalar = np.isscalar(res)
    if not asscalar:
        try:
            len(res)
        except TypeError:
            asscalar = True
    dtypes = []
    if asscalar:
        dtypes.append(np.asarray(res).dtype)
        outarr = zeros(outshape, object)
        outarr[tuple(ind)] = res
        Ntot = np.product(outshape)
        k = 1
        while k < Ntot:
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(ind)] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    else:
        res = array(res, copy=False, subok=True)
        j = i.copy()
        j[axis] = ([slice(None, None)] * res.ndim)
        j.put(indlist, ind)
        Ntot = np.product(outshape)
        holdshape = outshape
        outshape = list(arr.shape)
        outshape[axis] = res.shape
        dtypes.append(asarray(res).dtype)
        outshape = flatten_inplace(outshape)
        outarr = zeros(outshape, object)
        outarr[tuple(flatten_inplace(j.tolist()))] = res
        k = 1
        while k < Ntot:
            ind[-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            j.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(flatten_inplace(j.tolist()))] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    max_dtypes = np.dtype(np.asarray(dtypes).max())
    if not hasattr(arr, '_mask'):
        result = np.asarray(outarr, dtype=max_dtypes)
    else:
        result = asarray(outarr, dtype=max_dtypes)
        result.fill_value = ma.default_fill_value(result)
    return result
apply_along_axis.__doc__ = np.apply_along_axis.__doc__
def apply_over_axes(func, a, axes):
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = ma.expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning "
                        "an array of the correct shape")
    return val
if apply_over_axes.__doc__ is not None:
    apply_over_axes.__doc__ = np.apply_over_axes.__doc__[
        :np.apply_over_axes.__doc__.find('Notes')].rstrip() +    """
    Examples
    --------
    >>> a = np.ma.arange(24).reshape(2,3,4)
    >>> a[:,0,1] = np.ma.masked
    >>> a[:,1,:] = np.ma.masked
    >>> a
    masked_array(
      data=[[[0, --, 2, 3],
             [--, --, --, --],
             [8, 9, 10, 11]],
            [[12, --, 14, 15],
             [--, --, --, --],
             [20, 21, 22, 23]]],
      mask=[[[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]],
            [[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]]],
      fill_value=999999)
    >>> np.ma.apply_over_axes(np.ma.sum, a, [0,2])
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)
    Tuple axis arguments to ufuncs are equivalent:
    >>> np.ma.sum(a, axis=(0,2)).reshape((1,-1,1))
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)
    """
def average(a, axis=None, weights=None, returned=False):
    a = asarray(a)
    m = getmask(a)
    if weights is None:
        avg = a.mean(axis)
        scl = avg.dtype.type(a.count(axis))
    else:
        wgt = np.asanyarray(weights)
        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")
            wgt = np.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)
        if m is not nomask:
            wgt = wgt*(~a.mask)
        scl = wgt.sum(axis=axis, dtype=result_dtype)
        avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis)/scl
    if returned:
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if not hasattr(a, 'mask'):
        m = np.median(getdata(a, subok=True), axis=axis,
                      out=out, overwrite_input=overwrite_input,
                      keepdims=keepdims)
        if isinstance(m, np.ndarray) and 1 <= m.ndim:
            return masked_array(m, copy=False)
        else:
            return m
    r, k = _ureduce(a, func=_median, axis=axis, out=out,
                    overwrite_input=overwrite_input)
    if keepdims:
        return r.reshape(k)
    else:
        return r
def _median(a, axis=None, out=None, overwrite_input=False):
    if np.issubdtype(a.dtype, np.inexact):
        fill_value = np.inf
    else:
        fill_value = None
    if overwrite_input:
        if axis is None:
            asorted = a.ravel()
            asorted.sort(fill_value=fill_value)
        else:
            a.sort(axis=axis, fill_value=fill_value)
            asorted = a
    else:
        asorted = sort(a, axis=axis, fill_value=fill_value)
    if axis is None:
        axis = 0
    else:
        axis = normalize_axis_index(axis, asorted.ndim)
    if asorted.shape[axis] == 0:
        indexer = [slice(None)] * asorted.ndim
        indexer[axis] = slice(0, 0)
        indexer = tuple(indexer)
        return np.ma.mean(asorted[indexer], axis=axis, out=out)
    if asorted.ndim == 1:
        counts = count(asorted)
        idx, odd = divmod(count(asorted), 2)
        mid = asorted[idx + odd - 1:idx + 1]
        if np.issubdtype(asorted.dtype, np.inexact) and asorted.size > 0:
            s = mid.sum(out=out)
            if not odd:
                s = np.true_divide(s, 2., casting='safe', out=out)
            s = np.lib.utils._median_nancheck(asorted, s, axis, out)
        else:
            s = mid.mean(out=out)
        if np.ma.is_masked(s) and not np.all(asorted.mask):
            return np.ma.minimum_fill_value(asorted)
        return s
    counts = count(asorted, axis=axis, keepdims=True)
    h = counts // 2
    odd = counts % 2 == 1
    l = np.where(odd, h, h-1)
    lh = np.concatenate([l,h], axis=axis)
    low_high = np.take_along_axis(asorted, lh, axis=axis)
    def replace_masked(s):
        if np.ma.is_masked(s):
            rep = (~np.all(asorted.mask, axis=axis, keepdims=True)) & s.mask
            s.data[rep] = np.ma.minimum_fill_value(asorted)
            s.mask[rep] = False
    replace_masked(low_high)
    if np.issubdtype(asorted.dtype, np.inexact):
        s = np.ma.sum(low_high, axis=axis, out=out)
        np.true_divide(s.data, 2., casting='unsafe', out=s.data)
        s = np.lib.utils._median_nancheck(asorted, s, axis, out)
    else:
        s = np.ma.mean(low_high, axis=axis, out=out)
    return s
def compress_nd(x, axis=None):
    x = asarray(x)
    m = getmask(x)
    if axis is None:
        axis = tuple(range(x.ndim))
    else:
        axis = normalize_axis_tuple(axis, x.ndim)
    if m is nomask or not m.any():
        return x._data
    if m.all():
        return nxarray([])
    data = x._data
    for ax in axis:
        axes = tuple(list(range(ax)) + list(range(ax + 1, x.ndim)))
        data = data[(slice(None),)*ax + (~m.any(axis=axes),)]
    return data
def compress_rowcols(x, axis=None):
    if asarray(x).ndim != 2:
        raise NotImplementedError("compress_rowcols works for 2D arrays only.")
    return compress_nd(x, axis=axis)
def compress_rows(a):
    a = asarray(a)
    if a.ndim != 2:
        raise NotImplementedError("compress_rows works for 2D arrays only.")
    return compress_rowcols(a, 0)
def compress_cols(a):
    a = asarray(a)
    if a.ndim != 2:
        raise NotImplementedError("compress_cols works for 2D arrays only.")
    return compress_rowcols(a, 1)
def mask_rows(a, axis=np._NoValue):
    if axis is not np._NoValue:
        warnings.warn(
            "The axis argument has always been ignored, in future passing it "
            "will raise TypeError", DeprecationWarning, stacklevel=2)
    return mask_rowcols(a, 0)
def mask_cols(a, axis=np._NoValue):
    if axis is not np._NoValue:
        warnings.warn(
            "The axis argument has always been ignored, in future passing it "
            "will raise TypeError", DeprecationWarning, stacklevel=2)
    return mask_rowcols(a, 1)
def ediff1d(arr, to_end=None, to_begin=None):
    arr = ma.asanyarray(arr).flat
    ed = arr[1:] - arr[:-1]
    arrays = [ed]
    if to_begin is not None:
        arrays.insert(0, to_begin)
    if to_end is not None:
        arrays.append(to_end)
    if len(arrays) != 1:
        ed = hstack(arrays)
    return ed
def unique(ar1, return_index=False, return_inverse=False):
    output = np.unique(ar1,
                       return_index=return_index,
                       return_inverse=return_inverse)
    if isinstance(output, tuple):
        output = list(output)
        output[0] = output[0].view(MaskedArray)
        output = tuple(output)
    else:
        output = output.view(MaskedArray)
    return output
def intersect1d(ar1, ar2, assume_unique=False):
    if assume_unique:
        aux = ma.concatenate((ar1, ar2))
    else:
        aux = ma.concatenate((unique(ar1), unique(ar2)))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]
def setxor1d(ar1, ar2, assume_unique=False):
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    aux = ma.concatenate((ar1, ar2))
    if aux.size == 0:
        return aux
    aux.sort()
    auxf = aux.filled()
    flag = ma.concatenate(([True], (auxf[1:] != auxf[:-1]), [True]))
    flag2 = (flag[1:] == flag[:-1])
    return aux[flag2]
def in1d(ar1, ar2, assume_unique=False, invert=False):
    if not assume_unique:
        ar1, rev_idx = unique(ar1, return_inverse=True)
        ar2 = unique(ar2)
    ar = ma.concatenate((ar1, ar2))
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = ma.concatenate((bool_ar, [invert]))
    indx = order.argsort(kind='mergesort')[:len(ar1)]
    if assume_unique:
        return flag[indx]
    else:
        return flag[indx][rev_idx]
def isin(element, test_elements, assume_unique=False, invert=False):
    element = ma.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)
def union1d(ar1, ar2):
    return unique(ma.concatenate((ar1, ar2), axis=None))
def setdiff1d(ar1, ar2, assume_unique=False):
    if assume_unique:
        ar1 = ma.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]
def _covhelper(x, y=None, rowvar=True, allow_masked=True):
    x = ma.array(x, ndmin=2, copy=True, dtype=float)
    xmask = ma.getmaskarray(x)
    if not allow_masked and xmask.any():
        raise ValueError("Cannot process masked data.")
    if x.shape[0] == 1:
        rowvar = True
    rowvar = int(bool(rowvar))
    axis = 1 - rowvar
    if rowvar:
        tup = (slice(None), None)
    else:
        tup = (None, slice(None))
    if y is None:
        xnotmask = np.logical_not(xmask).astype(int)
    else:
        y = array(y, copy=False, ndmin=2, dtype=float)
        ymask = ma.getmaskarray(y)
        if not allow_masked and ymask.any():
            raise ValueError("Cannot process masked data.")
        if xmask.any() or ymask.any():
            if y.shape == x.shape:
                common_mask = np.logical_or(xmask, ymask)
                if common_mask is not nomask:
                    xmask = x._mask = y._mask = ymask = common_mask
                    x._sharedmask = False
                    y._sharedmask = False
        x = ma.concatenate((x, y), axis)
        xnotmask = np.logical_not(np.concatenate((xmask, ymask), axis)).astype(int)
    x -= x.mean(axis=rowvar)[tup]
    return (x, xnotmask, rowvar)
def cov(x, y=None, rowvar=True, bias=False, allow_masked=True, ddof=None):
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be an integer")
    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1
    (x, xnotmask, rowvar) = _covhelper(x, y, rowvar, allow_masked)
    if not rowvar:
        fact = np.dot(xnotmask.T, xnotmask) * 1. - ddof
        result = (dot(x.T, x.conj(), strict=False) / fact).squeeze()
    else:
        fact = np.dot(xnotmask, xnotmask.T) * 1. - ddof
        result = (dot(x, x.T.conj(), strict=False) / fact).squeeze()
    return result
def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, allow_masked=True,
             ddof=np._NoValue):
    msg = 'bias and ddof have no effect and are deprecated'
    if bias is not np._NoValue or ddof is not np._NoValue:
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    (x, xnotmask, rowvar) = _covhelper(x, y, rowvar, allow_masked)
    if not rowvar:
        fact = np.dot(xnotmask.T, xnotmask) * 1.
        c = (dot(x.T, x.conj(), strict=False) / fact).squeeze()
    else:
        fact = np.dot(xnotmask, xnotmask.T) * 1.
        c = (dot(x, x.T.conj(), strict=False) / fact).squeeze()
    try:
        diag = ma.diagonal(c)
    except ValueError:
        return 1
    if xnotmask.all():
        _denom = ma.sqrt(ma.multiply.outer(diag, diag))
    else:
        _denom = diagflat(diag)
        _denom._sharedmask = False
        n = x.shape[1 - rowvar]
        if rowvar:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    _x = mask_cols(vstack((x[i], x[j]))).var(axis=1)
                    _denom[i, j] = _denom[j, i] = ma.sqrt(ma.multiply.reduce(_x))
        else:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    _x = mask_cols(
                            vstack((x[:, i], x[:, j]))).var(axis=1)
                    _denom[i, j] = _denom[j, i] = ma.sqrt(ma.multiply.reduce(_x))
    return c / _denom
class MAxisConcatenator(AxisConcatenator):
    concatenate = staticmethod(concatenate)
    @classmethod
    def makemat(cls, arr):
        data = super(MAxisConcatenator, cls).makemat(arr.data, copy=False)
        return array(data, mask=arr.mask)
    def __getitem__(self, key):
        if isinstance(key, str):
            raise MAError("Unavailable for masked array.")
        return super(MAxisConcatenator, self).__getitem__(key)
class mr_class(MAxisConcatenator):
    def __init__(self):
        MAxisConcatenator.__init__(self, 0)
mr_ = mr_class()
def flatnotmasked_edges(a):
    m = getmask(a)
    if m is nomask or not np.any(m):
        return np.array([0, a.size - 1])
    unmasked = np.flatnonzero(~m)
    if len(unmasked) > 0:
        return unmasked[[0, -1]]
    else:
        return None
def notmasked_edges(a, axis=None):
    a = asarray(a)
    if axis is None or a.ndim == 1:
        return flatnotmasked_edges(a)
    m = getmaskarray(a)
    idx = array(np.indices(a.shape), mask=np.asarray([m] * a.ndim))
    return [tuple([idx[i].min(axis).compressed() for i in range(a.ndim)]),
            tuple([idx[i].max(axis).compressed() for i in range(a.ndim)]), ]
def flatnotmasked_contiguous(a):
    m = getmask(a)
    if m is nomask:
        return [slice(0, a.size)]
    i = 0
    result = []
    for (k, g) in itertools.groupby(m.ravel()):
        n = len(list(g))
        if not k:
            result.append(slice(i, i + n))
        i += n
    return result
def notmasked_contiguous(a, axis=None):
    a = asarray(a)
    nd = a.ndim
    if nd > 2:
        raise NotImplementedError("Currently limited to atmost 2D array.")
    if axis is None or nd == 1:
        return flatnotmasked_contiguous(a)
    result = []
    other = (axis + 1) % 2
    idx = [0, 0]
    idx[axis] = slice(None, None)
    for i in range(a.shape[other]):
        idx[other] = i
        result.append(flatnotmasked_contiguous(a[tuple(idx)]))
    return result
def _ezclump(mask):
    if mask.ndim > 1:
        mask = mask.ravel()
    idx = (mask[1:] ^ mask[:-1]).nonzero()
    idx = idx[0] + 1
    if mask[0]:
        if len(idx) == 0:
            return [slice(0, mask.size)]
        r = [slice(0, idx[0])]
        r.extend((slice(left, right)
                  for left, right in zip(idx[1:-1:2], idx[2::2])))
    else:
        if len(idx) == 0:
            return []
        r = [slice(left, right) for left, right in zip(idx[:-1:2], idx[1::2])]
    if mask[-1]:
        r.append(slice(idx[-1], mask.size))
    return r
def clump_unmasked(a):
    mask = getattr(a, '_mask', nomask)
    if mask is nomask:
        return [slice(0, a.size)]
    return _ezclump(~mask)
def clump_masked(a):
    mask = ma.getmask(a)
    if mask is nomask:
        return []
    return _ezclump(mask)
def vander(x, n=None):
    _vander = np.vander(x, n)
    m = getmask(x)
    if m is not nomask:
        _vander[m] = 0
    return _vander
vander.__doc__ = ma.doc_note(np.vander.__doc__, vander.__doc__)
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    x = asarray(x)
    y = asarray(y)
    m = getmask(x)
    if y.ndim == 1:
        m = mask_or(m, getmask(y))
    elif y.ndim == 2:
        my = getmask(mask_rows(y))
        if my is not nomask:
            m = mask_or(m, my[:, 0])
    else:
        raise TypeError("Expected a 1D or 2D array for y!")
    if w is not None:
        w = asarray(w)
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        m = mask_or(m, getmask(w))
    if m is not nomask:
        not_m = ~m
        if w is not None:
            w = w[not_m]
        return np.polyfit(x[not_m], y[not_m], deg, rcond, full, w, cov)
    else:
        return np.polyfit(x, y, deg, rcond, full, w, cov)
polyfit.__doc__ = ma.doc_note(np.polyfit.__doc__, polyfit.__doc__)
