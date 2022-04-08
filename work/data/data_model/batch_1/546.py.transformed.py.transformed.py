import functools
import sys
import math
import warnings
import numpy.core.numeric as _nx
from numpy.core.numeric import (
    asarray, ScalarType, array, alltrue, cumprod, arange, ndim
    )
from numpy.core.numerictypes import find_common_type, issubdtype
import numpy.matrixlib as matrixlib
from .function_base import diff
from numpy.core.multiarray import ravel_multi_index, unravel_index
from numpy.core.overrides import set_module
from numpy.core import overrides, linspace
from numpy.lib.stride_tricks import as_strided
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')
__all__ = [
    'ravel_multi_index', 'unravel_index', 'mgrid', 'ogrid', 'r_', 'c_',
    's_', 'index_exp', 'ix_', 'ndenumerate', 'ndindex', 'fill_diagonal',
    'diag_indices', 'diag_indices_from'
    ]
def _ix__dispatcher(*args):
    return args
@array_function_dispatch(_ix__dispatcher)
def ix_(*args):
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if not isinstance(new, _nx.ndarray):
            new = asarray(new)
            if new.size == 0:
                new = new.astype(_nx.intp)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if issubdtype(new.dtype, _nx.bool_):
            new, = new.nonzero()
        new = new.reshape((1,)*k + (new.size,) + (1,)*(nd-k-1))
        out.append(new)
    return tuple(out)
class nd_grid:
    def __init__(self, sparse=False):
        self.sparse = sparse
    def __getitem__(self, key):
        try:
            size = []
            typ = int
            for k in range(len(key)):
                step = key[k].step
                start = key[k].start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    size.append(int(abs(step)))
                    typ = float
                else:
                    size.append(
                        int(math.ceil((key[k].stop - start)/(step*1.0))))
                if (isinstance(step, (_nx.floating, float)) or
                        isinstance(start, (_nx.floating, float)) or
                        isinstance(key[k].stop, (_nx.floating, float))):
                    typ = float
            if self.sparse:
                nn = [_nx.arange(_x, dtype=_t)
                        for _x, _t in zip(size, (typ,)*len(size))]
            else:
                nn = _nx.indices(size, typ)
            for k in range(len(size)):
                step = key[k].step
                start = key[k].start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = int(abs(step))
                    if step != 1:
                        step = (key[k].stop - start)/float(step-1)
                nn[k] = (nn[k]*step+start)
            if self.sparse:
                slobj = [_nx.newaxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None, None)
                    nn[k] = nn[k][tuple(slobj)]
                    slobj[k] = _nx.newaxis
            return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, (_nx.complexfloating, complex)):
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (key.stop-start)/float(step-1)
                stop = key.stop + step
                return _nx.arange(0, length, 1, float)*step + start
            else:
                return _nx.arange(start, stop, step)
class MGridClass(nd_grid):
    def __init__(self):
        super(MGridClass, self).__init__(sparse=False)
mgrid = MGridClass()
class OGridClass(nd_grid):
    def __init__(self):
        super(OGridClass, self).__init__(sparse=True)
ogrid = OGridClass()
class AxisConcatenator:
    concatenate = staticmethod(_nx.concatenate)
    makemat = staticmethod(matrixlib.matrix)
    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self.axis = axis
        self.matrix = matrix
        self.trans1d = trans1d
        self.ndmin = ndmin
    def __getitem__(self, key):
        if isinstance(key, str):
            frame = sys._getframe().f_back
            mymat = matrixlib.bmat(key, frame.f_globals, frame.f_locals)
            return mymat
        if not isinstance(key, tuple):
            key = (key,)
        trans1d = self.trans1d
        ndmin = self.ndmin
        matrix = self.matrix
        axis = self.axis
        objs = []
        scalars = []
        arraytypes = []
        scalartypes = []
        for k, item in enumerate(key):
            scalar = False
            if isinstance(item, slice):
                step = item.step
                start = item.start
                stop = item.stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    size = int(abs(step))
                    newobj = linspace(start, stop, num=size)
                else:
                    newobj = _nx.arange(start, stop, step)
                if ndmin > 1:
                    newobj = array(newobj, copy=False, ndmin=ndmin)
                    if trans1d != -1:
                        newobj = newobj.swapaxes(-1, trans1d)
            elif isinstance(item, str):
                if k != 0:
                    raise ValueError("special directives must be the "
                            "first entry.")
                if item in ('r', 'c'):
                    matrix = True
                    col = (item == 'c')
                    continue
                if ',' in item:
                    vec = item.split(',')
                    try:
                        axis, ndmin = [int(x) for x in vec[:2]]
                        if len(vec) == 3:
                            trans1d = int(vec[2])
                        continue
                    except Exception as e:
                        raise ValueError(
                            "unknown special directive {!r}".format(item)
                        ) from e
                try:
                    axis = int(item)
                    continue
                except (ValueError, TypeError):
                    raise ValueError("unknown special directive")
            elif type(item) in ScalarType:
                newobj = array(item, ndmin=ndmin)
                scalars.append(len(objs))
                scalar = True
                scalartypes.append(newobj.dtype)
            else:
                item_ndim = ndim(item)
                newobj = array(item, copy=False, subok=True, ndmin=ndmin)
                if trans1d != -1 and item_ndim < ndmin:
                    k2 = ndmin - item_ndim
                    k1 = trans1d
                    if k1 < 0:
                        k1 += k2 + 1
                    defaxes = list(range(ndmin))
                    axes = defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]
                    newobj = newobj.transpose(axes)
            objs.append(newobj)
            if not scalar and isinstance(newobj, _nx.ndarray):
                arraytypes.append(newobj.dtype)
        final_dtype = find_common_type(arraytypes, scalartypes)
        if final_dtype is not None:
            for k in scalars:
                objs[k] = objs[k].astype(final_dtype)
        res = self.concatenate(tuple(objs), axis=axis)
        if matrix:
            oldndim = res.ndim
            res = self.makemat(res)
            if oldndim == 1 and col:
                res = res.T
        return res
    def __len__(self):
        return 0
class RClass(AxisConcatenator):
    def __init__(self):
        AxisConcatenator.__init__(self, 0)
r_ = RClass()
class CClass(AxisConcatenator):
    def __init__(self):
        AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)
c_ = CClass()
@set_module('numpy')
class ndenumerate:
    def __init__(self, arr):
        self.iter = asarray(arr).flat
    def __next__(self):
        return self.iter.coords, next(self.iter)
    def __iter__(self):
        return self
@set_module('numpy')
class ndindex:
    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        x = as_strided(_nx.zeros(1), shape=shape,
                       strides=_nx.zeros_like(shape))
        self._it = _nx.nditer(x, flags=['multi_index', 'zerosize_ok'],
                              order='C')
    def __iter__(self):
        return self
    def ndincr(self):
        warnings.warn(
            "`ndindex.ndincr()` is deprecated, use `next(ndindex)` instead",
            DeprecationWarning, stacklevel=2)
        next(self)
    def __next__(self):
        next(self._it)
        return self._it.multi_index
class IndexExpression:
    def __init__(self, maketuple):
        self.maketuple = maketuple
    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item
index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)
def _fill_diagonal_dispatcher(a, val, wrap=None):
    return (a,)
@array_function_dispatch(_fill_diagonal_dispatcher)
def fill_diagonal(a, val, wrap=False):
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        if not alltrue(diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + (cumprod(a.shape[:-1])).sum()
    a.flat[:end:step] = val
@set_module('numpy')
def diag_indices(n, ndim=2):
    idx = arange(n)
    return (idx,) * ndim
def _diag_indices_from(arr):
    return (arr,)
@array_function_dispatch(_diag_indices_from)
def diag_indices_from(arr):
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    if not alltrue(diff(arr.shape) == 0):
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(arr.shape[0], arr.ndim)
