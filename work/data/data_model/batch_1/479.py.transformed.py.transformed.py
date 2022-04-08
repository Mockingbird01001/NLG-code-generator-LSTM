import collections.abc
import functools
import re
import sys
import warnings
import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
    ones, zeros, arange, concatenate, array, asarray, asanyarray, empty,
    ndarray, around, floor, ceil, take, dot, where, intp,
    integer, isscalar, absolute
    )
from numpy.core.umath import (
    pi, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin,
    mod, exp, not_equal, subtract
    )
from numpy.core.fromnumeric import (
    ravel, nonzero, partition, mean, any, sum
    )
from numpy.core.numerictypes import typecodes
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
    _insert, add_docstring, bincount, normalize_axis_index, _monotonicity,
    interp as compiled_interp, interp_complex as compiled_interp_complex
    )
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
import builtins
from numpy.lib.histograms import histogram, histogramdd
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')
__all__ = [
    'select', 'piecewise', 'trim_zeros', 'copy', 'iterable', 'percentile',
    'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'disp', 'flip',
    'rot90', 'extract', 'place', 'vectorize', 'asarray_chkfinite', 'average',
    'bincount', 'digitize', 'cov', 'corrcoef',
    'msort', 'median', 'sinc', 'hamming', 'hanning', 'bartlett',
    'blackman', 'kaiser', 'trapz', 'i0', 'add_newdoc', 'add_docstring',
    'meshgrid', 'delete', 'insert', 'append', 'interp', 'add_newdoc_ufunc',
    'quantile'
    ]
def _rot90_dispatcher(m, k=None, axes=None):
    return (m,)
@array_function_dispatch(_rot90_dispatcher)
def rot90(m, k=1, axes=(0, 1)):
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    m = asanyarray(m)
    if axes[0] == axes[1] or absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")
    if (axes[0] >= m.ndim or axes[0] < -m.ndim
        or axes[1] >= m.ndim or axes[1] < -m.ndim):
        raise ValueError("Axes={} out of range for array of ndim={}."
            .format(axes, m.ndim))
    k %= 4
    if k == 0:
        return m[:]
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])
    axes_list = arange(0, m.ndim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],
                                                axes_list[axes[0]])
    if k == 1:
        return transpose(flip(m, axes[1]), axes_list)
    else:
        return flip(transpose(m, axes_list), axes[1])
def _flip_dispatcher(m, axis=None):
    return (m,)
@array_function_dispatch(_flip_dispatcher)
def flip(m, axis=None):
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    if axis is None:
        indexer = (np.s_[::-1],) * m.ndim
    else:
        axis = _nx.normalize_axis_tuple(axis, m.ndim)
        indexer = [np.s_[:]] * m.ndim
        for ax in axis:
            indexer[ax] = np.s_[::-1]
        indexer = tuple(indexer)
    return m[indexer]
@set_module('numpy')
def iterable(y):
    try:
        iter(y)
    except TypeError:
        return False
    return True
def _average_dispatcher(a, axis=None, weights=None, returned=None):
    return (a, weights)
@array_function_dispatch(_average_dispatcher)
def average(a, axis=None, weights=None, returned=False):
    a = np.asanyarray(a)
    if weights is None:
        avg = a.mean(axis)
        scl = avg.dtype.type(a.size/avg.size)
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
        scl = wgt.sum(axis=axis, dtype=result_dtype)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")
        avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis)/scl
    if returned:
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg
@set_module('numpy')
def asarray_chkfinite(a, dtype=None, order=None):
    a = asarray(a, dtype=dtype, order=order)
    if a.dtype.char in typecodes['AllFloat'] and not np.isfinite(a).all():
        raise ValueError(
            "array must not contain infs or NaNs")
    return a
def _piecewise_dispatcher(x, condlist, funclist, *args, **kw):
    yield x
    if np.iterable(condlist):
        yield from condlist
@array_function_dispatch(_piecewise_dispatcher)
def piecewise(x, condlist, funclist, *args, **kw):
    x = asanyarray(x)
    n2 = len(funclist)
    if isscalar(condlist) or (
            not isinstance(condlist[0], (list, ndarray)) and x.ndim != 0):
        condlist = [condlist]
    condlist = array(condlist, dtype=bool)
    n = len(condlist)
    if n == n2 - 1:
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected"
            .format(n, n, n+1)
        )
    y = zeros(x.shape, x.dtype)
    for cond, func in zip(condlist, funclist):
        if not isinstance(func, collections.abc.Callable):
            y[cond] = func
        else:
            vals = x[cond]
            if vals.size > 0:
                y[cond] = func(vals, *args, **kw)
    return y
def _select_dispatcher(condlist, choicelist, default=None):
    yield from condlist
    yield from choicelist
@array_function_dispatch(_select_dispatcher)
def select(condlist, choicelist, default=0):
    if len(condlist) != len(choicelist):
        raise ValueError(
            'list of cases must be same length as list of conditions')
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")
    choicelist = [np.asarray(choice) for choice in choicelist]
    choicelist.append(np.asarray(default))
    dtype = np.result_type(*choicelist)
    condlist = np.broadcast_arrays(*condlist)
    choicelist = np.broadcast_arrays(*choicelist)
    for i, cond in enumerate(condlist):
        if cond.dtype.type is not np.bool_:
            raise TypeError(
                'invalid entry {} in condlist: should be boolean ndarray'.format(i))
    if choicelist[0].ndim == 0:
        result_shape = condlist[0].shape
    else:
        result_shape = np.broadcast_arrays(condlist[0], choicelist[0])[0].shape
    result = np.full(result_shape, choicelist[-1], dtype)
    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        np.copyto(result, choice, where=cond)
    return result
def _copy_dispatcher(a, order=None, subok=None):
    return (a,)
@array_function_dispatch(_copy_dispatcher)
def copy(a, order='K', subok=False):
    return array(a, order=order, subok=subok, copy=True)
def _gradient_dispatcher(f, *varargs, axis=None, edge_order=None):
    yield f
    yield from varargs
@array_function_dispatch(_gradient_dispatcher)
def gradient(f, *varargs, axis=None, edge_order=1):
    f = np.asanyarray(f)
    N = f.ndim
    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _nx.normalize_axis_tuple(axis, N)
    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        dx = varargs * len_axes
    elif n == len_axes:
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = np.asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            if np.issubdtype(distances.dtype, np.integer):
                distances = distances.astype(np.float64)
            diffx = np.diff(distances)
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")
    outvals = []
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N
    otype = f.dtype
    if otype.type is np.datetime64:
        otype = np.dtype(otype.name.replace('datetime', 'timedelta'))
        f = f.view(otype)
    elif otype.type is np.timedelta64:
        pass
    elif np.issubdtype(otype, np.inexact):
        pass
    else:
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64
    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required.")
        out = np.empty_like(f, dtype=otype)
        uniform_spacing = np.ndim(ax_dx) == 0
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2)/(dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            a.shape = b.shape = c.shape = shape
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0
            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2. / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = - dx1 / (dx2 * (dx1 + dx2))
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2. / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = - (dx2 + dx1) / (dx1 * dx2)
                c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        outvals.append(out)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)
    if len_axes == 1:
        return outvals[0]
    else:
        return outvals
def _diff_dispatcher(a, n=None, axis=None, prepend=None, append=None):
    return (a, prepend, append)
@array_function_dispatch(_diff_dispatcher)
def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))
    a = asanyarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = normalize_axis_index(axis, nd)
    combined = []
    if prepend is not np._NoValue:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)
    combined.append(a)
    if append is not np._NoValue:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)
    if len(combined) > 1:
        a = np.concatenate(combined, axis)
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    op = not_equal if a.dtype == np.bool_ else subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])
    return a
def _interp_dispatcher(x, xp, fp, left=None, right=None, period=None):
    return (x, xp, fp)
@array_function_dispatch(_interp_dispatcher)
def interp(x, xp, fp, left=None, right=None, period=None):
    fp = np.asarray(fp)
    if np.iscomplexobj(fp):
        interp_func = compiled_interp_complex
        input_dtype = np.complex128
    else:
        interp_func = compiled_interp
        input_dtype = np.float64
    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None
        x = np.asarray(x, dtype=np.float64)
        xp = np.asarray(xp, dtype=np.float64)
        fp = np.asarray(fp, dtype=input_dtype)
        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError("Data points must be 1-D sequences")
        if xp.shape[0] != fp.shape[0]:
            raise ValueError("fp and xp are not of the same length")
        x = x % period
        xp = xp % period
        asort_xp = np.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = np.concatenate((xp[-1:]-period, xp, xp[0:1]+period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))
    return interp_func(x, xp, fp, left, right)
def _angle_dispatcher(z, deg=None):
    return (z,)
@array_function_dispatch(_angle_dispatcher)
def angle(z, deg=False):
    z = asanyarray(z)
    if issubclass(z.dtype.type, _nx.complexfloating):
        zimag = z.imag
        zreal = z.real
    else:
        zimag = 0
        zreal = z
    a = arctan2(zimag, zreal)
    if deg:
        a *= 180/pi
    return a
def _unwrap_dispatcher(p, discont=None, axis=None):
    return (p,)
@array_function_dispatch(_unwrap_dispatcher)
def unwrap(p, discont=pi, axis=-1):
    p = asarray(p)
    nd = p.ndim
    dd = diff(p, axis=axis)
    slice1 = [slice(None, None)]*nd
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    ddmod = mod(dd + pi, 2*pi) - pi
    _nx.copyto(ddmod, pi, where=(ddmod == -pi) & (dd > 0))
    ph_correct = ddmod - dd
    _nx.copyto(ph_correct, 0, where=abs(dd) < discont)
    up = array(p, copy=True, dtype='d')
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up
def _sort_complex(a):
    return (a,)
@array_function_dispatch(_sort_complex)
def sort_complex(a):
    b = array(a, copy=True)
    b.sort()
    if not issubclass(b.dtype.type, _nx.complexfloating):
        if b.dtype.char in 'bhBH':
            return b.astype('F')
        elif b.dtype.char == 'g':
            return b.astype('G')
        else:
            return b.astype('D')
    else:
        return b
def _trim_zeros(filt, trim=None):
    return (filt,)
@array_function_dispatch(_trim_zeros)
def trim_zeros(filt, trim='fb'):
    first = 0
    trim = trim.upper()
    if 'F' in trim:
        for i in filt:
            if i != 0.:
                break
            else:
                first = first + 1
    last = len(filt)
    if 'B' in trim:
        for i in filt[::-1]:
            if i != 0.:
                break
            else:
                last = last - 1
    return filt[first:last]
def _extract_dispatcher(condition, arr):
    return (condition, arr)
@array_function_dispatch(_extract_dispatcher)
def extract(condition, arr):
    return _nx.take(ravel(arr), nonzero(ravel(condition))[0])
def _place_dispatcher(arr, mask, vals):
    return (arr, mask, vals)
@array_function_dispatch(_place_dispatcher)
def place(arr, mask, vals):
    if not isinstance(arr, np.ndarray):
        raise TypeError("argument 1 must be numpy.ndarray, "
                        "not {name}".format(name=type(arr).__name__))
    return _insert(arr, mask, vals)
def disp(mesg, device=None, linefeed=True):
    if device is None:
        device = sys.stdout
    if linefeed:
        device.write('%s\n' % mesg)
    else:
        device.write('%s' % mesg)
    device.flush()
    return
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)
def _parse_gufunc_signature(signature):
    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            'not a valid gufunc signature: {}'.format(signature))
    return tuple([tuple(re.findall(_DIMENSION_NAME, arg))
                  for arg in re.findall(_ARGUMENT, arg_list)]
                 for arg_list in signature.split('->'))
def _update_dim_sizes(dim_sizes, arg, core_dims):
    if not core_dims:
        return
    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError(
            '%d-dimensional argument does not have enough '
            'dimensions for all core dimensions %r'
            % (arg.ndim, core_dims))
    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError(
                    'inconsistent size for core dimension %r: %r vs %r'
                    % (dim, size, dim_sizes[dim]))
        else:
            dim_sizes[dim] = size
def _parse_input_dimensions(args, input_core_dims):
    broadcast_args = []
    dim_sizes = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib.stride_tricks._broadcast_shape(*broadcast_args)
    return broadcast_shape, dim_sizes
def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    return [broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
            for core_dims in list_of_core_dims]
def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes):
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    arrays = tuple(np.empty(shape, dtype=dtype)
                   for shape, dtype in zip(shapes, dtypes))
    return arrays
@set_module('numpy')
class vectorize:
    def __init__(self, pyfunc, otypes=None, doc=None, excluded=None,
                 cache=False, signature=None):
        self.pyfunc = pyfunc
        self.cache = cache
        self.signature = signature
        self._ufunc = {}
        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc
        if isinstance(otypes, str):
            for char in otypes:
                if char not in typecodes['All']:
                    raise ValueError("Invalid otype specified: %s" % (char,))
        elif iterable(otypes):
            otypes = ''.join([_nx.dtype(x).char for x in otypes])
        elif otypes is not None:
            raise ValueError("Invalid otype specification")
        self.otypes = otypes
        if excluded is None:
            excluded = set()
        self.excluded = set(excluded)
        if signature is not None:
            self._in_and_out_core_dims = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dims = None
    def __call__(self, *args, **kwargs):
        excluded = self.excluded
        if not kwargs and not excluded:
            func = self.pyfunc
            vargs = args
        else:
            nargs = len(args)
            names = [_n for _n in kwargs if _n not in excluded]
            inds = [_i for _i in range(nargs) if _i not in excluded]
            the_args = list(args)
            def func(*vargs):
                for _n, _i in enumerate(inds):
                    the_args[_i] = vargs[_n]
                kwargs.update(zip(names, vargs[len(inds):]))
                return self.pyfunc(*the_args, **kwargs)
            vargs = [args[_i] for _i in inds]
            vargs.extend([kwargs[_n] for _n in names])
        return self._vectorize_call(func=func, args=vargs)
    def _get_ufunc_and_otypes(self, func, args):
        if not args:
            raise ValueError('args can not be empty')
        if self.otypes is not None:
            otypes = self.otypes
            nin = len(args)
            nout = len(self.otypes)
            if func is not self.pyfunc or nin not in self._ufunc:
                ufunc = frompyfunc(func, nin, nout)
            else:
                ufunc = None
            if func is self.pyfunc:
                ufunc = self._ufunc.setdefault(nin, ufunc)
        else:
            args = [asarray(arg) for arg in args]
            if builtins.any(arg.size == 0 for arg in args):
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')
            inputs = [arg.flat[0] for arg in args]
            outputs = func(*inputs)
            if self.cache:
                _cache = [outputs]
                def _func(*vargs):
                    if _cache:
                        return _cache.pop()
                    else:
                        return func(*vargs)
            else:
                _func = func
            if isinstance(outputs, tuple):
                nout = len(outputs)
            else:
                nout = 1
                outputs = (outputs,)
            otypes = ''.join([asarray(outputs[_k]).dtype.char
                              for _k in range(nout)])
            ufunc = frompyfunc(_func, len(args), nout)
        return ufunc, otypes
    def _vectorize_call(self, func, args):
        if self.signature is not None:
            res = self._vectorize_call_with_signature(func, args)
        elif not args:
            res = func()
        else:
            ufunc, otypes = self._get_ufunc_and_otypes(func=func, args=args)
            inputs = [array(a, copy=False, subok=True, dtype=object)
                      for a in args]
            outputs = ufunc(*inputs)
            if ufunc.nout == 1:
                res = array(outputs, copy=False, subok=True, dtype=otypes[0])
            else:
                res = tuple([array(x, copy=False, subok=True, dtype=t)
                             for x, t in zip(outputs, otypes)])
        return res
    def _vectorize_call_with_signature(self, func, args):
        input_core_dims, output_core_dims = self._in_and_out_core_dims
        if len(args) != len(input_core_dims):
            raise TypeError('wrong number of positional arguments: '
                            'expected %r, got %r'
                            % (len(input_core_dims), len(args)))
        args = tuple(asanyarray(arg) for arg in args)
        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args, input_core_dims)
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes,
                                         input_core_dims)
        args = [np.broadcast_to(arg, shape, subok=True)
                for arg, shape in zip(args, input_shapes)]
        outputs = None
        otypes = self.otypes
        nout = len(output_core_dims)
        for index in np.ndindex(*broadcast_shape):
            results = func(*(arg[index] for arg in args))
            n_results = len(results) if isinstance(results, tuple) else 1
            if nout != n_results:
                raise ValueError(
                    'wrong number of outputs from pyfunc: expected %r, got %r'
                    % (nout, n_results))
            if nout == 1:
                results = (results,)
            if outputs is None:
                for result, core_dims in zip(results, output_core_dims):
                    _update_dim_sizes(dim_sizes, result, core_dims)
                if otypes is None:
                    otypes = [asarray(result).dtype for result in results]
                outputs = _create_arrays(broadcast_shape, dim_sizes,
                                         output_core_dims, otypes)
            for output, result in zip(outputs, results):
                output[index] = result
        if outputs is None:
            if otypes is None:
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')
            if builtins.any(dim not in dim_sizes
                            for dims in output_core_dims
                            for dim in dims):
                raise ValueError('cannot call `vectorize` with a signature '
                                 'including new output dimensions on size 0 '
                                 'inputs')
            outputs = _create_arrays(broadcast_shape, dim_sizes,
                                     output_core_dims, otypes)
        return outputs[0] if nout == 1 else outputs
def _cov_dispatcher(m, y=None, rowvar=None, bias=None, ddof=None,
                    fweights=None, aweights=None, *, dtype=None):
    return (m, y, fweights, aweights)
@array_function_dispatch(_cov_dispatcher)
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None, *, dtype=None):
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")
    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")
    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)
    X = array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)
    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights
    avg, w_sum = average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof*sum(w*aweights)/w_sum
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=3)
        fact = 0.0
    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X*w).T
    c = dot(X, X_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()
def _corrcoef_dispatcher(x, y=None, rowvar=None, bias=None, ddof=None, *,
                         dtype=None):
    return (x, y)
@array_function_dispatch(_corrcoef_dispatcher)
def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue, *,
             dtype=None):
    if bias is not np._NoValue or ddof is not np._NoValue:
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning, stacklevel=3)
    c = cov(x, y, rowvar, dtype=dtype)
    try:
        d = diag(c)
    except ValueError:
        return c / c
    stddev = sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)
    return c
@set_module('numpy')
def blackman(M):
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(1-M, M, 2)
    return 0.42 + 0.5*cos(pi*n/(M-1)) + 0.08*cos(2.0*pi*n/(M-1))
@set_module('numpy')
def bartlett(M):
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(1-M, M, 2)
    return where(less_equal(n, 0), 1 + n/(M-1), 1 - n/(M-1))
@set_module('numpy')
def hanning(M):
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(1-M, M, 2)
    return 0.5 + 0.5*cos(pi*n/(M-1))
@set_module('numpy')
def hamming(M):
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(1-M, M, 2)
    return 0.54 + 0.46*cos(pi*n/(M-1))
_i0A = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
    ]
_i0B = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
    ]
def _chbevl(x, vals):
    b0 = vals[0]
    b1 = 0.0
    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + vals[i]
    return 0.5*(b0 - b2)
def _i0_1(x):
    return exp(x) * _chbevl(x/2.0-2, _i0A)
def _i0_2(x):
    return exp(x) * _chbevl(32.0/x - 2.0, _i0B) / sqrt(x)
def _i0_dispatcher(x):
    return (x,)
@array_function_dispatch(_i0_dispatcher)
def i0(x):
    x = np.asanyarray(x)
    if x.dtype.kind == 'c':
        raise TypeError("i0 not supported for complex values")
    if x.dtype.kind != 'f':
        x = x.astype(float)
    x = np.abs(x)
    return piecewise(x, [x <= 8.0], [_i0_1, _i0_2])
@set_module('numpy')
def kaiser(M, beta):
    if M == 1:
        return np.array([1.])
    n = arange(0, M)
    alpha = (M-1)/2.0
    return i0(beta * sqrt(1-((n-alpha)/alpha)**2.0))/i0(float(beta))
def _sinc_dispatcher(x):
    return (x,)
@array_function_dispatch(_sinc_dispatcher)
def sinc(x):
    x = np.asanyarray(x)
    y = pi * where(x == 0, 1.0e-20, x)
    return sin(y)/y
def _msort_dispatcher(a):
    return (a,)
@array_function_dispatch(_msort_dispatcher)
def msort(a):
    b = array(a, subok=True, copy=True)
    b.sort(0)
    return b
def _ureduce(a, func, **kwargs):
    a = np.asanyarray(a)
    axis = kwargs.get('axis', None)
    if axis is not None:
        keepdim = list(a.shape)
        nd = a.ndim
        axis = _nx.normalize_axis_tuple(axis, nd)
        for ax in axis:
            keepdim[ax] = 1
        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
        keepdim = tuple(keepdim)
    else:
        keepdim = (1,) * a.ndim
    r = func(a, **kwargs)
    return r, keepdim
def _median_dispatcher(
        a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)
@array_function_dispatch(_median_dispatcher)
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    r, k = _ureduce(a, func=_median, axis=axis, out=out,
                    overwrite_input=overwrite_input)
    if keepdims:
        return r.reshape(k)
    else:
        return r
def _median(a, axis=None, out=None, overwrite_input=False):
    a = np.asanyarray(a)
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]
    if np.issubdtype(a.dtype, np.inexact):
        kth.append(-1)
    if overwrite_input:
        if axis is None:
            part = a.ravel()
            part.partition(kth)
        else:
            a.partition(kth, axis=axis)
            part = a
    else:
        part = partition(a, kth, axis=axis)
    if part.shape == ():
        return part.item()
    if axis is None:
        axis = 0
    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        indexer[axis] = slice(index, index+1)
    else:
        indexer[axis] = slice(index-1, index+1)
    indexer = tuple(indexer)
    if np.issubdtype(a.dtype, np.inexact) and sz > 0:
        rout = mean(part[indexer], axis=axis, out=out)
        return np.lib.utils._median_nancheck(part, rout, axis, out)
    else:
        return mean(part[indexer], axis=axis, out=out)
def _percentile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                           interpolation=None, keepdims=None):
    return (a, q, out)
@array_function_dispatch(_percentile_dispatcher)
def percentile(a, q, axis=None, out=None,
               overwrite_input=False, interpolation='linear', keepdims=False):
    q = np.true_divide(q, 100)
    q = asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, interpolation, keepdims)
def _quantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                         interpolation=None, keepdims=None):
    return (a, q, out)
@array_function_dispatch(_quantile_dispatcher)
def quantile(a, q, axis=None, out=None,
             overwrite_input=False, interpolation='linear', keepdims=False):
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, interpolation, keepdims)
def _quantile_unchecked(a, q, axis=None, out=None, overwrite_input=False,
                        interpolation='linear', keepdims=False):
    r, k = _ureduce(a, func=_quantile_ureduce_func, q=q, axis=axis, out=out,
                    overwrite_input=overwrite_input,
                    interpolation=interpolation)
    if keepdims:
        return r.reshape(q.shape + k)
    else:
        return r
def _quantile_is_valid(q):
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if q[i] < 0.0 or q[i] > 1.0:
                return False
    else:
        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            return False
    return True
def _lerp(a, b, t, out=None):
    diff_b_a = subtract(b, a)
    lerp_interpolation = asanyarray(add(a, diff_b_a*t, out=out))
    subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t>=0.5)
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]
    return lerp_interpolation
def _quantile_ureduce_func(a, q, axis=None, out=None, overwrite_input=False,
                           interpolation='linear', keepdims=False):
    a = asarray(a)
    not_scalar = np.asanyarray
    if overwrite_input:
        if axis is None:
            ap = a.ravel()
        else:
            ap = a
    else:
        if axis is None:
            ap = a.flatten()
        else:
            ap = a.copy()
    if axis is None:
        axis = 0
    if q.ndim > 2:
        raise ValueError("q must be a scalar or 1d")
    Nx = ap.shape[axis]
    indices = not_scalar(q * (Nx - 1))
    if interpolation == 'lower':
        indices = floor(indices).astype(intp)
    elif interpolation == 'higher':
        indices = ceil(indices).astype(intp)
    elif interpolation == 'midpoint':
        indices = 0.5 * (floor(indices) + ceil(indices))
    elif interpolation == 'nearest':
        indices = around(indices).astype(intp)
    elif interpolation == 'linear':
        pass
    else:
        raise ValueError(
            "interpolation can only be 'linear', 'lower' 'higher', "
            "'midpoint', or 'nearest'")
    ap = np.moveaxis(ap, axis, 0)
    del axis
    if np.issubdtype(indices.dtype, np.integer):
        if np.issubdtype(a.dtype, np.inexact):
            ap.partition(concatenate((indices.ravel(), [-1])), axis=0)
            n = np.isnan(ap[-1])
        else:
            ap.partition(indices.ravel(), axis=0)
            n = np.array(False, dtype=bool)
        r = take(ap, indices, axis=0, out=out)
    else:
        indices_below = not_scalar(floor(indices)).astype(intp)
        indices_above = not_scalar(indices_below + 1)
        indices_above[indices_above > Nx - 1] = Nx - 1
        if np.issubdtype(a.dtype, np.inexact):
            ap.partition(concatenate((
                indices_below.ravel(), indices_above.ravel(), [-1]
            )), axis=0)
            n = np.isnan(ap[-1])
        else:
            ap.partition(concatenate((
                indices_below.ravel(), indices_above.ravel()
            )), axis=0)
            n = np.array(False, dtype=bool)
        weights_shape = indices.shape + (1,) * (ap.ndim - 1)
        weights_above = not_scalar(indices - indices_below).reshape(weights_shape)
        x_below = take(ap, indices_below, axis=0)
        x_above = take(ap, indices_above, axis=0)
        r = _lerp(x_below, x_above, weights_above, out=out)
    if np.any(n):
        if r.ndim == 0 and out is None:
            r = a.dtype.type(np.nan)
        else:
            r[..., n] = a.dtype.type(np.nan)
    return r
def _trapz_dispatcher(y, x=None, dx=None, axis=None):
    return (y, x)
@array_function_dispatch(_trapz_dispatcher)
def trapz(y, x=None, dx=1.0, axis=-1):
    y = asanyarray(y)
    if x is None:
        d = dx
    else:
        x = asanyarray(x)
        if x.ndim == 1:
            d = diff(x)
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        d = np.asarray(d)
        y = np.asarray(y)
        ret = add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
    return ret
def _meshgrid_dispatcher(*xi, copy=None, sparse=None, indexing=None):
    return xi
@array_function_dispatch(_meshgrid_dispatcher)
def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    ndim = len(xi)
    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")
    s0 = (1,) * ndim
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
              for i, x in enumerate(xi)]
    if indexing == 'xy' and ndim > 1:
        output[0].shape = (1, -1) + s0[2:]
        output[1].shape = (-1, 1) + s0[2:]
    if not sparse:
        output = np.broadcast_arrays(*output, subok=True)
    if copy:
        output = [x.copy() for x in output]
    return output
def _delete_dispatcher(arr, obj, axis=None):
    return (arr, obj)
@array_function_dispatch(_delete_dispatcher)
def delete(arr, obj, axis=None):
    wrap = None
    if type(arr) is not ndarray:
        try:
            wrap = arr.__array_wrap__
        except AttributeError:
            pass
    arr = asarray(arr)
    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)
    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)
    if isinstance(obj, slice):
        start, stop, step = obj.indices(N)
        xr = range(start, stop, step)
        numtodel = len(xr)
        if numtodel <= 0:
            if wrap:
                return wrap(arr.copy(order=arrorder))
            else:
                return arr.copy(order=arrorder)
        if step < 0:
            step = -step
            start = xr[-1]
            stop = xr[0] + 1
        newshape[axis] -= numtodel
        new = empty(newshape, arr.dtype, arrorder)
        if start == 0:
            pass
        else:
            slobj[axis] = slice(None, start)
            new[tuple(slobj)] = arr[tuple(slobj)]
        if stop == N:
            pass
        else:
            slobj[axis] = slice(stop-numtodel, None)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(stop, None)
            new[tuple(slobj)] = arr[tuple(slobj2)]
        if step == 1:
            pass
        else:
            keep = ones(stop-start, dtype=bool)
            keep[:stop-start:step] = False
            slobj[axis] = slice(start, stop-numtodel)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(start, stop)
            arr = arr[tuple(slobj2)]
            slobj2[axis] = keep
            new[tuple(slobj)] = arr[tuple(slobj2)]
        if wrap:
            return wrap(new)
        else:
            return new
    if isinstance(obj, (int, integer)) and not isinstance(obj, bool):
        if (obj < -N or obj >= N):
            raise IndexError(
                "index %i is out of bounds for axis %i with "
                "size %i" % (obj, axis, N))
        if (obj < 0):
            obj += N
        newshape[axis] -= 1
        new = empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, obj)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(obj, None)
        slobj2 = [slice(None)]*ndim
        slobj2[axis] = slice(obj+1, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]
    else:
        _obj = obj
        obj = np.asarray(obj)
        if obj.size == 0 and not isinstance(_obj, np.ndarray):
            obj = obj.astype(intp)
        if obj.dtype == bool:
            if obj.shape != (N,):
                raise ValueError('boolean array argument obj to delete '
                                 'must be one dimensional and match the axis '
                                 'length of {}'.format(N))
            keep = ~obj
        else:
            keep = ones(N, dtype=bool)
            keep[obj,] = False
        slobj[axis] = keep
        new = arr[tuple(slobj)]
    if wrap:
        return wrap(new)
    else:
        return new
def _insert_dispatcher(arr, obj, values, axis=None):
    return (arr, obj, values)
@array_function_dispatch(_insert_dispatcher)
def insert(arr, obj, values, axis=None):
    wrap = None
    if type(arr) is not ndarray:
        try:
            wrap = arr.__array_wrap__
        except AttributeError:
            pass
    arr = asarray(arr)
    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)
    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)
    if isinstance(obj, slice):
        indices = arange(*obj.indices(N), dtype=intp)
    else:
        indices = np.array(obj)
        if indices.dtype == bool:
            warnings.warn(
                "in the future insert will treat boolean arrays and "
                "array-likes as a boolean index instead of casting it to "
                "integer", FutureWarning, stacklevel=3)
            indices = indices.astype(intp)
        elif indices.ndim > 1:
            raise ValueError(
                "index array argument obj to insert must be one dimensional "
                "or scalar")
    if indices.size == 1:
        index = indices.item()
        if index < -N or index > N:
            raise IndexError(
                "index %i is out of bounds for axis %i with "
                "size %i" % (obj, axis, N))
        if (index < 0):
            index += N
        values = array(values, copy=False, ndmin=arr.ndim, dtype=arr.dtype)
        if indices.ndim == 0:
            values = np.moveaxis(values, 0, axis)
        numnew = values.shape[axis]
        newshape[axis] += numnew
        new = empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(index, index+numnew)
        new[tuple(slobj)] = values
        slobj[axis] = slice(index+numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]
        if wrap:
            return wrap(new)
        return new
    elif indices.size == 0 and not isinstance(obj, np.ndarray):
        indices = indices.astype(intp)
    indices[indices < 0] += N
    numnew = len(indices)
    order = indices.argsort(kind='mergesort')
    indices[order] += np.arange(numnew)
    newshape[axis] += numnew
    old_mask = ones(newshape[axis], dtype=bool)
    old_mask[indices] = False
    new = empty(newshape, arr.dtype, arrorder)
    slobj2 = [slice(None)]*ndim
    slobj[axis] = indices
    slobj2[axis] = old_mask
    new[tuple(slobj)] = values
    new[tuple(slobj2)] = arr
    if wrap:
        return wrap(new)
    return new
def _append_dispatcher(arr, values, axis=None):
    return (arr, values)
@array_function_dispatch(_append_dispatcher)
def append(arr, values, axis=None):
    arr = asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = ravel(values)
        axis = arr.ndim-1
    return concatenate((arr, values), axis=axis)
def _digitize_dispatcher(x, bins, right=None):
    return (x, bins)
@array_function_dispatch(_digitize_dispatcher)
def digitize(x, bins, right=False):
    x = _nx.asarray(x)
    bins = _nx.asarray(bins)
    if np.issubdtype(x.dtype, _nx.complexfloating):
        raise TypeError("x may not be complex")
    mono = _monotonicity(bins)
    if mono == 0:
        raise ValueError("bins must be monotonically increasing or decreasing")
    side = 'left' if right else 'right'
    if mono == -1:
        return len(bins) - _nx.searchsorted(bins[::-1], x, side=side)
    else:
        return _nx.searchsorted(bins, x, side=side)
