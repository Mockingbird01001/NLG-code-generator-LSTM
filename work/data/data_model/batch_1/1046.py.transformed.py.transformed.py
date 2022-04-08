
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch, set_module
__all__ = ['broadcast_to', 'broadcast_arrays', 'broadcast_shapes']
class DummyArray:
    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base
def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        new_array = new_array.view(type=type(original_array))
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array
def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    x = np.array(x, copy=False, subok=subok)
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)
    array = np.asarray(DummyArray(interface, base=x))
    array.dtype = x.dtype
    view = _maybe_view_as_subclass(x, array)
    if view.flags.writeable and not writeable:
        view.flags.writeable = False
    return view
def _sliding_window_view_dispatcher(x, window_shape, axis=None, *,
                                    subok=None, writeable=None):
    return (x,)
@array_function_dispatch(_sliding_window_view_dispatcher)
def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    x = np.array(x, copy=False, subok=subok)
    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')
    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')
    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)
def _broadcast_to(array, shape, subok, readonly):
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    array = np.array(array, copy=False, subok=subok)
    if not shape and array.shape:
        raise ValueError('cannot broadcast a non-scalar to a scalar array')
    if any(size < 0 for size in shape):
        raise ValueError('all elements of broadcast shape must be non-'
                         'negative')
    extras = []
    it = np.nditer(
        (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras,
        op_flags=['readonly'], itershape=shape, order='C')
    with it:
        broadcast = it.itviews[0]
    result = _maybe_view_as_subclass(array, broadcast)
    if not readonly and array.flags._writeable_no_warn:
        result.flags.writeable = True
        result.flags._warn_on_write = True
    return result
def _broadcast_to_dispatcher(array, shape, subok=None):
    return (array,)
@array_function_dispatch(_broadcast_to_dispatcher, module='numpy')
def broadcast_to(array, shape, subok=False):
    return _broadcast_to(array, shape, subok=subok, readonly=True)
def _broadcast_shape(*args):
    b = np.broadcast(*args[:32])
    for pos in range(32, len(args), 31):
        b = broadcast_to(0, b.shape)
        b = np.broadcast(b, *args[pos:(pos + 31)])
    return b.shape
@set_module('numpy')
def broadcast_shapes(*args):
    arrays = [np.empty(x, dtype=[]) for x in args]
    return _broadcast_shape(*arrays)
def _broadcast_arrays_dispatcher(*args, subok=None):
    return args
@array_function_dispatch(_broadcast_arrays_dispatcher, module='numpy')
def broadcast_arrays(*args, subok=False):
    args = [np.array(_m, copy=False, subok=subok) for _m in args]
    shape = _broadcast_shape(*args)
    if all(array.shape == shape for array in args):
        return args
    return [_broadcast_to(array, shape, subok=subok, readonly=False)
            for array in args]
