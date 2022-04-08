
from numpy.core import integer, empty, arange, asarray, roll
from numpy.core.overrides import array_function_dispatch, set_module
__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']
integer_types = (int, integer)
def _fftshift_dispatcher(x, axes=None):
    return (x,)
@array_function_dispatch(_fftshift_dispatcher, module='numpy.fft')
def fftshift(x, axes=None):
    x = asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]
    return roll(x, shift, axes)
@array_function_dispatch(_fftshift_dispatcher, module='numpy.fft')
def ifftshift(x, axes=None):
    x = asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]
    return roll(x, shift, axes)
@set_module('numpy.fft')
def fftfreq(n, d=1.0):
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n-1)//2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val
@set_module('numpy.fft')
def rfftfreq(n, d=1.0):
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0/(n*d)
    N = n//2 + 1
    results = arange(0, N, dtype=int)
    return results * val
