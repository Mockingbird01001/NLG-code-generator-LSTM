import warnings
warnings.warn("Importing from numpy.matlib is deprecated since 1.19.0. "
              "The matrix subclass is not the recommended way to represent "
              "matrices or deal with linear algebra (see "
              "https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). "
              "Please adjust your code to use regular ndarray. ",
              PendingDeprecationWarning, stacklevel=2)
import numpy as np
from numpy.matrixlib.defmatrix import matrix, asmatrix
from numpy import *
__version__ = np.__version__
__all__ = np.__all__[:]
__all__ += ['rand', 'randn', 'repmat']
def empty(shape, dtype=None, order='C'):
    return ndarray.__new__(matrix, shape, dtype, order=order)
def ones(shape, dtype=None, order='C'):
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(1)
    return a
def zeros(shape, dtype=None, order='C'):
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(0)
    return a
def identity(n,dtype=None):
    a = array([1]+n*[0], dtype=dtype)
    b = empty((n, n), dtype=dtype)
    b.flat = a
    return b
def eye(n,M=None, k=0, dtype=float, order='C'):
    return asmatrix(np.eye(n, M=M, k=k, dtype=dtype, order=order))
def rand(*args):
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.rand(*args))
def randn(*args):
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.randn(*args))
def repmat(a, m, n):
    a = asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1, 1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    return c.reshape(rows, cols)
