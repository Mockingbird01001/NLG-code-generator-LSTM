
import numpy as onp
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
DEFAULT_RANDN_DTYPE = onp.float32
@np_utils.np_doc('random.seed')
def seed(s):
  try:
    s = int(s)
  except TypeError:
    raise ValueError(
        f'Argument `s` got an invalid value {s}. Only integers are supported.')
  random_seed.set_seed(s)
@np_utils.np_doc('random.randn')
def randn(*args):
  return standard_normal(size=args)
@np_utils.np_doc('random.standard_normal')
def standard_normal(size=None):
  if size is None:
    size = ()
  elif np_utils.isscalar(size):
    size = (size,)
  dtype = np_dtypes.default_float_type()
  return random_ops.random_normal(size, dtype=dtype)
@np_utils.np_doc('random.uniform')
def uniform(low=0.0, high=1.0, size=None):
  dtype = np_dtypes.default_float_type()
  low = np_array_ops.asarray(low, dtype=dtype)
  high = np_array_ops.asarray(high, dtype=dtype)
  if size is None:
    size = array_ops.broadcast_dynamic_shape(low.shape, high.shape)
  return random_ops.random_uniform(
      shape=size, minval=low, maxval=high, dtype=dtype)
@np_utils.np_doc('random.poisson')
def poisson(lam=1.0, size=None):
  if size is None:
    size = ()
  elif np_utils.isscalar(size):
    size = (size,)
  return random_ops.random_poisson(shape=size, lam=lam, dtype=np_dtypes.int_)
@np_utils.np_doc('random.random')
def random(size=None):
  return uniform(0., 1., size)
@np_utils.np_doc('random.rand')
def rand(*size):
  return uniform(0., 1., size)
@np_utils.np_doc('random.randint')
  low = int(low)
  if high is None:
    high = low
    low = 0
  if size is None:
    size = ()
  elif isinstance(size, int):
    size = (size,)
  dtype_orig = dtype
  dtype = np_utils.result_type(dtype)
  accepted_dtypes = (onp.int32, onp.int64)
  if dtype not in accepted_dtypes:
    raise ValueError(
        f'Argument `dtype` got an invalid value {dtype_orig}. Only those '
        f'convertible to {accepted_dtypes} are supported.')
  return random_ops.random_uniform(
      shape=size, minval=low, maxval=high, dtype=dtype)
