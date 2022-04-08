
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _check_params(window_length, dtype):
  if not dtype.is_floating:
    raise ValueError('dtype must be a floating point type. Found %s' % dtype)
  window_length = ops.convert_to_tensor(window_length, dtype=dtypes.int32)
  window_length.shape.assert_has_rank(0)
  return window_length
@tf_export('signal.kaiser_window')
@dispatch.add_dispatch_support
def kaiser_window(window_length, beta=12., dtype=dtypes.float32, name=None):
  with ops.name_scope(name, 'kaiser_window'):
    window_length = _check_params(window_length, dtype)
    window_length_const = tensor_util.constant_value(window_length)
    if window_length_const == 1:
      return array_ops.ones([1], dtype=dtype)
    halflen_float = (
        math_ops.cast(window_length, dtype=dtypes.float32) - 1.0) / 2.0
    arg = math_ops.range(-halflen_float, halflen_float + 0.1,
                         dtype=dtypes.float32)
    arg = math_ops.cast(arg, dtype=dtype)
    beta = math_ops.cast(beta, dtype=dtype)
    one = math_ops.cast(1.0, dtype=dtype)
    two = math_ops.cast(2.0, dtype=dtype)
    halflen_float = math_ops.cast(halflen_float, dtype=dtype)
    num = beta * math_ops.sqrt(
        one - math_ops.pow(arg, two) / math_ops.pow(halflen_float, two))
    window = math_ops.exp(num - beta) * (
        special_math_ops.bessel_i0e(num) / special_math_ops.bessel_i0e(beta))
  return window
@tf_export('signal.kaiser_bessel_derived_window')
@dispatch.add_dispatch_support
def kaiser_bessel_derived_window(window_length, beta=12.,
                                 dtype=dtypes.float32, name=None):
  with ops.name_scope(name, 'kaiser_bessel_derived_window'):
    window_length = _check_params(window_length, dtype)
    halflen = window_length // 2
    kaiserw = kaiser_window(halflen + 1, beta, dtype=dtype)
    kaiserw_csum = math_ops.cumsum(kaiserw)
    halfw = math_ops.sqrt(kaiserw_csum[:-1] / kaiserw_csum[-1])
    window = array_ops.concat((halfw, halfw[::-1]), axis=0)
  return window
@tf_export('signal.vorbis_window')
@dispatch.add_dispatch_support
def vorbis_window(window_length, dtype=dtypes.float32, name=None):
  with ops.name_scope(name, 'vorbis_window'):
    window_length = _check_params(window_length, dtype)
    arg = math_ops.cast(math_ops.range(window_length), dtype=dtype)
    window = math_ops.sin(np.pi / 2.0 * math_ops.pow(math_ops.sin(
        np.pi / math_ops.cast(window_length, dtype=dtype) *
        (arg + 0.5)), 2.0))
  return window
@tf_export('signal.hann_window')
@dispatch.add_dispatch_support
def hann_window(window_length, periodic=True, dtype=dtypes.float32, name=None):
  return _raised_cosine_window(name, 'hann_window', window_length, periodic,
                               dtype, 0.5, 0.5)
@tf_export('signal.hamming_window')
@dispatch.add_dispatch_support
def hamming_window(window_length, periodic=True, dtype=dtypes.float32,
                   name=None):
  return _raised_cosine_window(name, 'hamming_window', window_length, periodic,
                               dtype, 0.54, 0.46)
def _raised_cosine_window(name, default_name, window_length, periodic,
                          dtype, a, b):
  if not dtype.is_floating:
    raise ValueError('dtype must be a floating point type. Found %s' % dtype)
  with ops.name_scope(name, default_name, [window_length, periodic]):
    window_length = ops.convert_to_tensor(window_length, dtype=dtypes.int32,
                                          name='window_length')
    window_length.shape.assert_has_rank(0)
    window_length_const = tensor_util.constant_value(window_length)
    if window_length_const == 1:
      return array_ops.ones([1], dtype=dtype)
    periodic = math_ops.cast(
        ops.convert_to_tensor(periodic, dtype=dtypes.bool, name='periodic'),
        dtypes.int32)
    periodic.shape.assert_has_rank(0)
    even = 1 - math_ops.mod(window_length, 2)
    n = math_ops.cast(window_length + periodic * even - 1, dtype=dtype)
    count = math_ops.cast(math_ops.range(window_length), dtype)
    cos_arg = constant_op.constant(2 * np.pi, dtype=dtype) * count / n
    if window_length_const is not None:
      return math_ops.cast(a - b * math_ops.cos(cos_arg), dtype=dtype)
    return control_flow_ops.cond(
        math_ops.equal(window_length, 1),
        lambda: array_ops.ones([window_length], dtype=dtype),
        lambda: math_ops.cast(a - b * math_ops.cos(cos_arg), dtype=dtype))
