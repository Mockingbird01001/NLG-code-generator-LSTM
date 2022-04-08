
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('histogram_fixed_width_bins')
@dispatch.add_dispatch_support
def histogram_fixed_width_bins(values,
                               value_range,
                               nbins=100,
                               dtype=dtypes.int32,
                               name=None):
  """Bins the given values for use in a histogram.
  Given the tensor `values`, this operation returns a rank 1 `Tensor`
  representing the indices of a histogram into which each element
  of `values` would be binned. The bins are equal width and
  determined by the arguments `value_range` and `nbins`.
  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').
  Returns:
    A `Tensor` holding the indices of the binned values whose shape matches
    `values`.
  Raises:
    TypeError: If any unsupported dtype is provided.
    tf.errors.InvalidArgumentError: If value_range does not
        satisfy value_range[0] < value_range[1].
  Examples:
  ...
  >>> nbins = 5
  >>> value_range = [0.0, 5.0]
  >>> new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
  >>> indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=5)
  >>> indices.numpy()
  array([0, 0, 1, 2, 4, 4], dtype=int32)
  """
  with ops.name_scope(name, 'histogram_fixed_width_bins',
                      [values, value_range, nbins]):
    values = ops.convert_to_tensor(values, name='values')
    shape = array_ops.shape(values)
    values = array_ops.reshape(values, [-1])
    value_range = ops.convert_to_tensor(value_range, name='value_range')
    nbins = ops.convert_to_tensor(nbins, dtype=dtypes.int32, name='nbins')
    check = control_flow_ops.Assert(
        math_ops.greater(nbins, 0), ['nbins %s must > 0' % nbins])
    nbins = control_flow_ops.with_dependencies([check], nbins)
    nbins_float = math_ops.cast(nbins, values.dtype)
    scaled_values = math_ops.truediv(
        values - value_range[0],
        value_range[1] - value_range[0],
        name='scaled_values')
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)
    return array_ops.reshape(indices, shape)
@tf_export('histogram_fixed_width')
@dispatch.add_dispatch_support
def histogram_fixed_width(values,
                          value_range,
                          nbins=100,
                          dtype=dtypes.int32,
                          name=None):
  """Return histogram of values.
  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fell into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.
  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').
  Returns:
    A 1-D `Tensor` holding histogram of values.
  Raises:
    TypeError: If any unsupported dtype is provided.
    tf.errors.InvalidArgumentError: If value_range does not
        satisfy value_range[0] < value_range[1].
  Examples:
  ...
  >>> nbins = 5
  >>> value_range = [0.0, 5.0]
  >>> new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
  >>> hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
  >>> hist.numpy()
  array([2, 1, 1, 0, 2], dtype=int32)
  """
  with ops.name_scope(name, 'histogram_fixed_width',
                      [values, value_range, nbins]) as name:
    return gen_math_ops._histogram_fixed_width(
        values, value_range, nbins, dtype=dtype, name=name)
