
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_dtypes
def convert_to_tensor(value, dtype=None, dtype_hint=None):
  """Wrapper over `tf.convert_to_tensor`.
  Args:
    value: value to convert
    dtype: (optional) the type we would like it to be converted to.
    dtype_hint: (optional) soft preference for the type we would like it to be
      converted to. `tf.convert_to_tensor` will attempt to convert value to this
      type first, but will not fail if conversion is not possible falling back
      to inferring the type instead.
  Returns:
    Value converted to tf.Tensor.
  """
  if (dtype is None and isinstance(value, int) and
      value >= 2**63):
    dtype = dtypes.uint64
  elif dtype is None and dtype_hint is None and isinstance(value, float):
    dtype = np_dtypes.default_float_type()
  return ops.convert_to_tensor(value, dtype=dtype, dtype_hint=dtype_hint)
ndarray = ops.Tensor
