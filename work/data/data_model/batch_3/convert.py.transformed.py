
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
def optional_param_to_tensor(argument_name,
                             argument_value,
                             argument_default=0,
                             argument_dtype=dtypes.int64):
  if argument_value is not None:
    return ops.convert_to_tensor(
        argument_value, dtype=argument_dtype, name=argument_name)
  else:
    return constant_op.constant(
        argument_default, dtype=argument_dtype, name=argument_name)
def partial_shape_to_tensor(shape_like):
  try:
    shape_like = tensor_shape.as_shape(shape_like)
    return ops.convert_to_tensor(
        [dim if dim is not None else -1 for dim in shape_like.as_list()],
        dtype=dtypes.int64)
  except (TypeError, ValueError):
    ret = ops.convert_to_tensor(shape_like, preferred_dtype=dtypes.int64)
    if ret.shape.dims is not None and len(ret.shape.dims) != 1:
      raise ValueError("The given shape {} must be a 1-D tensor of `tf.int64` "
                       "values, but the shape was {}.".format(
                           shape_like, ret.shape))
    if ret.dtype != dtypes.int64:
      raise TypeError("The given shape {} must be a 1-D tensor of `tf.int64` "
                      "values, but the element type was {}.".format(
                          shape_like, ret.dtype.name))
    return ret
