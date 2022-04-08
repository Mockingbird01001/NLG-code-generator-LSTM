
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import handle_data_util
def _DTypeFromTensor(tensor):
  dtype = tensor.dtype
  if dtype.base_dtype == dtypes.variant:
    if isinstance(tensor, ops.EagerTensor):
    else:
      handle_data = handle_data_util.get_resource_handle_data(tensor)
    if (handle_data is not None
        and handle_data.is_set
        and handle_data.shape_and_type):
      first_type = handle_data.shape_and_type[0].dtype
      if (first_type != types_pb2.DT_INVALID
          and all(shape_and_type.dtype == first_type
                  for shape_and_type in handle_data.shape_and_type)):
        return first_type
  return dtype
def IsTrainable(tensor_or_dtype):
  if tensor_util.is_tf_type(tensor_or_dtype):
    dtype = _DTypeFromTensor(tensor_or_dtype)
  else:
    dtype = tensor_or_dtype
  dtype = dtypes.as_dtype(dtype)
  return dtype.base_dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                              dtypes.complex64, dtypes.complex128,
                              dtypes.resource, dtypes.variant, dtypes.bfloat16)
