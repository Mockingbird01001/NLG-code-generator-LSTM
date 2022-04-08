
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import custom_aggregator_op_wrapper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
_custom_aggregator_op = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_custom_aggregator_op.so'))
def custom_aggregator(input_tensor, tensor_id: str):
  if input_tensor.dtype != dtypes.float32:
    raise ValueError('Custom aggregator op only accept float32 values.')
  return custom_aggregator_op_wrapper.custom_aggregator(input_tensor, tensor_id)
