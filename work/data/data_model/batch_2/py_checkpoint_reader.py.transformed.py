
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_checkpoint_reader import CheckpointReader
from tensorflow.python.util.tf_export import tf_export
def error_translator(e):
  error_message = str(e)
  if 'not found in checkpoint' in error_message or (
      'Failed to find any '
      'matching files for') in error_message:
    raise errors_impl.NotFoundError(None, None, error_message)
  elif 'Sliced checkpoints are not supported' in error_message or (
      'Data type '
      'not '
      'supported') in error_message:
    raise errors_impl.UnimplementedError(None, None, error_message)
  elif 'Failed to get matching files on' in error_message:
    raise errors_impl.InvalidArgumentError(None, None, error_message)
  elif 'Unable to open table file' in error_message:
    raise errors_impl.DataLossError(None, None, error_message)
  elif 'Failed to find the saved tensor slices' in error_message or (
      'not convertible to numpy dtype' in error_message):
    raise errors_impl.InternalError(None, None, error_message)
  else:
    raise errors_impl.OpError(None, None, error_message, errors_impl.UNKNOWN)
def get_variable_to_dtype_map(self):
  return {
      name: dtypes.DType(type_enum)
  }
CheckpointReader.get_variable_to_dtype_map = get_variable_to_dtype_map
def has_tensor(self, tensor_str):
CheckpointReader.has_tensor = has_tensor
def get_tensor(self, tensor_str):
  try:
    return CheckpointReader.CheckpointReader_GetTensor(
        self, compat.as_bytes(tensor_str))
  except RuntimeError as e:
    error_translator(e)
CheckpointReader.get_tensor = get_tensor
@tf_export(v1=['train.NewCheckpointReader'])
def NewCheckpointReader(filepattern):
  try:
    return CheckpointReader(compat.as_bytes(filepattern))
  except RuntimeError as e:
    error_translator(e)
