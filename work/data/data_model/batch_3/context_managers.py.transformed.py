
import contextlib
from tensorflow.python.framework import ops
from tensorflow.python.ops import tensor_array_ops
def control_dependency_on_returns(return_value):
  def control_dependency_handle(t):
    if isinstance(t, tensor_array_ops.TensorArray):
      return t.flow
    return t
  if return_value is None:
    return contextlib.contextmanager(lambda: (yield))()
  if not isinstance(return_value, (list, tuple)):
    return_value = (return_value,)
  return_value = tuple(control_dependency_handle(t) for t in return_value)
  return ops.control_dependencies(return_value)
