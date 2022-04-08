
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import tf_inspect
def assert_stmt(expression1, expression2):
  if not callable(expression2):
    raise ValueError('{} must be a callable'.format(expression2))
  args, _, keywords, _ = tf_inspect.getargspec(expression2)
  if args or keywords:
    raise ValueError('{} may not have any arguments'.format(expression2))
  if tensor_util.is_tf_type(expression1):
    return _tf_assert_stmt(expression1, expression2)
  else:
    return _py_assert_stmt(expression1, expression2)
def _tf_assert_stmt(expression1, expression2):
  """Overload of assert_stmt that stages a TF Assert.
  This implementation deviates from Python semantics as follows:
    (1) the assertion is verified regardless of the state of __debug__
    (2) on assertion failure, the graph execution will fail with
        tensorflow.errors.ValueError, rather than AssertionError.
  Args:
    expression1: tensorflow.Tensor, must evaluate to a tf.bool scalar
    expression2: Callable[[], Union[tensorflow.Tensor, List[tensorflow.Tensor]]]
  Returns:
    tensorflow.Operation
  """
  expression2_tensors = expression2()
  if not isinstance(expression2_tensors, list):
    expression2_tensors = [expression2_tensors]
  return control_flow_ops.Assert(expression1, expression2_tensors)
def _py_assert_stmt(expression1, expression2):
  assert expression1, expression2()
  return None
