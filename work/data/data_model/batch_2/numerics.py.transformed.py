
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=["debugging.assert_all_finite", "verify_tensor_all_finite"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("verify_tensor_all_finite")
def verify_tensor_all_finite(t=None, msg=None, name=None, x=None, message=None):
  """Assert that the tensor does not contain any NaN's or Inf's.
  Args:
    t: Tensor to check.
    msg: Message to log on failure.
    name: A name for this operation (optional).
    x: Alias for t.
    message: Alias for msg.
  Returns:
    Same tensor as `t`.
  """
  x = deprecation.deprecated_argument_lookup("x", x, "t", t)
  message = deprecation.deprecated_argument_lookup(
      "message", message, "msg", msg)
  return verify_tensor_all_finite_v2(x, message, name)
@tf_export("debugging.assert_all_finite", v1=[])
@dispatch.add_dispatch_support
def verify_tensor_all_finite_v2(x, message, name=None):
  """Assert that the tensor does not contain any NaN's or Inf's.
  Args:
    x: Tensor to check.
    message: Message to log on failure.
    name: A name for this operation (optional).
  Returns:
    Same tensor as `x`.
  """
  with ops.name_scope(name, "VerifyFinite", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    with ops.colocate_with(x):
      verify_input = array_ops.check_numerics(x, message=message)
      out = control_flow_ops.with_dependencies([verify_input], x)
  return out
@tf_export(v1=["add_check_numerics_ops"])
def add_check_numerics_ops():
  """Connect a `tf.debugging.check_numerics` to every floating point tensor.
  `check_numerics` operations themselves are added for each `half`, `float`,
  or `double` tensor in the current default graph. For all ops in the graph, the
  `check_numerics` op for all of its (`half`, `float`, or `double`) inputs
  is guaranteed to run before the `check_numerics` op on any of its outputs.
  Note: This API is not compatible with the use of `tf.cond` or
  `tf.while_loop`, and will raise a `ValueError` if you attempt to call it
  in such a graph.
  Returns:
    A `group` op depending on all `check_numerics` ops added.
  Raises:
    ValueError: If the graph contains any numeric operations in a control flow
      structure.
    RuntimeError: If called with eager execution enabled.
  @compatibility(eager)
  Not compatible with eager execution. To check for `Inf`s and `NaN`s under
  eager execution, call `tf.debugging.enable_check_numerics()` once before
  executing the checked operations.
  @end_compatibility
  """
  if context.executing_eagerly():
    raise RuntimeError(
        "add_check_numerics_ops() is not compatible with eager execution. "
        "To check for Inf's and NaN's under eager execution, call "
        "tf.debugging.enable_check_numerics() once before executing the "
        "checked operations.")
  check_op = []
  for op in ops.get_default_graph().get_operations():
    for output in op.outputs:
      if output.dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
          raise ValueError("`tf.add_check_numerics_ops() is not compatible "
                           "with TensorFlow control flow operations such as "
                           "`tf.cond()` or `tf.while_loop()`.")
        message = op.name + ":" + str(output.value_index)
        with ops.control_dependencies(check_op):
          check_op = [array_ops.check_numerics(output, message=message)]
  return control_flow_ops.group(*check_op)
