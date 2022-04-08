
import collections
import threading
import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
IGNORE_OP_OUTPUTS = (
)
SAFE_OPS = (
    b"Concat",
    b"ConcatV2",
    b"ExpandDims",
    b"Fill",
    b"Gather",
    b"Maximum",
    b"Minimum",
    b"Reshape",
    b"Slice",
    b"Squeeze",
    b"Stack",
    b"StridedSlice",
    b"StridedSliceGrad",
    b"TensorListConcatV2",
    b"TensorListGather",
    b"TensorListGetItem",
    b"TensorListPopBack",
    b"TensorListStack",
    b"Transpose",
    b"Unpack",
)
_state = threading.local()
_check_numerics_callback_create_counter = monitoring.Counter(
    "/tensorflow/api/python/debugging/check_numerics_callback_create_counter",
    "Counter for number of times the check_numerics op callback is created.")
def limit_string_length(string, max_len=50):
  """Limit the length of input string.
  Args:
    string: Input string.
    max_len: (int or None) If int, the length limit. If None, no limit.
  Returns:
    Possibly length-limited string.
  """
  if max_len is None or len(string) <= max_len:
    return string
  else:
    return "..." + string[len(string) - max_len:]
_CHECK_NUMERICS_INPUT_LOOKUP = collections.defaultdict(dict)
def _maybe_lookup_original_input_tensor(graph, tensor):
  if (graph and
      graph in _CHECK_NUMERICS_INPUT_LOOKUP and
      tensor.name in _CHECK_NUMERICS_INPUT_LOOKUP[graph]):
    return _CHECK_NUMERICS_INPUT_LOOKUP[graph][tensor.name]
  else:
    return tensor
def get_check_numerics_error_message(slot,
                                     num_outputs,
                                     op_type,
                                     tensor,
                                     inputs,
                                     graph=None,
                                     traceback=None,
                                     stack_height_limit=30,
                                     path_length_limit=50):
  """Create a meaningful and user-friendly error message about offending tensor.
  The error message reveals the following info about the op that outputs
  NaN/Infinity: dtype, shape (to the extent known at graph-construction time),
  input tensors, stack trace for op creation (if is graph mode).
  Args:
    slot: (int) slot index of the tensor output.
    num_outputs: (int) total number of outputs of the op.
    op_type: (str) Type of the that generates `tensor`.
    tensor: (Tensor) the offending tensor, i.e., the tensor that contains
      Infinities or NaNs.
    inputs: (array of Tensor) inputs to the op that generates `tensor`.
    graph: (tf.Graph) the graph object that `tensor` belongs to. Available only
      under graph mode.
    traceback: (list of trace frames) the stack trace of the op's creation.
      Available only under graph model.
    stack_height_limit: (int or None) If int, limit to the height of the stack
      trace printed in the error message. If None, no limit to the height.
    path_length_limit: (int or None) Length limit for file paths included in the
      formatted stack trace.
  Returns:
    (str) A formatted error message.
  """
  eager_vs_graph_qualifier = "graph" if graph else "eagerly-executing"
  message = "\n"
  message += (
      "\n!!! Detected Infinity or NaN in output %d of "
      (slot, eager_vs_graph_qualifier, op_type, num_outputs))
  message += "  dtype: %s\n" % tensor.dtype
  message += "  shape: %s\n" % (tensor.shape,)
  if not graph:
    is_inf = np.isinf(tensor)
    num_neg_inf = np.sum(np.logical_and(np.less(tensor, 0.), is_inf))
    num_pos_inf = np.sum(np.logical_and(np.greater(tensor, 0.), is_inf))
    num_nan = np.sum(np.isnan(tensor))
    if num_neg_inf > 0:
    if num_pos_inf > 0:
    if num_nan:
  if len(inputs) > 1:
    message += "\n  Input tensors (%d):\n" % len(inputs)
    for slot, input_tensor in enumerate(inputs):
      message += "         %d: %s\n" % (
          slot, _maybe_lookup_original_input_tensor(graph, input_tensor))
  elif len(inputs) == 1:
    message += "\n  Input tensor: %s\n" % (
        _maybe_lookup_original_input_tensor(graph, inputs[0]))
  if graph and hasattr(graph, "name") and graph.name:
    message += "  Graph name: \"%s\"\n" % graph.name
  if graph and traceback:
    message += (
        "\n  Stack trace of op's creation (\"->\": inferred user code):\n")
    if stack_height_limit is not None and len(traceback) > stack_height_limit:
      num_omitted_frames = len(traceback) - stack_height_limit
      message += "    + ... (Omitted %d frames)\n" % num_omitted_frames
    for filepath, lineno, function_name, source_line in traceback[
        -stack_height_limit:]:
      user_code_indicator = "    "
      if not source_utils.guess_is_tensorflow_py_library(filepath):
        user_code_indicator = " -> "
      message += "    + %s (L%d) %s\n" % (
          limit_string_length(filepath, path_length_limit), lineno,
          function_name)
      if source_line is not None:
        message += "%s|   %s\n" % (user_code_indicator, source_line)
  message += "\n"
  return message
def _debug_summary(x):
  return gen_debug_ops.debug_numeric_summary_v2(
      x,
      tensor_debug_mode=(
          debug_event_pb2.TensorDebugMode.REDUCE_INF_NAN_THREE_SLOTS))
class CheckNumericsCallback(object):
  def __init__(self, stack_height_limit, path_length_limit):
    self._stack_height_limit = stack_height_limit
    self._path_length_limit = path_length_limit
    self._placeholder_to_debug_tensor = dict()
  def callback(self,
               op_type,
               inputs,
               attrs,
               outputs,
               op_name=None,
               graph=None):
    op_type_bytes = compat.as_bytes(op_type)
    is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
    if (op_type_bytes in op_callbacks_common.OP_CALLBACK_SKIP_OPS or
        op_type_bytes in SAFE_OPS):
      return None
    if graph:
      instrumented_outputs = []
      if is_v1_graph_mode:
        for input_tensor in inputs:
          if input_tensor in self._placeholder_to_debug_tensor and outputs:
                self._placeholder_to_debug_tensor[input_tensor].op)
      for slot, output in enumerate(outputs):
        if (output.dtype.is_floating and
            (op_type_bytes, slot) not in IGNORE_OP_OUTPUTS):
          checked_output = array_ops.check_numerics_v2(
              output if is_v1_graph_mode else _debug_summary(output),
              get_check_numerics_error_message(
                  slot,
                  len(outputs),
                  op_type,
                  output,
                  inputs,
                  graph=graph,
                  traceback=output.op.traceback,
                  stack_height_limit=self._stack_height_limit,
                  path_length_limit=self._path_length_limit))
          _CHECK_NUMERICS_INPUT_LOOKUP[graph][checked_output.name] = output
          instrumented_outputs.append(self._get_output_tensor(
              op_type_bytes, output, checked_output, is_v1_graph_mode))
        else:
          instrumented_outputs.append(output)
      return instrumented_outputs
    else:
      if op_type_bytes == b"CheckNumericsV2":
        return None
      for slot, output in enumerate(outputs):
        if (output.dtype.is_floating and
            (op_type_bytes, slot) not in IGNORE_OP_OUTPUTS):
          array_ops.check_numerics_v2(
              output,
              get_check_numerics_error_message(
                  slot, len(outputs), op_type, output, inputs,
                  stack_height_limit=self._stack_height_limit,
                  path_length_limit=self._path_length_limit))
  def _get_output_tensor(self,
                         op_type,
                         tensor,
                         checked_tensor,
                         is_v1_graph_mode):
    if is_v1_graph_mode:
      if op_type == b"Placeholder":
        self._placeholder_to_debug_tensor[tensor] = checked_tensor
        return tensor
      else:
        return checked_tensor
    else:
      return tensor
@tf_export("debugging.enable_check_numerics")
def enable_check_numerics(stack_height_limit=30,
                          path_length_limit=50):
  r"""Enable tensor numerics checking in an eager/graph unified fashion.
  The numerics checking mechanism will cause any TensorFlow eager execution or
  graph execution to error out as soon as an op's output tensor contains
  infinity or NaN.
  This method is idempotent. Calling it multiple times has the same effect
  as calling it once.
  This method takes effect only on the thread in which it is called.
  When a op's float-type output tensor contains any Infinity or NaN, an
  `tf.errors.InvalidArgumentError` will be thrown, with an error message that
  reveals the following information:
    - The type of the op that generated the tensor with bad numerics.
    - Data type (dtype) of the tensor.
    - Shape of the tensor (to the extent known at the time of eager execution
      or graph construction).
    - Name of the containing graph (if available).
    - (Graph mode only): The stack trace of the intra-graph op's creation,
      with a stack-height limit and a path-length limit for visual clarity.
      The stack frames that belong to the user's code (as opposed to
      tensorflow's internal code) are highlighted with a text arrow ("->").
    - (Eager mode only): How many of the offending tensor's elements are
      `Infinity` and `NaN`, respectively.
  Once enabled, the check-numerics mechanism can be disabled by using
  `tf.debugging.disable_check_numerics()`.
  Example usage:
  1. Catching infinity during the execution of a `tf.function` graph:
     ```py
     import tensorflow as tf
     tf.debugging.enable_check_numerics()
     @tf.function
     def square_log_x_plus_1(x):
       v = tf.math.log(x + 1)
       return tf.math.square(v)
     x = -1.0
     y = square_log_x_plus_1(x)
     z = -y
    ```
  2. Catching NaN during eager execution:
     ```py
     import numpy as np
     import tensorflow as tf
     tf.debugging.enable_check_numerics()
     x = np.array([[0.0, -1.0], [4.0, 3.0]])
     y = tf.math.sqrt(x)
     z = tf.matmul(y, y)
     ```
  NOTE: If your code is running on TPUs, be sure to call
  `tf.config.set_soft_device_placement(True)` before calling
  `tf.debugging.enable_check_numerics()` as this API uses automatic outside
  compilation on TPUs. For example:
  ```py
  tf.config.set_soft_device_placement(True)
  tf.debugging.enable_check_numerics()
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  strategy = tf.distribute.TPUStrategy(resolver)
  with strategy.scope():
  ```
  Args:
    stack_height_limit: Limit to the height of the printed stack trace.
      Applicable only to ops in `tf.function`s (graphs).
    path_length_limit: Limit to the file path included in the printed stack
      trace. Applicable only to ops in `tf.function`s (graphs).
  """
  if not hasattr(_state, "check_numerics_callback"):
    _state.check_numerics_callback = CheckNumericsCallback(
        stack_height_limit, path_length_limit)
  op_callbacks.add_op_callback(_state.check_numerics_callback.callback)
  logging.info(
      "Enabled check-numerics callback in thread %s",
      threading.current_thread().name)
  _check_numerics_callback_create_counter.get_cell().increase_by(1)
@tf_export("debugging.disable_check_numerics")
def disable_check_numerics():
  """Disable the eager/graph unified numerics checking mechanism.
  This method can be used after a call to `tf.debugging.enable_check_numerics()`
  to disable the numerics-checking mechanism that catches infinity and NaN
  values output by ops executed eagerly or in tf.function-compiled graphs.
  This method is idempotent. Calling it multiple times has the same effect
  as calling it once.
  This method takes effect only on the thread in which it is called.
  """
  if not hasattr(_state, "check_numerics_callback"):
    return
  try:
    op_callbacks.remove_op_callback(_state.check_numerics_callback.callback)
    delattr(_state, "check_numerics_callback")
    logging.info(
        "Disabled check-numerics callback in thread %s",
        threading.current_thread().name)
  except KeyError:
    pass
