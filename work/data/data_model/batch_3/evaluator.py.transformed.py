
import re
from tensorflow.python.debug.lib import debug_data
_DUMP_TENSOR_PATTERN = re.compile(r"`.*?`")
_DEVICE_NAME_PREFIX_PATTERN = re.compile(
    r"/job:(\w)+/replica:(\d)+/task:(\d)+/(\w)+:(\d)+:")
_EXEC_INDEX_SUFFIX_PATTERN = re.compile(r"\[(\d)*\]$")
_DEFAULT_DEBUG_OP = "DebugIdentity"
def _parse_debug_tensor_name(debug_tensor_name):
  """Parse a debug tensor name in a to-be-evaluated expression.
  Args:
    debug_tensor_name: name of the debug tensor, with or without
      device name as a prefix, with or without debug op, with or
      without '[<exec_index>]' as a suffix.
      E.g., without device name prefix, without debug op suffix:
        "hidden_0/MatMul:0"
      E.g., with device name prefix:
        "/job:worker/replica:0/task:1/gpu:0:hidden_0/MatMul:0"
      E.g., with debug op suffix:
        "hidden_0/MatMul:0:DebugNumericSummary"
      E.g., with device name prefix and debug op suffix:
        "/job:worker/replica:0/task:1/gpu:0:hidden_0/MatMul:0:DebugNumericSummary"
      E.g., with device name prefix, debug op and an exec index:
        "/job:worker/replica:0/task:1/gpu:0:hidden_0/MatMul:0:DebugNumericSummary[1]"
  Returns:
    device_name: If device name prefix exists, the device name; otherwise,
      `None`.
    node_name: Name of the node.
    output_slot: Output slot index as an `int`.
    debug_op: If the debug op suffix exists, the debug op name; otherwise,
      `None`.
    exec_index: Execution index (applicable to cases in which a debug tensor
      is computed multiple times in a `tf.Session.run` call, e.g., due to
      `tf.while_loop`). If the exec_index suffix does not exist, this value
      defaults to `0`.
  Raises:
    ValueError: If the input `debug_tensor_name` is malformed.
  """
  device_prefix_match = re.match(_DEVICE_NAME_PREFIX_PATTERN, debug_tensor_name)
  if device_prefix_match:
    device_name = debug_tensor_name[
        device_prefix_match.start() : device_prefix_match.end() - 1]
    debug_tensor_name = debug_tensor_name[device_prefix_match.end():]
  else:
    device_name = None
  split_items = debug_tensor_name.split(":")
  if len(split_items) not in (2, 3):
    raise ValueError(
        "The debug tensor name in the to-be-evaluated expression is malformed: "
        "'%s'" % debug_tensor_name)
  exec_index_match = re.search(_EXEC_INDEX_SUFFIX_PATTERN, split_items[-1])
  if exec_index_match:
    exec_index = int(split_items[-1][
        exec_index_match.start() + 1 : exec_index_match.end() - 1])
    split_items[-1] = split_items[-1][:exec_index_match.start()]
  else:
    exec_index = 0
  if len(split_items) == 2:
    node_name = split_items[0]
    output_slot = int(split_items[1])
    debug_op = _DEFAULT_DEBUG_OP
  else:
    split_items = debug_tensor_name.split(":")
    node_name = split_items[0]
    output_slot = int(split_items[1])
    debug_op = split_items[2]
  return device_name, node_name, output_slot, debug_op, exec_index
class ExpressionEvaluator(object):
  def __init__(self, dump):
    self._dump = dump
    self._cached_tensor_values = {}
  def evaluate(self, expression):
    dump_tensors_iter = re.finditer(_DUMP_TENSOR_PATTERN, expression)
    rewritten_expression = expression
    for match in reversed(list(dump_tensors_iter)):
      tensor_name = match.group(0)[1:-1].strip()
      device_name, node_name, output_slot, debug_op, exec_index = (
          _parse_debug_tensor_name(tensor_name))
      if tensor_name not in self._cached_tensor_values:
        try:
          value = self._dump.get_tensors(
              node_name, output_slot, debug_op,
              device_name=device_name)[exec_index]
        except debug_data.WatchKeyDoesNotExistInDebugDumpDirError:
          raise ValueError(
              "Eval failed due to the value of %s:%d:DebugIdentity being "
              "unavailable" % (node_name, output_slot))
        self._cached_tensor_values[tensor_name] = value
      rewritten_expression = (
          rewritten_expression[:match.start(0)] +
          "self._cached_tensor_values['" + tensor_name + "']" +
          rewritten_expression[match.end(0):])
