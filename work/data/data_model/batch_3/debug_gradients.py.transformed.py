
import re
import uuid
import six
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
_GRADIENT_DEBUG_TAG = "gradient_debug_"
_gradient_debuggers = {}
def _tensor_to_grad_debug_op_name(tensor, grad_debugger_uuid):
  op_name, slot = debug_graphs.parse_node_or_tensor_name(tensor.name)
  return "%s_%d/%s%s" % (op_name, slot, _GRADIENT_DEBUG_TAG, grad_debugger_uuid)
def _parse_grad_debug_op_name(op_name):
  name_items = op_name.split("/")
  assert len(name_items) > 1
  assert name_items[-1].startswith(_GRADIENT_DEBUG_TAG)
  grad_debugger_uuid = name_items[-1][len(_GRADIENT_DEBUG_TAG):]
  if "_" in grad_debugger_uuid:
    grad_debugger_uuid = grad_debugger_uuid[:grad_debugger_uuid.index("_")]
  orig_tensor_slot = int(name_items[-2][name_items[-2].rfind("_") + 1:])
  orig_base_op_name = name_items[-2][:name_items[-2].rfind("_")]
  orig_tensor_name = ("/".join(name_items[:-2] + [orig_base_op_name]) +
                      ":%d" % orig_tensor_slot)
  return grad_debugger_uuid, orig_tensor_name
class GradientsDebugger(object):
  def __init__(self, y_tensor=None):
    self._uuid = uuid.uuid4().hex
    _gradient_debuggers[self._uuid] = self
    self._gradient_tensors = {}
    self._y_tensor = y_tensor
    self._graph = None
    if y_tensor:
      self._graph = y_tensor.graph
    self._is_active_context = False
  @property
  def y_tensor(self):
    return self._y_tensor
  @property
  def graph(self):
    return self._graph
  def __enter__(self):
    self._is_active_context = True
  def __exit__(self, unused_type, unused_value, unused_traceback):
    self._is_active_context = False
  def identify_gradient(self, input_tensor):
    """Create a debug identity tensor that registers and forwards gradients.
    The side effect of this method is that when gradient tensor(s) are created
    with respect to the any paths that include the `input_tensor`, the gradient
    tensor(s) with respect to `input_tensor` will be registered with this
    this `GradientsDebugger` instance and can later be retrieved, with the
    methods `gradient_tensor` and `gradient_tensors`.
    Example:
    ```python
    x = tf.Variable(1.0)
    y = tf.add(x, x)
    grad_debugger = tf_debug.GradientsDebugger()
    debug_y = grad_debugger.identify_gradient(y)
    z = tf.square(debug_y)
    with grad_debugger:
      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)
    y_grad = grad_debugger.gradient_tensor(y)
    ```
    Args:
      input_tensor: the input `tf.Tensor` object whose related gradient tensors
        are to be registered with this `GradientsDebugger` instance when they
        are created, e.g., during `tf.gradients` calls or the construction
        of optimization (training) op that uses `tf.gradients`.
    Returns:
      A forwarded identity of `input_tensor`, as a `tf.Tensor`.
    Raises:
      ValueError: If an op with name that duplicates the gradient-debugging op
        already exists in the graph (highly unlikely).
    """
    grad_debug_op_name = _tensor_to_grad_debug_op_name(input_tensor, self._uuid)
    identity_op = (
        gen_array_ops.debug_gradient_ref_identity
        if input_tensor.dtype._is_ref_dtype else
        gen_array_ops.debug_gradient_identity)
    debug_grad_identity = identity_op(input_tensor, name=grad_debug_op_name)
    assert debug_grad_identity.dtype == input_tensor.dtype
    if debug_grad_identity.op.name != grad_debug_op_name:
      raise ValueError(
          "The graph already contains an op named %s" % grad_debug_op_name)
    return debug_grad_identity
  def watch_gradients_by_tensors(self, graph, tensors):
    """Watch gradient tensors by x-tensor(s).
    The side effect of this method is that when gradient tensor(s) are created
    with respect to the any paths that include the `x_tensor`s, the gradient
    tensor(s) with respect to the tensor will be registered with this
    this `GradientsDebugger` instance and can later be retrieved, with the
    methods `gradient_tensor` and `gradient_tensors`.
    Unlike the method `identify_gradient`, this method is used to retrieve
    gradient tensors after the construction of the forward subgraph has
    completed (but before the construction of the backward subgraph).
    This method is the same as `watch_gradients_by_x_tensor_names` except that
    the tensors are specified by the Python `tf.Tensor` or `tf.Variable`
    objects, instead by name patterns.
    Example:
    ```python
    x = tf.Variable(1.0)
    y = tf.add(x, x, name="y")
    z = tf.square(debug_y)
    grad_debugger = tf_debug.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensors(y):
      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)
    y_grad = grad_debugger.gradient_tensor(y)
    y_grad = grad_debugger.gradient_tensor("y:0")
    ```
    Args:
      graph: the `tf.Graph` to watch the gradients on.
      tensors: a `tf.Tensor` or `tf.Variable` object, or a list of such objects.
    Returns:
      The GradientsDebugger instance itself.
    """
    if not isinstance(tensors, list):
      tensors = [tensors]
    tensor_name_regex = []
    for tensor in tensors:
      tensor_name_regex.append(re.escape(tensor.name) + "$")
    tensor_name_regex = "(" + "|".join(tensor_name_regex) + ")"
    return self.watch_gradients_by_tensor_names(graph, tensor_name_regex)
  def watch_gradients_by_tensor_names(self, graph, tensor_name_regex):
    """Watch gradient tensors by name(s) of the x-tensor(s).
    The side effect of this method is that when gradient tensor(s) are created
    with respect to the x-tensors, the gradient tensor(s) will be registered
    with this `GradientsDebugger` instance and can later be retrieved.
    Unlike the `identify_gradient` method, this method is used after the
    construction of the forward graph has completed. Unlike the
    `watch_gradients_by_tensor` method, this method does not use handles to the
    tensors of interest; it uses their names.
    This method is the same as `watch_gradients_by_tensors` except that the
    x-tensors are specified by name patterns, instead of `tf.Tensor` or
    `tf.Variable` objects.
    Example:
    ```python
    x = tf.Variable(1.0, name="x")
    y = tf.add(x, x, name="y")
    z = tf.square(debug_y)
    grad_debugger = tf_debug.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensor_names(r"(x|y):0$"):
      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)
    x_grad = grad_debugger.gradient_tensor("x:0")
    y_grad = grad_debugger.gradient_tensor("y:0")
    ```
    Args:
      graph: the `tf.Graph` to watch the gradients on.
      tensor_name_regex: the regular-expression pattern of the name(s) of the
        x-tensor(s) to watch. x-tensor refers to the tensors on the denominator
        of the differentiation.
    Returns:
      The GradientsDebugger instance itself.
    """
    tensor_name_pattern = re.compile(tensor_name_regex)
    with graph.as_default():
      for op in graph.get_operations():
        for output in op.outputs:
          if tensor_name_pattern.match(output.name):
            debug_op = self.identify_gradient(output)
            for consumer in list(output.consumers()):
              if consumer == debug_op.op:
                continue
              for i, consumer_input in enumerate(consumer.inputs):
                if consumer_input == output:
    return self
  def _check_same_graph(self, tensor):
    if self._graph is None:
      self._graph = tensor.graph
    elif self._graph != tensor.graph:
      raise ValueError(
          "The graph of the value (%s) is not the same as the graph %s" %
          (tensor.graph, self._graph))
  def register_gradient_tensor(self,
                               x_tensor_name,
                               gradient_tensor):
    """Register the gradient tensor for an x-tensor.
    Args:
      x_tensor_name: (`str`) the name of the independent `tf.Tensor`, i.e.,
        the tensor on the denominator of the differentiation.
      gradient_tensor: the gradient `tf.Tensor`.
    """
    if len(_gradient_debuggers) == 1 or self._is_active_context:
      self._check_same_graph(gradient_tensor)
      self._gradient_tensors[x_tensor_name] = gradient_tensor
  def gradient_tensor(self, x_tensor):
    """Get the gradient tensor of an x-tensor.
    Args:
      x_tensor: (`tf.Tensor`, `tf.Variable` or `str`) The x-tensor object or its
        name. x-tensor refers to the independent `tf.Tensor`, i.e., the tensor
        on the denominator of the differentiation.
    Returns:
      If found, the gradient tensor.
    Raises:
      TypeError: If `x_tensor` is not a `tf.Tensor`, `tf.Variable` or `str`.
      LookupError: If the `x_tensor` has not been registered with a gradient
        tensor.
    """
    x_tensor_name = self._get_tensor_name(x_tensor)
    if x_tensor_name not in self._gradient_tensors:
      raise LookupError(
          "This GradientsDebugger has not received any gradient tensor for "
          "x-tensor %s" % x_tensor_name)
    return self._gradient_tensors[x_tensor_name]
  def gradient_tensors(self):
    return self._gradient_tensors
  def _get_tensor_name(self, tensor):
    if isinstance(tensor, (ops.Tensor, variables.Variable)):
      return tensor.name
    elif isinstance(tensor, six.string_types):
      return tensor
    else:
      raise TypeError(
          "x_tensor must be a str or tf.Tensor or tf.Variable, "
          "but instead has type %s" % type(tensor))
def clear_gradient_debuggers():
  _gradient_debuggers.clear()
@ops.RegisterGradient("DebugGradientIdentity")
def _identify_gradient_grad(op, dy):
  grad_debugger_uuid, orig_tensor_name = _parse_grad_debug_op_name(op.name)
  grad_debugger = _gradient_debuggers[grad_debugger_uuid]
  grad_debugger.register_gradient_tensor(orig_tensor_name, dy)
  return dy
@ops.RegisterGradient("DebugGradientRefIdentity")
def _identify_gradient_grad_ref(op, dy):
  return _identify_gradient_grad(op, dy)
def gradient_values_from_dump(grad_debugger, x_tensor, dump):
  """Find gradient values from a `DebugDumpDir` object.
  Args:
    grad_debugger: the `tf_debug.GradientsDebugger` instance to be used.
    x_tensor: (`tf.Tensor`, `tf.Variable` or `str`) The x-tensor object or its
      name. x-tensor refers to the independent `tf.Tensor`, i.e., the tensor
      on the denominator of the differentiation.
    dump: A `tfdbg.DebugDumpDir` object.
  Returns:
    If this `GradientsDebugger` instance has the gradient tensor of `x_tensor`
      registered: a list of `numpy.ndarray` representing the value of the
      gradient tensor from `dump`. The list could be empty, if the gradient
      tensor is not executed in the `tf.Session.run()` call that generated
      the `dump`. The list could also contain multiple values of the gradient
      tensor, e.g., if gradient tensor is computed repeatedly in a
      `tf.while_loop` during the run that generated the `dump`.
  Raises:
    LookupError: If this `GradientsDebugger` instance does not have the
      gradient tensor of `x_tensor` registered.
    ValueError: If this `GradientsDebugger` has a `tf.Graph` object that
      does not match the `tf.Graph` object of the `dump`.
    TypeError: If `x_tensor` is not a `tf.Tensor`, `tf.Variable` or `str`.
  """
  if (dump.python_graph and grad_debugger.graph and
      dump.python_graph != grad_debugger.graph):
    raise ValueError(
        "This GradientsDebugger instance has a graph (%s) that differs from "
        "the graph of the DebugDumpDir object (%s)." %
        (grad_debugger.graph, dump.python_graph))
  gradient_tensor = grad_debugger.gradient_tensor(x_tensor)
  node_name, output_slot = debug_graphs.parse_node_or_tensor_name(
      gradient_tensor.name)
  try:
    return dump.get_tensors(node_name, output_slot, "DebugIdentity")
  except debug_data.WatchKeyDoesNotExistInDebugDumpDirError:
    return []
