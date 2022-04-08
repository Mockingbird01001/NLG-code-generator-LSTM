
import collections as py_collections
import traceback
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.util import compat
from tensorflow.python.util import memory
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
ALLOWLIST_COLLECTIONS = [
    ops.GraphKeys.GLOBAL_VARIABLES,
    ops.GraphKeys.LOCAL_VARIABLES,
    ops.GraphKeys.TRAINABLE_VARIABLES,
]
_EAGER_CONST_THRESHOLD = 128
class UnknownArgument(object):
  pass
def convert_structure_to_signature(structure, arg_names=None):
  def encode_arg(arg, path):
    if isinstance(arg, ops.Tensor):
      user_specified_name = None
      try:
        user_specified_name = compat.as_str(
            arg.op.get_attr("_user_specified_name"))
      except ValueError:
        pass
      if path and user_specified_name and user_specified_name != path[0]:
        name = user_specified_name
      else:
        name = "/".join(str(p) for p in path)
      return tensor_spec.TensorSpec(arg.shape, arg.dtype, name)
    if isinstance(arg, composite_tensor.CompositeTensor):
    if isinstance(arg, resource_variable_ops.BaseResourceVariable):
      return resource_variable_ops.VariableSpec.from_value(arg)
    if isinstance(arg, (
        int,
        float,
        bool,
        str,
        type(None),
        dtypes.DType,
        tensor_spec.TensorSpec,
        type_spec.TypeSpec,
    )):
      return arg
    return UnknownArgument()
  flattened = nest.flatten_with_tuple_paths(structure)
  if arg_names:
    if len(arg_names) != len(structure):
      raise ValueError(
          "Passed in arg_names don't match actual signature (%s)." % arg_names)
    flattened = [
        ((arg_names[path[0]],) + path[1:], arg) for path, arg in flattened
    ]
  mapped = [encode_arg(arg, path) for path, arg in flattened]
  return nest.pack_sequence_as(structure, mapped)
@tf_export("__internal__.FuncGraph", v1=[])
class FuncGraph(ops.Graph):
  """Graph representing a function body.
  Attributes:
    name: The name of the function.
    inputs: Placeholder tensors representing the inputs to this function. The
      tensors are in this FuncGraph. This represents "regular" inputs as well as
      captured inputs (i.e. the values of self.captures), with the regular
      inputs coming first.
    outputs: Tensors that will be returned by this function. The tensors are in
      this FuncGraph.
    control_outputs: Operations that must be executed before the function
      represented by this graph can be said to have been executed.
    structured_input_signature: A tuple of (args, kwargs), which are both
      possibly-nested python objects that were received by this function. Note
      that these structures might contain Python `None`s.
    structured_outputs: A possibly-nested python object which will be returned
      by this function. The Tensors in this structure are the same as those of
      self.outputs. Note that this structure might contain Python `None`s.
    variables: Variables that should be watched during function execution.
    outer_graph: The graph this function is defined in. May be another FuncGraph
      or the global default Graph.
    captures: Maps external tensor -> internal tensor (i.e. input placeholder).
      The entries are in the order they were captured.
    control_captures: Set of external ops on which this graph has a control
      dependency.
    seed: The graph-level random seed.
    capture_by_value: If True, the func graph will capture Variables by value
      instead of reference.
  """
  def __init__(self,
               name,
               collections=None,
               capture_by_value=None,
               structured_input_signature=None,
               structured_outputs=None):
    """Construct a new FuncGraph.
    The graph will inherit its graph key, collections, seed, and distribution
    strategy stack from the current context or graph.
    Args:
      name: the name of the function.
      collections: a dictionary of collections this FuncGraph should start with.
        If not specified (None), the FuncGraph will read (but not write to) the
        outer graph's collections that are not allowlisted, and both read and
        write to the outer graph's collections that are allowlisted. The current
        allowlisted collections are the global variables, the local variables,
        and the trainable variables. Defaults to None.
      capture_by_value: An optional boolean. If True, the func graph will
        capture Variables by value instead of reference. By default inherit from
        outer graphs, and failing that will default to False.
      structured_input_signature: Optional. The structured input signature to
        use for initializing the FuncGraph. See the docstring for FuncGraph for
        more information.
      structured_outputs: Optional. The structured outputs to use for
        initializing the FuncGraph. See the docstring for FuncGraph for more
        information.
    """
    super(FuncGraph, self).__init__()
    self.name = name
    self.inputs = []
    self.outputs = []
    self.control_outputs = []
    self.control_captures = object_identity.ObjectIdentitySet()
    self.structured_input_signature = structured_input_signature
    self.structured_outputs = structured_outputs
    self._weak_variables = []
    self._watched_variables = object_identity.ObjectIdentityWeakSet()
    self.is_control_flow_graph = False
    outer_graph = ops.get_default_graph()
    self._weak_outer_graph = weakref.ref(outer_graph)
    while outer_graph.building_function:
      outer_graph = outer_graph.outer_graph
    self._fallback_outer_graph = outer_graph
    self._captures = py_collections.OrderedDict()
    self._output_names = None
    self._deferred_captures = py_collections.OrderedDict()
    if capture_by_value is not None:
      self.capture_by_value = capture_by_value
    elif self.outer_graph is not None and isinstance(self.outer_graph,
                                                     FuncGraph):
      self.capture_by_value = self.outer_graph.capture_by_value
    else:
      self.capture_by_value = False
    self._building_function = True
    self._last_op_using_resource_tensor = {}
    graph = self.outer_graph
    if context.executing_eagerly():
      self.seed = context.global_seed()
      self._seed_used = False
    else:
      self.seed = graph.seed
      self._seed_used = False
    if collections is None:
      for collection_name in graph.get_all_collection_keys():
        if collection_name not in ALLOWLIST_COLLECTIONS:
          self._collections[collection_name] = graph.get_collection(
              collection_name)
      for collection_name in ALLOWLIST_COLLECTIONS:
        self._collections[collection_name] = graph.get_collection_ref(
            collection_name)
    else:
      self._collections = collections
    self._saveable = True
    self._saving_errors = set()
    self._scope_exit_callbacks = None
  def __str__(self):
    return "FuncGraph(name=%s, id=%s)" % (self.name, id(self))
  def watch_variable(self, v):
    while self is not None and isinstance(self, FuncGraph):
      self._watched_variables.add(v)
      self = self.outer_graph
  def capture_call_time_value(self,
                              closure,
                              spec,
                              key=None,
                              default_value=None,
                              placeholder=None):
    """Returns a placeholder which at call time has the value closure().
    The `tf.function` supports the notion of captures, that is, it allows Python
    functions to have closure variables, which bind over some value outside the
    function. However, this name binding is "early binding" performed before the
    program is run, i.e.,
    ```
    @tf.function
    def f():
      return x
    x = tf.constant(1)
    x = tf.constant(2)
    ```
    while in Python, name binding is performed as the program is running.
    ```
    def f():
      return x
    x = 1
    x = 2
    ```
    `capture_call_time_value` allows tf.function to mimic late binding as a
    Python function does, by passing in a `closure` callable argument to be
    executed when the tf.function is invoked eagerly.  E.g.
    ```
    @tf.function
    def f():
      return ops.get_default_graph.capture_call_time_value(lambda: x)
    x = tf.constant(1)
    x = tf.constant(2)
    ```
    Note that a `capture_call_time_value` function itself does not work well in
    the saving process (since the tf.function in which it's called is not
    invoked eagerly) unless passed a `default_value` argument. At saving time,
    the `default_value` argument is returned instead.
    Args:
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      key: optional. If not None, multiple calls to lazy_capture with the same
        key in the same graph will return the same placeholder, and the first
        closure will be used at function call time.
      default_value: optional value to return in environments that cannot safely
        evaluate closure.
      placeholder: optional. If not None, the graph will take the passed-in
        `placeholder` as the internal capture instead of creating a new one.
        This is useful when loading from a SavedModel.
    Returns:
      Nest of placeholders which, at function call time, will be fed with the
      result of calling closure().
    Raises:
      ValueError: at function call time, if the return value of closure() is
       not compatible with `spec`.
    """
    if key is None:
      key = object()
    if key not in self._deferred_captures:
      if placeholder is None:
        def convert_to_placeholder(s):
          if not isinstance(s, tensor_spec.DenseSpec):
            raise TypeError(
                "Expected a nest of `TypeSpec` objects, found %s of type %s." %
                (s, type(s)))
          return array_ops.placeholder(dtype=s.dtype, shape=s.shape)
        placeholder = nest.map_structure(
            convert_to_placeholder, spec, expand_composites=True)
      def wrapped_closure():
        if save_context.in_save_context() and default_value is not None:
          return default_value
        if not context.executing_eagerly():
          graph = ops.get_default_graph()
          while graph.is_control_flow_graph:
            graph = graph.outer_graph
          with graph.as_default():
            ret_nest = graph.capture_call_time_value(
                closure, spec, key=key, default_value=default_value)
        else:
          ret_nest = closure()
        nest.assert_same_structure(spec, ret_nest, expand_composites=True)
        y = nest.map_structure(
            lambda s, r: s._to_components(r),
            spec,
            ret_nest,
            expand_composites=False)
        return nest.flatten(y, expand_composites=True)
      wrapped_closure.output_spec = spec
      self._deferred_captures[key] = (wrapped_closure, placeholder)
    return self._deferred_captures[key][1]
  def control_dependencies(self, control_inputs):
    if control_inputs is None:
      return super(FuncGraph, self).control_dependencies(control_inputs)
    filtered_control_inputs = []
    for c in control_inputs:
      if (isinstance(c, indexed_slices.IndexedSlices) or
          (hasattr(c, "_handle") and hasattr(c, "op"))):
        c = c.op
      if graph_element is None:
        graph_element = c
      if graph_element is not None and getattr(graph_element, "graph",
                                               None) is not self:
        self.control_captures.add(graph_element)
      else:
        filtered_control_inputs.append(graph_element)
    return super(FuncGraph, self).control_dependencies(filtered_control_inputs)
  def as_default(self):
    outer_cm = super(FuncGraph, self).as_default()
    @tf_contextlib.contextmanager
    def inner_cm():
      graph = ops.get_default_graph()
      old_strategy_stack = self._distribution_strategy_stack
      self._distribution_strategy_stack = list(
          graph._distribution_strategy_stack)
      old_device_stack = self._device_function_stack
      if (not context.executing_eagerly() and
          (device_stack_has_callable(graph._device_function_stack) or
           (self._distribution_strategy_stack and
            not ops.executing_eagerly_outside_functions()))):
        self._device_function_stack = graph._device_function_stack.copy()
      old_creator_stack = self._variable_creator_stack
      self._variable_creator_stack = graph._variable_creator_stack
      old_graph_key = self._graph_key
      self._graph_key = graph._graph_key
      old_scope_exit_callbacks = self._scope_exit_callbacks
      self._scope_exit_callbacks = []
      with outer_cm as g:
        try:
          yield g
        finally:
          try:
            for fn in self._scope_exit_callbacks:
              fn()
          finally:
            self._scope_exit_callbacks = old_scope_exit_callbacks
            self._distribution_strategy_stack = old_strategy_stack
            self._device_function_stack = old_device_stack
            self._variable_creator_stack = old_creator_stack
            self._graph_key = old_graph_key
    return inner_cm()
  @property
  def outer_graph(self):
    """The Graph this FuncGraph is nested in.
    Functions may capture Tensors from graphs they are nested in (transitive).
    Returns:
      A Graph object. Initially set to the current default graph when the
      FuncGraph was created. If the previous `outer_graph` was deleted because
      the function that owns it was deleted, `outer_graph` is reset to the
      outermost default graph active when the FuncGraph was created. This
      FuncGraph won't have captured anything from the new `outer_graph` (and
      likely not from the previous setting, since that would have created a
      strong reference), but it is returned so that FuncGraphs always have a
      parent.
    """
    current = self._weak_outer_graph()
    if current is None:
      return self._fallback_outer_graph
    return current
  @outer_graph.setter
  def outer_graph(self, new_outer_graph):
    self._weak_outer_graph = weakref.ref(new_outer_graph)
  @property
  def output_types(self):
    return [t.dtype for t in self.outputs]
  @property
  def output_shapes(self):
    return [t.shape for t in self.outputs]
  @property
  def trainable_variables(self):
    return tuple(v for v in self.variables if v.trainable)
  @property
  def variables(self):
    def deref(weak_v):
      v = weak_v()
      if v is None:
        raise AssertionError(
            "Called a function referencing variables which have been deleted. "
            "This likely means that function-local variables were created and "
            "not referenced elsewhere in the program. This is generally a "
            "mistake; consider storing variables in an object attribute on "
            "first call.")
      return v
    return tuple(deref(v) for v in self._weak_variables)
  @variables.setter
  def variables(self, var_list):
    self._weak_variables = [weakref.ref(v) for v in var_list]
  def _capture_by_value(
      self,
      op_type,
      inputs,
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    reverse_captures = dict((id(v), k) for k, v in self.captures)
    uncaptured_inputs = [reverse_captures.get(id(t), t) for t in inputs]
    with ops.init_scope():
      if context.executing_eagerly():
        attr_list = ("dtype", int(attrs["dtype"].type))
        value, = execute.execute(
            compat.as_bytes(op_type), 1, uncaptured_inputs, attr_list,
            context.context())
      else:
            op_type, uncaptured_inputs, dtypes, input_types, name, attrs,
            op_def, compute_device)
        value = op.outputs[0]
    captured_value = self.capture(value)
    return captured_value.op
  def _create_op_internal(
      self,
      op_type,
      inputs,
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    """Like Graph.create_op, except handles external input tensors.
    This overload adds functionality to create_op to "capture" any external
    input tensors, i.e. tensors from the eager context or outer function graphs
    if this is a nested function. See `capture` for more information.
    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: (Optional) A list of `DType` objects that will be the types of the
        tensors that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of the
        tensors that the operation consumes. By default, uses the base `DType`
        of each input in `inputs`. Operations that expect reference-typed inputs
        must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.
    Returns:
      An `Operation` object.
    """
    if self.capture_by_value and op_type in [
        "ReadVariableOp", "ResourceGather"
    ]:
      return self._capture_by_value(op_type, inputs, dtypes, input_types, name,
                                    attrs, op_def, compute_device)
    if op_type == "Enter" and inputs[0].op.type == "Enter":
      if inputs[0].op.get_attr("frame_name") == attrs["frame_name"].s:
        return inputs[0].op
    captured_inputs = []
    for inp in inputs:
      if ctxt is not None and hasattr(ctxt, "AddValue"):
        inp = ctxt.AddValue(inp)
      inp = self.capture(inp)
      captured_inputs.append(inp)
        op_type, captured_inputs, dtypes, input_types, name, attrs, op_def,
        compute_device)
  def capture(self, tensor, name=None, shape=None):
    if isinstance(tensor, ops.EagerTensor):
      if name is None:
        name = str(ops.uid())
      if (tensor.dtype in dtypes.TF_VALUE_DTYPES and
          np.prod(tensor.shape) <= _EAGER_CONST_THRESHOLD):
        return self.capture_eager_tensor(tensor, name)
      return self._capture_helper(tensor, name, shape)
    if tensor.graph is not self:
      if name is None:
        name = tensor.op.name
      inner_graph = tensor.graph
      while inner_graph is not None and isinstance(inner_graph, FuncGraph):
        if inner_graph is self:
          try:
            tb = tensor.op.traceback
          except AttributeError:
            tensor_traceback = "<unknown>"
          else:
            tensor_traceback_list = []
            for frame in traceback.format_list(tb.get_user_frames()):
              tensor_traceback_list.extend(
                  [f"  {line}" for line in frame.split("\n") if line.strip()])
            tensor_traceback = "\n".join(tensor_traceback_list)
          raise errors.InaccessibleTensorError(
              f"{tensor!r} is out of scope and cannot be used here. Use return "
              "values, explicit Python locals or TensorFlow collections to "
              "access it.\n"
              "for more information.\n\n"
              f"{tensor!r} was defined here:\n{tensor_traceback}\n\n"
              f"The tensor {tensor!r} cannot be accessed from {self}, because "
              f"it was defined in {tensor.graph}, which is out of scope.")
        inner_graph = inner_graph.outer_graph
      return self._capture_helper(tensor, name)
    return tensor
  def _capture_helper(self, tensor, name, shape=None):
    capture = self._captures.get(id(tensor))
    if capture is None:
      placeholder = _create_substitute_placeholder(
          tensor, name=name, dtype=tensor.dtype, shape=shape)
      if isinstance(tensor, ops.EagerTensor) and tensor.is_packed:
            "_composite_device",
            attr_value_pb2.AttrValue(s=compat.as_bytes(tensor.device)))
      self.add_capture(tensor, placeholder)
    else:
      placeholder = capture[1]
    tape.record_operation(
        "captured_value", [placeholder], [tensor],
        backward_function=lambda x: [x],
        forward_function=lambda x: [x])
    return placeholder
  @property
  def captures(self):
    return self._captures.values()
  def add_capture(self, tensor, placeholder):
    self._captures[id(tensor)] = (tensor, placeholder)
    self.inputs.append(placeholder)
  def replace_capture(self, tensor, placeholder):
    self._captures[id(tensor)] = (tensor, placeholder)
  def replace_capture_with_deferred_capture(self,
                                            tensor,
                                            closure,
                                            spec,
                                            placeholder,
                                            default_value=None):
    """Replaces existing capture `tensor` with a deferred capture `closure`.
    Caution: It is the caller's responsibility to make sure that, after calling
    this function, the TypeSpec of the `inputs` (i.e. internal placeholders) and
    the `_captured_inputs` (i.e. external captures) of a concrete function that
    wraps this function graph are still compatible. Thus user should pairing
    usage of this function with `ConcreteFunction.set_external_captures` to make
    sure the order still matches. For example,
    ```
    concrete_fn.graph.replace_capture_with_deferred_capture(tensor2,
                                                            closure2,
                                                            placeholder2,
                                                            some_spec,
                                                            some_default)
    concrete_fn.set_external_captures([tensor1, closure2, tensor3])
    ```
    Args:
      tensor: Tensor already captured.
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      placeholder: the internal placeholder corresponding to the captured
        `tensor`.
      default_value: optional value to use in environments that cannot safely
        evaluate closure.
    """
    if id(tensor) in self._captures:
      self.pop_capture(tensor)
    self.capture_call_time_value(
        closure,
        spec,
        key=id(tensor),
        default_value=default_value,
        placeholder=placeholder)
  def reset_captures(self, capture_list):
    self._captures = py_collections.OrderedDict()
    for tensor, placeholder in capture_list:
      self._captures[id(tensor)] = (tensor, placeholder)
  def pop_capture(self, tensor):
    capture = self._captures.pop(id(tensor), None)
    if capture is None:
      return None
    return capture[1]
  def clear_captures(self):
    while self._captures:
      self._captures.popitem()
    memory.dismantle_ordered_dict(self._captures)
    while self._deferred_captures:
      self._deferred_captures.popitem()
    memory.dismantle_ordered_dict(self._deferred_captures)
  def capture_distributed_variable(self, variable, placeholder):
    self._captures[id(variable)] = (variable, placeholder)
    tape.record_operation(
        "captured_value", [placeholder], [variable],
        backward_function=lambda x: [x],
        forward_function=lambda x: [x])
  def capture_eager_tensor(self, tensor, name):
    capture = self._captures.get(id(tensor))
    if capture is None:
      with ops.control_dependencies(None):
        constant_value = tensor_util.constant_value(tensor)
        if constant_value is None:
          return self._capture_helper(tensor, name)
        graph_const = constant_op.constant(
            constant_value, dtype=tensor.dtype, shape=tensor.shape, name=name)
      self.add_capture(tensor, graph_const)
    else:
      graph_const = capture[1]
    tape.record_operation(
        "captured_value", [graph_const], [tensor],
        backward_function=lambda x: [x],
        forward_function=lambda x: [x])
    return graph_const
  def captured(self, tensor):
    return id(tensor) in self._captures
  @property
  def external_captures(self):
    return [c[0] for c in self._captures.values()]
  @property
  def internal_captures(self):
    return [c[1] for c in self._captures.values()]
  @property
  def deferred_external_captures(self):
    return [c[0] for c in self._deferred_captures.values()]
  @property
  def deferred_internal_captures(self):
    return [c[1] for c in self._deferred_captures.values()]
  @property
  def variable_captures(self):
    return {
        id(self._captures[id(v)][1]): v
        for v in self.variables
        if id(v) in self._captures
    }
  def mark_as_unsaveable(self, error_message):
    self._saveable = False
    if isinstance(error_message, str):
      error_message = [error_message]
    self._saving_errors.update(error_message)
  @property
  def saveable(self):
    return self._saveable
  @property
  def saving_errors(self):
    return self._saving_errors
  def _add_scope_exit_callback(self, fn):
    if not callable(fn):
      raise TypeError("fn is not callable: {}".format(fn))
    if self._scope_exit_callbacks is None:
      raise RuntimeError(
          "Attempting to add a scope exit callback, but the default graph is "
          "not the context scope graph.  Did you forget to call "
          "'with graph.as_default(): ...'?")
    self._scope_exit_callbacks.append(fn)
def func_graph_from_py_func(name,
                            python_func,
                            args,
                            kwargs,
                            signature=None,
                            func_graph=None,
                            autograph=False,
                            autograph_options=None,
                            add_control_dependencies=True,
                            arg_names=None,
                            op_return_value=None,
                            collections=None,
                            capture_by_value=None,
                            acd_record_initial_resource_uses=False):
  """Returns a `FuncGraph` generated from `python_func`.
  Args:
    name: an identifier for the function.
    python_func: the Python function to trace.
    args: the positional args with which the Python function should be called;
      ignored if a signature is provided.
    kwargs: the keyword args with which the Python function should be called;
      ignored if a signature is provided.
    signature: a possibly nested sequence of `TensorSpecs` specifying the shapes
      and dtypes of the arguments. When a signature is provided, `args` and
      `kwargs` are ignored, and `python_func` is traced with Tensors conforming
      to `signature`. If `None`, the shapes and dtypes are inferred from the
      inputs.
    func_graph: Optional. An instance of FuncGraph. If provided, we will use
      this graph else a new one is built and returned.
    autograph: whether to use autograph to compile `python_func`.
      See https://www.tensorflow.org/guide/autograph for more information.
    autograph_options: additional knobs to control when `autograph=True`.
      See https://www.tensorflow.org/guide/autograph for more information.
    add_control_dependencies: If True, automatically adds control dependencies
      to ensure program order matches execution order and stateful ops always
      execute.
    arg_names: Optional list of argument names, used to give input placeholders
      recognizable names.
    op_return_value: Optional. A Tensor. If set and `python_func` returns
      Operations, those return values will be replaced with this value. If not
      set, returning an Operation triggers an error.
    collections: a dictionary of collections this FuncGraph should start with.
      If not specified (None), the FuncGraph will read (but not write to) the
      outer graph's collections that are not allowlisted, and both read and
      write to the outer graph's collections that are allowlisted. The current
      allowlisted collections are the global variables, the local variables, and
      the trainable variables. Defaults to None.
    capture_by_value: An optional boolean. If True, the func graph will capture
      Variables by value instead of reference. By default inherit from outer
      graphs, and failing that will default to False.
    acd_record_initial_resource_uses: If `True` and `add_control_dependencies`
      is enabled, the results (those marked with
      AutomaticControlDependencies.mark_result) will be annotated with a private
      attribute, "_res_first_used_by", which points to the first nodes which
      used the any of the resources that the result op is using.
  Returns:
    A FuncGraph.
  Raises:
    TypeError: If any of `python_func`'s return values is neither `None`, a
      `Tensor` or a `tf.experimental.ExtensionType`.
  """
  if op_return_value is not None:
    assert isinstance(op_return_value, ops.Tensor), op_return_value
  if func_graph is None:
    func_graph = FuncGraph(
        name, collections=collections, capture_by_value=capture_by_value)
  assert isinstance(func_graph, FuncGraph)
  if add_control_dependencies:
    deps_control_manager = auto_control_deps.AutomaticControlDependencies(
        record_initial_resource_uses=acd_record_initial_resource_uses)
  else:
    deps_control_manager = ops.NullContextmanager()
  with func_graph.as_default(), deps_control_manager as deps_ctx:
    current_scope = variable_scope.get_variable_scope()
    default_use_resource = current_scope.use_resource
    current_scope.set_use_resource(True)
    if signature is not None:
      args = signature
      kwargs = {}
    func_args = _get_defun_inputs_from_args(args, arg_names)
    func_kwargs = _get_defun_inputs_from_kwargs(kwargs)
    func_graph.structured_input_signature = (convert_structure_to_signature(
        func_args, arg_names), convert_structure_to_signature(func_kwargs))
    flat_func_args = nest.flatten(func_args, expand_composites=True)
    flat_func_kwargs = nest.flatten(func_kwargs, expand_composites=True)
    func_graph.inputs = [
        arg for arg in flat_func_args + flat_func_kwargs
        if isinstance(arg, ops.Tensor)
    ]
    func_args_before = nest.pack_sequence_as(
        func_args, flat_func_args, expand_composites=True)
    func_kwargs_before = nest.pack_sequence_as(
        func_kwargs, flat_func_kwargs, expand_composites=True)
    def convert(x):
      if x is None:
        return None
      if op_return_value is not None and isinstance(x, ops.Operation):
        with ops.control_dependencies([x]):
          x = array_ops.identity(op_return_value)
      elif not isinstance(x, tensor_array_ops.TensorArray):
        try:
          x = ops.convert_to_tensor_or_composite(x)
        except (ValueError, TypeError):
          raise TypeError(
              "To be compatible with tf.function, Python functions "
              "must return zero or more Tensors or ExtensionTypes or None "
              f"values; in compilation of {str(python_func)}, found return "
              f"value of type {type(x).__name__}, which is not a Tensor or "
              "ExtensionType.")
      if add_control_dependencies:
        x = deps_ctx.mark_as_return(x)
      return x
    try:
      if autograph:
        _, original_func = tf_decorator.unwrap(python_func)
        def autograph_handler(*args, **kwargs):
          try:
            return autograph.converted_call(
                original_func,
                args,
                kwargs,
                options=autograph.ConversionOptions(
                    recursive=True,
                    optional_features=autograph_options,
                    user_requested=True,
                ))
            if hasattr(e, "ag_error_metadata"):
              raise e.ag_error_metadata.to_exception(e)
            else:
              raise
        converted_func = tf_decorator.make_decorator(original_func,
                                                     autograph_handler)
        python_func = tf_decorator.rewrap(python_func, original_func,
                                          converted_func)
      else:
        _, original_func = tf_decorator.unwrap(python_func)
      func_outputs = python_func(*func_args, **func_kwargs)
      func_outputs = nest.map_structure(
          convert, func_outputs, expand_composites=True)
      check_func_mutation(func_args_before, func_kwargs_before, func_args,
                          func_kwargs, original_func)
    finally:
      current_scope.set_use_resource(default_use_resource)
    arg_variables = object_identity.ObjectIdentitySet()
    inputs = []
    for arg in (nest.flatten(func_args, expand_composites=True) +
                nest.flatten(func_kwargs, expand_composites=True)):
      if isinstance(arg, resource_variable_ops.BaseResourceVariable):
        resource_placeholder = func_graph.pop_capture(arg.handle)
        if resource_placeholder is None:
          continue
        arg_variables.add(arg)
        inputs.append(resource_placeholder)
      elif isinstance(arg, ops.Tensor):
        inputs.append(arg)
    variables = [v for v in graph_variables if v not in arg_variables]
    func_graph.inputs = (
        inputs + func_graph.internal_captures + nest.flatten(
            func_graph.deferred_internal_captures, expand_composites=True))
    func_graph.structured_outputs = func_outputs
    func_graph.outputs.extend(
        func_graph.capture(x)
        for x in flatten(func_graph.structured_outputs)
        if x is not None)
    func_graph.variables = variables
  if add_control_dependencies:
    func_graph.control_outputs.extend(deps_control_manager.ops_which_must_run)
    func_graph.collective_manager_ids_used = (
        deps_control_manager.collective_manager_ids_used)
  return func_graph
def maybe_captured(tensor):
  if (not isinstance(tensor, ops.EagerTensor) and
      tensor.op.graph.building_function and tensor.op.type == "Placeholder"):
    for input_t, placeholder_t in tensor.op.graph.captures:
      if tensor == placeholder_t:
        return maybe_captured(input_t)
  return tensor
def device_stack_has_callable(device_stack):
  return any(
      for spec in device_stack.peek_objs())
def has_mutation(n1, n2):
  try:
    nest.assert_same_structure(n1, n2, expand_composites=True)
  except ValueError:
    return True
  for arg1, arg2 in zip(
      nest.flatten(n1, expand_composites=True),
      nest.flatten(n2, expand_composites=True)):
    if arg1 is not arg2:
      return True
  return False
def check_func_mutation(old_args, old_kwargs, new_args, new_kwargs, func):
  if not has_mutation((old_args, old_kwargs), (new_args, new_kwargs)):
    return
  func_name = getattr(func, "__qualname__", getattr(func, "__name__", func))
  signature = tf_inspect.signature(func)
  try:
    old_bound = signature.bind(*old_args, **old_kwargs).arguments
    new_bound = signature.bind(*new_args, **new_kwargs).arguments
  except TypeError as e:
    raise ValueError(
        f"{func_name}{signature} should not modify its Python input "
        f"arguments. Check if it modifies any lists or dicts passed as "
        f"arguments. Modifying a copy is allowed.") from e
  assert set(old_bound) == set(new_bound)
  modified_args = [
      arg_name for arg_name in new_bound
      if has_mutation(old_bound[arg_name], new_bound[arg_name])
  ]
  changes = ", ".join(modified_args)
  raise ValueError(f"{func_name}{signature} should not modify its Python "
                   f"input arguments. Modifying a copy is allowed. The "
                   f"following parameter(s) were modified: {changes}")
def flatten(sequence):
  flat_sequence = nest.flatten(sequence, expand_composites=True)
  return [
      item.flow if isinstance(item, tensor_array_ops.TensorArray) else item
      for item in flat_sequence
  ]
def pack_sequence_as(structure, flat_sequence):
  flat_sequence = list(flat_sequence)
  flattened_structure = nest.flatten(structure, expand_composites=True)
  if len(flattened_structure) != len(flat_sequence):
    raise ValueError("Mismatch in element count")
  for i in range(len(flat_sequence)):
    if isinstance(flattened_structure[i], tensor_array_ops.TensorArray):
      flat_sequence[i] = tensor_array_ops.build_ta_with_new_flow(
          old_ta=flattened_structure[i], flow=flat_sequence[i])
  return nest.pack_sequence_as(structure, flat_sequence, expand_composites=True)
def _create_substitute_placeholder(value, name=None, dtype=None, shape=None):
  if shape is None:
    shape = value.shape
  with ops.control_dependencies(None):
    placeholder = graph_placeholder(
        dtype=dtype or value.dtype, shape=shape, name=name)
  handle_data_util.copy_handle_data(value, placeholder)
  return placeholder
def _get_defun_inputs_from_args(args, names):
  return _get_defun_inputs(args, names, structured_args=args)
def _get_defun_inputs_from_kwargs(kwargs):
  if kwargs:
    names, args = zip(*sorted(kwargs.items()))
  else:
    names = []
    args = []
  return _get_defun_inputs(args, names, structured_args=kwargs)
def _get_composite_tensor_spec(x):
          if isinstance(x, composite_tensor.CompositeTensor) else x)
def _get_defun_inputs(args, names, structured_args):
  func_graph = ops.get_default_graph()
  function_inputs = []
  if names is None:
    names = [None] * len(args)
  for arg_value, name in zip(args, names):
    arg_value = nest.map_structure(_get_composite_tensor_spec, arg_value)
    flat_args = nest.flatten(arg_value, expand_composites=True)
    for arg in flat_args:
      if isinstance(arg, (ops.Tensor, tensor_spec.TensorSpec)):
        arg_is_spec = isinstance(arg, tensor_spec.TensorSpec)
        if arg_is_spec and arg.name:
          requested_name = arg.name
        else:
          requested_name = name
        try:
          placeholder = graph_placeholder(
              arg.dtype, arg.shape, name=requested_name)
        except ValueError:
          placeholder = graph_placeholder(arg.dtype, arg.shape)
        if not arg_is_spec:
          handle_data_util.copy_handle_data(arg, placeholder)
        if name is not None:
              "_user_specified_name",
              attr_value_pb2.AttrValue(s=compat.as_bytes(requested_name)))
        function_inputs.append(placeholder)
      elif isinstance(arg, (resource_variable_ops.BaseResourceVariable,
                            resource_variable_ops.VariableSpec)):
        if isinstance(arg, resource_variable_ops.VariableSpec):
          name = arg.name or name
          with func_graph.outer_graph.as_default():
            placeholder = graph_placeholder(
                dtypes.resource, arg.shape, name=name)
            arg = resource_variable_ops.BaseResourceVariable(
                name=name,
                shape=arg.shape,
                dtype=arg.dtype,
                handle=placeholder,
                handle_name=name,
                trainable=arg.trainable)
        placeholder = func_graph.capture(arg.handle, name=name)
            "_user_specified_name",
            attr_value_pb2.AttrValue(s=compat.as_bytes(name)))
        function_inputs.append(arg)
      else:
        function_inputs.append(arg)
  return nest.pack_sequence_as(
      structured_args, function_inputs, expand_composites=True)
def dismantle_func_graph(func_graph):
  """Removes reference cycles in `func_graph` FuncGraph.
  Helpful for making sure the garbage collector doesn't need to run when
  the FuncGraph goes out of scope, e.g. in tests using defun with
  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True).
  Args:
    func_graph: A `FuncGraph` object to destroy. `func_graph` is unusable after
      this function.
  """
  func_graph.clear_captures()
  ops.dismantle_graph(func_graph)
def override_func_graph_name_scope(func_graph, name_scope):
