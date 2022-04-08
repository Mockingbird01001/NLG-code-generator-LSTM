
import weakref
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class VariableHolder(object):
  def __init__(self, fn=None, share_variables=False):
    self._fn = fn
    self._share_variables = share_variables
    self._variables_by_name = data_structures.Mapping()
  @property
  def variables(self):
    return self._variables_by_name
  def variable_creator_scope(self, next_creator, **kwargs):
    collections = kwargs.pop("collections", None)
    v = None
    with ops.name_scope(
        kwargs.get("name", None), "Variable", skip_on_eager=False) as name:
      variable_name = ops.name_from_scope_name(name)
      kwargs["name"] = name
    if self._share_variables:
      v = self._variables_by_name.get(variable_name, None)
    if v is None:
      v = next_creator(**kwargs)
      self._variables_by_name[variable_name] = v
    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if v.trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    ops.add_to_collections(collections, v)
    return v
  def __call__(self, *args, **kwargs):
    return self.call_with_variable_creator_scope(self._fn)(*args, **kwargs)
  def call_with_variable_creator_scope(self, fn):
    def wrapped(*args, **kwargs):
      with variable_scope.variable_creator_scope(self.variable_creator_scope):
        return fn(*args, **kwargs)
    return wrapped
def _get_element_from_tensor_info(tensor_info, graph):
  encoding = tensor_info.WhichOneof("encoding")
  if encoding == "name":
    return graph.as_graph_element(tensor_info.name)
  elif encoding == "coo_sparse":
    return sparse_tensor.SparseTensor(
        graph.get_tensor_by_name(tensor_info.coo_sparse.indices_tensor_name),
        graph.get_tensor_by_name(tensor_info.coo_sparse.values_tensor_name),
        graph.get_tensor_by_name(
            tensor_info.coo_sparse.dense_shape_tensor_name))
  elif encoding == "composite_tensor":
    spec_proto = struct_pb2.StructuredValue(
        type_spec_value=tensor_info.composite_tensor.type_spec)
    spec = nested_structure_coder.decode_proto(spec_proto)
    components = [graph.get_tensor_by_name(component.name) for component in
                  tensor_info.composite_tensor.components]
  else:
    raise ValueError(f"Invalid TensorInfo.encoding: {encoding}. Valid "
                     "encodings are 'name', 'coo_sparse', and "
                     "'composite_tensor'.")
def _lift_single_variable(old_variable, graph, variable_holder):
  new_variable = resource_variable_ops.UninitializedVariable(
      shape=old_variable.shape,
      dtype=old_variable.dtype,
      name=old_variable.op.name,
      trainable=old_variable.trainable,
      extra_handle_data=old_variable.handle)
  graph.add_capture(new_variable.handle, old_variable.handle)
  graph.capture(new_variable.handle)
  variable_name = new_variable.name.split(":")[0]
  variable_holder._variables_by_name[variable_name] = new_variable
  graph._weak_variables.append(weakref.ref(new_variable))
  graph.watch_variable(new_variable)
  return new_variable
def _lift_unlifted_variables(graph, variable_holder):
  with graph.as_default():
    global_collection_variables = ops.get_collection(
        ops.GraphKeys.GLOBAL_VARIABLES)
    local_collection_variables = ops.get_collection(
        ops.GraphKeys.LOCAL_VARIABLES)
    existing_captures = {id(c) for c in graph.internal_captures}
    lifted_variables = {}
    def _should_lift_variable(v):
               and v.graph.building_function)
              and isinstance(v, resource_variable_ops.BaseResourceVariable)
              and id(v.handle) not in existing_captures)
    for old_variable in global_collection_variables:
      if _should_lift_variable(old_variable):
        new_variable = _lift_single_variable(
            old_variable, graph, variable_holder)
        lifted_variables[id(old_variable)] = new_variable
        existing_captures.add(id(old_variable.handle))
    for old_variable in local_collection_variables:
      if _should_lift_variable(old_variable):
        new_variable = _lift_single_variable(
            old_variable, graph, variable_holder)
        lifted_variables[id(old_variable)] = new_variable
        existing_captures.add(id(old_variable.handle))
          outer_graph = new_variable.graph
          global_collection = outer_graph.get_collection_ref(
              ops.GraphKeys.GLOBAL_VARIABLES)
          global_collection.remove(new_variable)
          outer_graph.add_to_collection(
              ops.GraphKeys.LOCAL_VARIABLES, new_variable)
    for collection_name in [
        ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.LOCAL_VARIABLES
    ]:
      mutable_collection = ops.get_collection_ref(collection_name)
      for index, current in enumerate(mutable_collection):
        mutable_collection[index] = lifted_variables.get(id(current), current)
        if not resource_variable_ops.is_resource_variable(
            mutable_collection[index]):
          logging.log_first_n(
              logging.WARN,
              "Unable to create a python object for variable {} because it is "
              "a reference variable. It may not be visible to training APIs. "
              "If this is a problem, consider rebuilding the SavedModel after "
              "running tf.compat.v1.enable_resource_variables().".format(
                  mutable_collection[index]),
              5)
class WrappedFunction(function.ConcreteFunction):
  def __init__(self, fn_graph, variable_holder, attrs=None, signature=None):
    self._variable_holder = variable_holder
    _lift_unlifted_variables(fn_graph, variable_holder)
    for f in fn_graph.as_graph_def().library.function:
      context.context().add_function_def(f)
    self._signature = signature
    super(WrappedFunction, self).__init__(fn_graph, attrs=attrs)
  def _call_impl(self, args, kwargs, cancellation_manager=None):
    if self._arg_keywords is None:
      if kwargs:
        raise NotImplementedError(
            "Keyword arguments are not supported when calling a "
            f"wrap_function-decorated function. Got {kwargs}.")
      if self._signature is not None:
        args = list(args)
        for i, arg in enumerate(args):
          if isinstance(self._signature[i], tensor_spec.DenseSpec):
            args[i] = ops.convert_to_tensor(arg, self._signature[i].dtype)
      return self._call_flat(args, self.captured_inputs)
    else:
      return super(WrappedFunction, self)._call_impl(
          args, kwargs, cancellation_manager)
  def prune(self, feeds, fetches, name=None, input_signature=None):
    """Extract a subgraph of this function's underlying graph.
    Wraps the subgraph in a new `WrappedFunction` object.
    Args:
      feeds: Input tensors to the subgraph to extract, as `Tensor` objects.
      fetches: Possibly-nested Python data structure containing information
        about outputs of the target subgraph. Each entry can either be a
        `Tensor` object (for data outputs), an `Operation` object (for control
        outputs), or a `TensorInfo` proto. Any additional shape/dtype
        information provided in a `TensorInfo` and not present in the original
        graph will be added to the returned subgraph.
      name: (optional) Name to give to the underlying `FuncGraph` of the
        returned object. If no name is provided, the graph's name will be
        `"pruned"`.
      input_signature: (optional) possibly-nested Python data structure
        containing `TensorSpec` objects, with which to populate the returned
        functions's `FuncGraph`'s `structured_input_signature` field.
    Returns:
      A new `WrappedFunction` object containing a copy of the portion of this
        object's graph that goes from `feeds` to `fetches`.
    """
    name = name or "pruned"
    flat_feeds = nest.flatten(feeds, expand_composites=True)
    flat_feeds = [self.graph.as_graph_element(t) for t in flat_feeds]
    for f in flat_feeds:
      if not isinstance(f, ops.Tensor):
        raise ValueError("All memebers of argument `feeds` must be tensors. "
                         f"Got {f} with type {type(f)}.")
    internal_captures = {id(c) for c in self.graph.internal_captures}
    flat_feeds = [f for f in flat_feeds if id(f) not in internal_captures]
    operation_fetches = []
    tensor_fetches = []
    tensor_infos = []
    def _fetch_preprocessing_callback(fetch):
      if isinstance(fetch, ops.Operation):
        operation_fetches.append(fetch)
        return fetch
      elif isinstance(fetch, meta_graph_pb2.TensorInfo):
        tensor_infos.append(fetch)
        decoded = _get_element_from_tensor_info(fetch, self._func_graph)
        if (tensor_util.is_tf_type(decoded) or
            isinstance(decoded, composite_tensor.CompositeTensor)):
          tensor_fetches.append(decoded)
        else:
          operation_fetches.append(decoded)
        return decoded
      elif isinstance(fetch, (ops.Tensor, composite_tensor.CompositeTensor)):
        tensor_fetches.append(fetch)
        return fetch
      else:
        graph_element = self.graph.as_graph_element(fetch)
        return _fetch_preprocessing_callback(graph_element)
    fetches = nest.map_structure(_fetch_preprocessing_callback, fetches)
    tensor_fetches = nest.flatten(tensor_fetches, expand_composites=True)
    for f in flat_feeds + tensor_fetches + operation_fetches:
      if f.graph is not self._func_graph:
        raise ValueError("Can only prune function whose feeds and fetches "
                         f"from graph {self._func_graph}. Input "
                         f"{f} is from a different graph {f.graph}.")
    with self._func_graph.as_default():
      pruned_graph = func_graph.FuncGraph(name)
    lift_map = lift_to_graph.lift_to_graph(
        operation_fetches + tensor_fetches,
        pruned_graph,
        sources=flat_feeds + self.graph.internal_captures,
        base_graph=self._func_graph)
    pruned_graph.outputs.extend(lift_map[x] for x in tensor_fetches)
    pruned_graph.control_outputs.extend(
        [lift_map[operation] for operation in operation_fetches])
    pruned_graph.inputs.extend(lift_map[x] for x in flat_feeds)
    for external_capture, internal_capture in self.graph.captures:
      pruned_graph.add_capture(external_capture, lift_map[internal_capture])
    for ti in tensor_infos:
        t = pruned_graph.as_graph_element(ti.name)
        if tensor_util.is_tf_type(t):
          t.set_shape(tensor_shape.TensorShape(ti.tensor_shape))
    for f in self.graph._functions.values():
      pruned_graph._add_function(f)
    pruned_graph.variables = self.graph.variables
    def _structured_output_mapping(fetched):
      lifted = lift_map[fetched]
      if isinstance(lifted, ops.Operation):
        return None
      return lifted
    pruned_graph.structured_outputs = nest.map_structure(
        _structured_output_mapping, fetches, expand_composites=True)
    pruned_graph.structured_input_signature = input_signature
    pruned_fn = WrappedFunction(
        pruned_graph, variable_holder=self._variable_holder)
    return pruned_fn
def _filter_returned_ops(fn):
  """Filtering out any ops returned by function.
  Args:
    fn: a function
  Returns:
    A tuple of (
      Wrapped function that returns `None` in place of any ops,
      dict that maps the index in the flat output structure to the returned op
    )
  """
  returned_ops = {}
  def wrap_and_filter_returned_ops(*args, **kwargs):
    outputs = fn(*args, **kwargs)
    flat_outputs = nest.flatten(outputs)
    for n in range(len(flat_outputs)):
      output = flat_outputs[n]
      if isinstance(output, ops.Operation):
        returned_ops[n] = output
        flat_outputs[n] = None
    return nest.pack_sequence_as(outputs, flat_outputs)
  return wrap_and_filter_returned_ops, returned_ops
class WrappedGraph(object):
  """Class for wrapping multiple TF 1.X functions in a single graph.
  Maintains a dictionary mapping names to wrapped functions. See
  `tf.compat.v1.wrap_function` to learn more about wrapping V1 functions.
  Functions wrapped using this class have access to variables and collections
  created in other wrapped functions, using the standard TF 1.X API (
  `tf.compat.v1.get_variable` or
  `tf.compat.v1.get_default_graph().get_collection(...)`)
  Outside a function, variables and collections may be accessed using the
  `variables` and `graph` properties.
  Example:
  ```
  def add_v1(x):
    with tf.compat.v1.variable_scope('vars', reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable('v', shape=[], dtype=tf.int32)
    return v + x
  def increment_var_v1(x):
    with tf.compat.v1.variable_scope('vars', reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable('v', shape=[], dtype=tf.int32)
    return v.assign_add(x)
  g = WrappedGraph()
  add = g.wrap_function(add_v1, [tf.TensorSpec([], tf.int32)])
  increment_var = g.wrap_function(increment_var_v1,
                                  [tf.TensorSpec([], tf.int32)])
  assert len(g.variables) == 1
  assert g.variables[0].numpy() == 0
  increment_var(tf.constant(5))
  assert g.variables[0].numpy() == 5
  ```
  """
  def __init__(self, variable_holder=None, **kwargs):
    self._variable_holder = (
        variable_holder or VariableHolder(share_variables=True))
    name = kwargs.pop("name", "wrapped_function_graph")
    collections = kwargs.pop("collections", {})
    self.graph = func_graph.FuncGraph(name, collections=collections, **kwargs)
    self._wrapped_function = WrappedFunction(self.graph, self._variable_holder)
    self._functions = {}
  @property
  def functions(self):
    return self._functions
  @property
  def variables(self):
    return self._variable_holder.variables
  def wrap_function(self, fn, signature, name=None):
    """Wraps a TF 1.X function and returns an eager-compatible function.
    All functions wrapped in the same `WrappedGraph` will have access to the
    same graph (`tf.compat.v1.get_default_graph` to get the graph object
    within a function, or `WrappedGraph.graph` to get the graph outside a
    function). Variables created within the function will be added to the
    `variables` list.
    Function inputs: All inputs to the function must be tensors (nested ok),
    with their shapes and dtypes defined in the `signature` argument.
    Function outputs:
      * The 1.X function may return tensors, variables, and ops. The wrapped
        eager-compatible function will always return tensors in the same nested
        structure.
      * Variables are replaced with a tensor containing the latest read values.
      * Returned ops are executed, and replaced with None.
      * The order of op execution and variable reads in the return is
        nondeterministic. For example:
        ```
        def update_var(x):
          v = tf.Variable(0)
          op = tf.compat.v1.assign(v, x).op
          return v, op
        g = WrappedGraph()
        fn = g.wrap_function(update_var)
        read_value, _ = fn(tf.constant(3))
        ```
    To ensure that ops in the function are executed (e.g. ops added to the
    `tf.GraphKeys.UPDATE_OPS` collection), include them in the function returns.
    Args:
      fn: a 1.X tensorflow function.
      signature: a possibly nested sequence of `TensorSpecs` specifying the
        shapes and dtypes of the arguments.
      name: an optional string name for the function. The function will be saved
        with key `name` in the `functions` dictionary.
    Returns:
      An eager-compatible function.
    """
    return self._wrap_function(fn, signature=signature, name=name)
  def _wrap_function(self,
                     fn,
                     args=None,
                     kwargs=None,
                     signature=None,
                     name=None):
    fn_with_filter_and_scope, returned_ops = _filter_returned_ops(
        self._variable_holder.call_with_variable_creator_scope(fn))
    func_graph.func_graph_from_py_func(
        fn_with_filter_and_scope,
        args=args,
        kwargs=kwargs,
        signature=signature,
        add_control_dependencies=False,
        func_graph=self.graph)
    fn_inputs = self.graph.inputs[:-len(self.graph.captures)]
    flat_fn_outputs = nest.flatten(self.graph.structured_outputs)
    for index, op in returned_ops.items():
      flat_fn_outputs[index] = op
    fn_outputs = nest.pack_sequence_as(self.graph.structured_outputs,
                                       flat_fn_outputs)
    name = name or fn.__name__
    wrapped_function = self._wrapped_function.prune(
        fn_inputs, fn_outputs, name, self.graph.structured_input_signature)
    self._functions[name] = wrapped_function
    return wrapped_function
@tf_export(v1=["wrap_function"])
def wrap_function(fn, signature, name=None):
  """Wraps the TF 1.x function fn into a graph function.
  The python function `fn` will be called once with symbolic arguments specified
  in the `signature`, traced, and turned into a graph function. Any variables
  created by `fn` will be owned by the object returned by `wrap_function`. The
  resulting graph function can be called with tensors which match the
  signature.
  ```python
  def f(x, do_add):
    v = tf.Variable(5.0)
    if do_add:
      op = v.assign_add(x)
    else:
      op = v.assign_sub(x)
    with tf.control_dependencies([op]):
      return v.read_value()
  f_add = tf.compat.v1.wrap_function(f, [tf.TensorSpec((), tf.float32), True])
  assert float(f_add(1.0)) == 6.0
  assert float(f_add(1.0)) == 7.0
  f_sub= tf.compat.v1.wrap_function(f, [tf.TensorSpec((), tf.float32), False])
  assert float(f_sub(1.0)) == 4.0
  assert float(f_sub(1.0)) == 3.0
  ```
  Both `tf.compat.v1.wrap_function` and `tf.function` create a callable
  TensorFlow graph. But while `tf.function` runs all stateful operations
  (e.g. `tf.print`) and sequences operations to provide the same semantics as
  eager execution, `wrap_function` is closer to the behavior of `session.run` in
  TensorFlow 1.x. It will not run any operations unless they are required to
  compute the function's outputs, either through a data dependency or a control
  dependency. Nor will it sequence operations.
  Unlike `tf.function`, `wrap_function` will only trace the Python function
  once. As with placeholders in TF 1.x, shapes and dtypes must be provided to
  `wrap_function`'s `signature` argument.
  Since it is only traced once, variables and state may be created inside the
  function and owned by the function wrapper object.
  Args:
    fn: python function to be wrapped
    signature: the placeholder and python arguments to be passed to the wrapped
      function
    name: Optional. The name of the function.
  Returns:
    the wrapped graph function.
  """
  holder = VariableHolder(fn)
  func_graph_name = "wrapped_function"
  if name is not None:
    func_graph_name = "wrapped_function_" + name
  return WrappedFunction(
      func_graph.func_graph_from_py_func(
          func_graph_name,
          holder,
          args=None,
          kwargs=None,
          signature=signature,
          add_control_dependencies=False,
          collections={}),
      variable_holder=holder,
      signature=signature)
def function_from_graph_def(graph_def, inputs, outputs, captures=None):
  """Creates a ConcreteFunction from a GraphDef.
  Args:
    graph_def: A GraphDef to make a function out of.
    inputs: A Tensor name or nested structure of names in `graph_def` which
      should be inputs to the function.
    outputs: A Tensor name or nested structure of names in `graph_def` which
      should be outputs of the function.
    captures: (Optional) A dictionary mapping node names in `graph_def` that
      should be captured as inputs to tensors containing the value of the
      captured inputs.
  Returns:
    A ConcreteFunction.
  """
  def _imports_graph_def():
    importer.import_graph_def(graph_def, name="")
    graph = ops.get_default_graph()
    if captures is not None:
      for c in captures:
        graph.add_capture(captures[c], graph.get_tensor_by_name(str(c) + ":0"))
  wrapped_import = wrap_function(_imports_graph_def, [])
  import_graph = wrapped_import.graph
  return wrapped_import.prune(
      nest.map_structure(import_graph.as_graph_element, inputs),
      nest.map_structure(import_graph.as_graph_element, outputs))
