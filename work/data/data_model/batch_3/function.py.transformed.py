
import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
class Defun(object):
  """Decorator used to define TensorFlow functions.
  Use this decorator to make a Python function usable directly as a TensorFlow
  function.
  The decorated function must add ops to the default graph and return zero or
  more `Tensor` objects.  Call the decorator with named arguments, one for each
  argument of the function to decorate, with the expected type of the argument
  as value.
  For example if the function to decorate accepts two `tf.float32` arguments
  named `x` and `y`, call the decorator with:
      @Defun(tf.float32, tf.float32)
      def foo(x, y):
        ...
  When you call the decorated function, it adds the `call` ops to the
  default graph. In addition, it adds the definition of the function into the
  default graph. Because the addition of the function into the graph
  is deferred, the decorator can be used anywhere in the program.
  Any variables created inside of the function are hoisted into the outer graph.
  Note that the variables are created in the variable scope that was active
  during the first call to the function. Subsequent function calls will refer to
  the same set of variables.
  Definitions of functions in a graph are frozen as soon as the graph is used to
  create a session. However, new functions and new calls to existing functions
  may be added to the graph, with the new functions themselves becoming
  immediately frozen.
  Example, but also see the [How To on functions](link_needed).
  ```python
  @tf.Defun(tf.float32, tf.float32)
  def MyFunc(x, y):
    return x + y, x - y
  a = tf.constant([1.0])
  b = tf.constant([2.0])
  c, d = MyFunc(a, b, name='mycall')
  ```
  """
  def __init__(self, *input_types, **kwargs):
    """Create a `Defun` decorator.
    Args:
      *input_types: A list of `tf.DType`
      **kwargs: Optional keyword arguments, including
         func_name - (optional).  A python string, the name to use to
           declare this `Function` in the graph.
         grad_func - (optional).  A function implementing the gradient
           of the function-to-register.  This is must be a
           `_DefinedFunction` object. The gradient
           function must satisfy the criterion defined in
           function.proto:GradientDef.
         python_grad_func - (optional).  A function implementing the
           gradient of the function python-side. This function must
           take the current op and the gradients w.r.t. its outputs,
           and return the gradients w.r.t. the inputs. That is it must
           implement the interface expected by `tf.RegisterGradient`).
           This will be called by tf.gradients to add the gradient ops
           to the graph. At most one of grad_func and python_grad_func
           can be specified.
         out_names = (optional). A list of strings, one per output
           tensor.
         shape_func - (optional). A function taking the op and returning a list
           of static shapes to set for the function's outputs.
    """
    self._input_types = input_types
    self._func_name = kwargs.pop("func_name", None)
    self._grad_func = kwargs.pop("grad_func", None)
    self._python_grad_func = kwargs.pop("python_grad_func", None)
    self._out_names = kwargs.pop("out_names", None)
    self._extra_kwargs = kwargs
  def __call__(self, func):
    if not callable(func):
      raise ValueError(f"Function {func} must be a callable.")
    argspec = tf_inspect.getargspec(func)
    if argspec.keywords or argspec.defaults:
      raise ValueError(
          "Functions with argument defaults or keywords arguments are not "
          f"supported. {func} has defaults {argspec.defaults} and keywords "
          f"{argspec.keywords}.")
    min_args = len(argspec.args)
    max_args = min_args
    if argspec.varargs:
      max_args = 1000000
    argnames = argspec.args
    if tf_inspect.ismethod(func):
      min_args -= 1
      argnames = argnames[1:]
    if self._input_types:
      num = len(self._input_types)
      if num < min_args or num > max_args:
        raise ValueError(
            "The number of tf.function input types is not compatible with the "
            f"allowed arguments of {func}. The tf.function have {num} input "
            f"types, while the python function allows minimum {min_args} and "
            f"maximum {max_args} arguments.")
      return _DefinedFunction(
          func,
          argnames,
          self._input_types,
          self._func_name,
          self._grad_func,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)
    if min_args == 0 and max_args == 0:
      return _DefinedFunction(
          func, [], [],
          self._func_name,
          self._grad_func,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)
    return _OverloadedFunction(
        func,
        argnames,
        self._func_name,
        self._grad_func,
        self._python_grad_func,
        out_names=self._out_names,
        **self._extra_kwargs)
class _DefinedFunctionDeleter(object):
  __slots__ = ["name"]
  def __init__(self, name):
    self.name = name
  def __del__(self):
    try:
      context.remove_function(self.name)
    except TypeError:
    except AttributeError:
class _DefinedFunction(object):
  def __init__(self,
               func,
               argnames,
               input_types,
               func_name=None,
               grad_func=None,
               python_grad_func=None,
               out_names=None,
               shape_func=None,
               capture_by_value=False,
               allowlisted_stateful_ops=None,
               capture_resource_var_by_value=True,
               **kwargs):
    """Creates _DefinedFunction.
    Args:
      func:  A python callable which constructs a tf function body.
      argnames: A list of strings for function argument names.
      input_types: The function's argument types. Can be a tuple, list of
        tf data types.
      func_name: The function name. Defaults to None, in which derives from
        'func'.
      grad_func: This function's gradient function, if not None. Defaults
        to None.
      python_grad_func: A python callable implementing the gradient of
        the function python-side.
      out_names: An optional list of strings for the function return value
        names.
      shape_func: An optional function mapping an op to a list of static
        output shapes.
      capture_by_value: Boolean (defaults to False). If True, captured values
        will be copied into the function body.
      allowlisted_stateful_ops: A set of ops that if stateful we ignore and
        copy into the function body, when `capture_by_value` is True.
      capture_resource_var_by_value: Boolean (defaults to True). If False,
        captured resource variable returns the handle instead of value.
      **kwargs: The keyword arguments. **kwargs is passed to every call
        site of this function.
    Raises:
      ValueError: The function definition is invalid.
    """
    self._func = func
    self._input_types = input_types
    self._func_name = func_name
    self._grad_func = grad_func
    self._python_grad_func = python_grad_func
    self._out_names = out_names
    self._shape_func = shape_func
    self._capture_by_value = capture_by_value
    self._allowlisted_stateful_ops = allowlisted_stateful_ops
    if self._allowlisted_stateful_ops is None:
      self._allowlisted_stateful_ops = set()
    self._capture_resource_var_by_value = capture_resource_var_by_value
    self._extra_kwargs = kwargs
    self._definition = None
    self._c_func = None
    self._function_deleter = None
    device_funcs = ops.get_default_graph()._device_functions_outer_to_inner
    self._caller_device = device_funcs[-1] if device_funcs else None
    self._op_def = None
    assert isinstance(input_types, (list, tuple))
    self._arg_types = input_types
    self._arg_names = [argnames[i] if i < len(argnames) else ("arg%d" % i)
                       for i in range(len(input_types))]
  @property
  def name(self):
    self._create_definition_if_needed()
    return self._func_name
  @property
  def definition(self):
    self._create_definition_if_needed()
    if self._c_func:
      with c_api_util.tf_buffer() as buf:
        with self._c_func.get() as func:
          c_api.TF_FunctionToFunctionDef(func, buf)
          fdef = function_pb2.FunctionDef()
          proto_data = c_api.TF_GetBuffer(buf)
          fdef.ParseFromString(compat.as_bytes(proto_data))
          with ops.init_scope():
            if context.executing_eagerly():
              context.add_function(func)
              self._function_deleter = _DefinedFunctionDeleter(
                  fdef.signature.name)
      return fdef
    return self._definition
  @property
  def _signature(self):
    self._create_definition_if_needed()
    return self._op_def
  def set_grad_func(self, grad_func):
    assert not self._grad_func
    assert isinstance(grad_func, _DefinedFunction)
    self._grad_func = grad_func
  @property
  def grad_func_name(self):
    return self._grad_func.name if self._grad_func else None
  @property
  def python_grad_func(self):
    return self._python_grad_func
  @property
  def declared_input_types(self):
    return self._input_types
  @property
  def captured_inputs(self):
    self._create_definition_if_needed()
    return self._extra_inputs
  @property
  def stateful_ops(self):
    """Returns the list of stateful ops in function definition.
    Returns:
      A list of (op.name, op.type) pairs.
    """
    self._create_definition_if_needed()
    return self._stateful_ops
  def _create_definition_if_needed(self):
    with context.graph_mode():
      self._create_definition_if_needed_impl()
  def _create_definition_if_needed_impl(self):
    if self._definition is not None or self._c_func is not None:
      return
    variable_keys = []
    parent_graph = ops.get_default_graph()
    collections_ref = {
        key: parent_graph.get_collection_ref(key) for key in variable_keys}
    temp_graph = func_graph_from_py_func(
        self._func,
        self._arg_names,
        self._arg_types,
        self._func_name,
        self._capture_by_value,
        self._caller_device,
        collections_ref=collections_ref,
        allowlisted_stateful_ops=self._allowlisted_stateful_ops,
        capture_resource_var_by_value=self._capture_resource_var_by_value)
    self._extra_inputs = temp_graph.extra_inputs
    self._sub_functions = temp_graph._functions
    if self._func_name:
      base_func_name = self._func_name
    else:
      base_func_name = function_utils.get_func_name(self._func)
      if self._grad_func:
        base_func_name += ("_%s" % self._grad_func.name)
    kwargs_attr = _parse_kwargs_as_attrs(base_func_name, **self._extra_kwargs)
      self._definition = graph_to_function_def.graph_to_function_def(
          temp_graph,
          temp_graph.get_operations(),
          temp_graph.inputs,
          temp_graph.outputs,
          out_names=self._out_names)
      for k in kwargs_attr:
        self._definition.attr[k].CopyFrom(kwargs_attr[k])
      self._hash_str = self._create_hash_str(
          self._definition.signature.input_arg,
          self._definition.signature.output_arg, self._definition.node_def)
      if not self._func_name:
        self._func_name = "_".join([base_func_name, self._hash_str])
      self._definition.signature.name = self._func_name
      if self._func.__doc__:
        self._definition.signature.description = self._func.__doc__
      self._op_def = self._definition.signature
      output_names = ([compat.as_bytes(x) for x in self._out_names]
                      if self._out_names else [])
      description = self._func.__doc__ or None
      c_func = c_api.TF_GraphToFunction_wrapper(
          temp_graph._c_graph,
          base_func_name,
          [t._as_tf_output() for t in temp_graph.inputs],
          [t._as_tf_output() for t in temp_graph.outputs],
          output_names,
          description)
      self._c_func = c_api_util.ScopedTFFunction(c_func, base_func_name)
      self._set_c_attrs(kwargs_attr)
      self._op_def = self.definition.signature
      if self._func_name:
        assert self._func_name == self._op_def.name
      else:
        self._func_name = compat.as_str(self._op_def.name)
    self._stateful_ops = [(op.name, op.type)
                          for op in temp_graph.get_operations()
  def _set_c_attrs(self, attrs):
    for name, attr_value in attrs.items():
      serialized = attr_value.SerializeToString()
      with self._c_func.get() as func:
        c_api.TF_FunctionSetAttrValueProto(func, compat.as_str(name),
                                           serialized)
  def _create_hash_str(self, input_arg, output_arg, node_def):
    """Creates an 8-character string unique to this input.
    Args:
      input_arg: the input_arg field of an OpDef
                 (e.g. self._definition.signature.input_arg)
      output_arg: the output_arg field of an OpDef
                 (e.g. self._definition.signature.output_arg)
      node_def: the node_def field of a FunctionDef
                (e.g. self._definition.node_def)
    Returns:
      The unique string for this input
    """
    hasher = hashlib.sha1()
    def update_num(n):
      hasher.update(compat.as_bytes("%x" % n))
    def update_str(s):
      update_num(len(s))
      hasher.update(compat.as_bytes(s))
    def update_strs(slist):
      update_num(len(slist))
      for s in slist:
        update_str(s)
    for adef in input_arg:
      update_str(adef.SerializeToString())
    for adef in output_arg:
      update_str(adef.SerializeToString())
    for n in sorted(node_def, key=lambda n: n.name):
      update_str(n.name)
      update_str(n.op)
      update_strs(n.input)
      update_num(len(n.attr))
      for k in sorted(n.attr):
        update_str(k)
        update_str(n.attr[k].SerializeToString())
    return hasher.hexdigest()[:8]
  def add_to_graph(self, g):
    self._create_definition_if_needed()
    if context.executing_eagerly():
      context.context().add_function_def(self.definition)
    else:
      g._add_function(self)
    for f in self._sub_functions.values():
      f.add_to_graph(g)
    if self._grad_func:
      self._grad_func.add_to_graph(g)
  def __call__(self, *args, **kwargs):
    self.add_to_graph(ops.get_default_graph())
    args = [ops.convert_to_tensor(_) for _ in args] + self._extra_inputs
    ret, op = _call(self._signature, *args, **kwargs)
    assert isinstance(op, ops.Operation)
    setattr(op, "__defun", self)
    if self._shape_func is not None:
      shapes = self._shape_func(op)
      if len(shapes) != len(op.outputs):
        raise ValueError(f"shape_func {self._shape_func} produced "
                         f"{len(shapes):d} shapes, which does not match "
                         f"{len(op.outputs)} outputs.")
      for (t, shape) in zip(op.outputs, shapes):
        t.set_shape(shape)
    return ret
class _OverloadedFunction(object):
  def __init__(self,
               func,
               argnames,
               func_name=None,
               grad_func=None,
               python_grad_func=None,
               out_names=None,
               **kwargs):
    self._func = func
    self._argnames = argnames
    self._func_name = func_name
    assert grad_func is None or isinstance(grad_func, _OverloadedFunction)
    self._grad_func = grad_func
    self._python_grad_func = python_grad_func
    self._out_names = out_names
    self._extra_kwargs = kwargs
    self._overload = {}
  def instantiate(self, input_types):
    key = _type_list_to_str(input_types)
    defined = self._overload.get(key)
    if not defined:
      name = self._func_name
      if name is not None:
        name = "_".join([name, key])
      defined = _DefinedFunction(
          self._func,
          self._argnames,
          input_types,
          name,
          None,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)
      if self._grad_func:
        output_types = [
        ]
        defined._grad_func = self._grad_func.instantiate(input_types +
                                                         output_types)
      self._overload[key] = defined
    return defined
  def __call__(self, *args, **kwargs):
    input_types = []
    args = list(args)
    for (i, x) in enumerate(args):
      x = ops.convert_to_tensor(x)
      if not isinstance(x, ops.Tensor):
        raise ValueError(f"Expected a Tensor but got {x} with type {type(x)}.")
      input_types.append(x.dtype)
      args[i] = x
    return self.instantiate(input_types)(*args, **kwargs)
class _FuncGraph(ops.Graph):
  """A helper for constructing a function.
  _FuncGraph overrides ops.Graph's create_op() so that we can keep
  track of all inputs into every op created inside the function.  If
  any input is from other graphs, we keep track of it in self.capture
  and substitute the input with a place holder.
  Each captured input's corresponding place holder is converted into a
  function argument and the caller passes in the captured tensor.
  """
  def __init__(self, name, capture_by_value, allowlisted_stateful_ops,
               capture_resource_var_by_value, *args, **kwargs):
    super(_FuncGraph, self).__init__(*args, **kwargs)
    self._capture_by_value = capture_by_value
    self._allowlisted_stateful_ops = allowlisted_stateful_ops
    self._capture_resource_var_by_value = capture_resource_var_by_value
    self._building_function = True
    self._outer_graph = ops.get_default_graph()
    self._vscope = vs.get_variable_scope()
    self._old_custom_getter = self._vscope.custom_getter
    self.name = name
    self.inputs = []
    self.outputs = []
    self._captured = {}
    self.extra_inputs = []
    self.extra_args = []
    self.extra_vars = []
  @property
  def outer_graph(self):
    return self._outer_graph
  @tf_contextlib.contextmanager
  def container(self, container_name):
    original_container = self._container
    with ops.init_scope():
      original_init_container = ops.get_default_graph()._container
    try:
      self._container = container_name
      with ops.init_scope():
        ops.get_default_graph()._container = container_name
      yield self._container
    finally:
      self._container = original_container
      with ops.init_scope():
        ops.get_default_graph()._container = original_init_container
  def getvar(
      self,
      getter,
      name,
      shape=None,
      dtype=None,
      initializer=None,
      reuse=None,
      trainable=True,
      use_resource=None,
      **kwargs):
    with self._outer_graph.as_default():
      var = self._vscope.get_variable(
          vs._get_default_variable_store(),
          name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          use_resource=use_resource)
      self.extra_vars.append(var)
      if (isinstance(var, resource_variable_ops.BaseResourceVariable) and
          self._capture_resource_var_by_value):
        return var.value()
      return var
  def _create_op_internal(
      self,
      op_type,
      inputs,
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    for i, x in enumerate(inputs):
      if isinstance(x, ops.EagerTensor) or x.graph is not self:
        inputs[i] = self.capture(x)
    return super(_FuncGraph, self)._create_op_internal(
        op_type,
        inputs,
        dtypes=dtypes,
        input_types=input_types,
        name=name,
        attrs=attrs,
        op_def=op_def,
        compute_device=compute_device)
  def capture(self, tensor, name=None):
    if tensor.ref() in self._captured:
      return self._captured[tensor.ref()]
    elif self._capture_by_value:
      return self._add_tensor_and_parents(tensor)
    else:
      return self._capture_tensor_as_extra_input(tensor, name)
  @property
  def captures(self):
    return [(k.deref(), v) for k, v in self._captured.items()]
  def _capture_tensor_as_extra_input(self, tensor, name=None):
    self.extra_inputs.append(tensor)
    with ops.control_dependencies(None):
      ph = array_ops.placeholder(
          tensor.dtype, shape=tensor.get_shape(), name=name)
    if isinstance(tensor, ops.EagerTensor):
      handle_data = tensor._handle_data
      if handle_data:
        handle_data = handle_data.SerializeToString()
    else:
      handle_data = c_api.GetHandleShapeAndType(tensor.graph._c_graph,
                                                tensor._as_tf_output())
    if handle_data:
      c_api.SetHandleShapeAndType(ph.graph._c_graph, ph._as_tf_output(),
                                  compat.as_bytes(handle_data))
    self.inputs.append(ph)
    self._captured[tensor.ref()] = ph
    self.extra_args.append(ph)
    if _is_guaranteed_const(tensor):
      with ops.control_dependencies(None):
        return array_ops.guarantee_const(ph)
    else:
      return ph
  def _add_tensor_and_parents(self, tensor):
    op = self._add_op_and_parents(tensor.op)
    return op.outputs[tensor.value_index]
  def _add_op_and_parents(self, op):
    op_def = graph_to_function_def._get_op_def(op)
    if op._is_stateful and op not in self._allowlisted_stateful_ops:
      raise ValueError(f"Cannot capture a stateful node (name:{op.name}, "
                       f"type:{op.type}) by value.")
    elif op.type in ("Placeholder", "PlaceholderV2"):
      raise ValueError(f"Cannot capture a placeholder (name:{op.name}, "
                       f"type:{op.type}) by value.")
    captured_inputs = [self._add_tensor_and_parents(x) for x in op.inputs]
    captured_op = self._create_op_internal(
        op.type,
        captured_inputs, [o.dtype for o in op.outputs],
        name=op.name,
        attrs=op.node_def.attr,
        op_def=op_def)
    for t, captured_t in zip(op.outputs, captured_op.outputs):
      self._captured[t.ref()] = captured_t
    return captured_op
def func_graph_from_py_func(func,
                            arg_names,
                            arg_types,
                            name=None,
                            capture_by_value=False,
                            device=None,
                            colocation_stack=None,
                            container=None,
                            collections_ref=None,
                            arg_shapes=None,
                            allowlisted_stateful_ops=None,
                            capture_resource_var_by_value=True):
  """Returns a _FuncGraph generated from `func`.
  Args:
    func: A Python callable which constructs a TF function body. The arguments
      must correspond to `arg_types`. Returns a value or list/tuple of values.
      No returned value can be None.
    arg_names: A sequence of strings for the function argument names.
    arg_types: A sequence of the function's argument types.
    name: The function name. If None, the name is derived from `func`.
    capture_by_value: boolean. If True, captured values will be copied into the
      function body.
    device: device name or function.
    colocation_stack: A colocation stack (list) the _FuncGraph should use.
    container: A container name the _FuncGraph should start with.
    collections_ref: A reference to a collections dict the _FuncGraph should
      use internally.
    arg_shapes: A sequence of the function's argument shapes.
    allowlisted_stateful_ops: A set of ops that if stateful we ignore and
      re-create.
    capture_resource_var_by_value: Boolean (defaults to True). If False,
      captured resource variable returns the handle instead of value.
  Returns:
    A _FuncGraph.
  Raises:
    ValueError: if func returns None.
  """
  if not name:
    name = function_utils.get_func_name(func)
  func_graph = _FuncGraph(name, capture_by_value, allowlisted_stateful_ops,
                          capture_resource_var_by_value)
  with func_graph.as_default(), ops.device(device):
    if collections_ref is not None:
      func_graph._collections = collections_ref
    if container is not None:
      func_graph._container = container
    if colocation_stack is not None:
      func_graph._colocation_stack = colocation_stack
    if arg_shapes is None:
      arg_shapes = [None] * len(arg_types)
    for (argname, argtype, argshape) in zip(arg_names, arg_types, arg_shapes):
      argholder = array_ops.placeholder(argtype, shape=argshape, name=argname)
      func_graph.inputs.append(argholder)
    with vs.variable_scope("", custom_getter=func_graph.getvar):
      outputs = func(*func_graph.inputs)
    if outputs is None:
      outputs = []
    else:
      if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
      if any(_ is None for _ in outputs):
        raise ValueError(f"Function {name} can not return None.")
    outputs = [ops.convert_to_tensor(t) for t in outputs]
    outputs = [func_graph.capture(t) if t.graph is not func_graph else t
               for t in outputs]
    func_graph.outputs = outputs
  return func_graph
def _is_guaranteed_const(tensor):
  if isinstance(tensor, ops.EagerTensor):
    return False
  class Work(object):
    def __init__(self, op, leaving):
      self.op = op
      self.leaving = leaving
  is_guaranteed_const = lambda op: op.node_def.op == "GuaranteeConst"
  constants = set([])
  def all_inputs_const(op):
    return op.inputs and all(inp.op in constants for inp in op.inputs)
  visited = set([])
  stack = [Work(tensor.op, leaving=False)]
  while stack:
    work = stack.pop()
    if work.leaving:
      if all_inputs_const(work.op):
        constants.add(work.op)
      continue
    visited.add(work.op)
    if is_guaranteed_const(work.op):
      constants.add(work.op)
      continue
    stack.append(Work(work.op, leaving=True))
    for inp in work.op.inputs:
      if inp.op not in visited:
        stack.append(Work(inp.op, leaving=False))
  return tensor.op in constants
def _call(sig, *inputs, **kwargs):
  if len(inputs) != len(sig.input_arg):
    raise ValueError(f"Expected {len(sig.input_arg):d} arguments, got "
                     f"{len(inputs):d}.")
  name = kwargs.pop("name", None)
  g = ops.get_default_graph()
  func_name = sig.name
  if name is None:
    name = func_name
  attrs = _parse_kwargs_as_attrs(func_name, **kwargs)
  output_types = [dtypes.DType(x.type) for x in sig.output_arg]
      func_name, list(inputs), output_types, name=name, attrs=attrs, op_def=sig)
  if op.outputs:
    if len(op.outputs) == 1:
      ret = op.outputs[0]
    else:
      ret = tuple(op.outputs)
  else:
    ret = op
  return ret, op
def _from_definition(fdef, grad_func=None):
  func = None
  argnames = [arg.name for arg in fdef.signature.input_arg]
  input_types = tuple(
      dtypes.as_dtype(arg.type) for arg in fdef.signature.input_arg)
  func_name = fdef.signature.name
  python_grad_func = None
  out_names = [arg.name for arg in fdef.signature.output_arg]
  result = _DefinedFunction(func, argnames, input_types, func_name, grad_func,
                            python_grad_func, out_names)
  serialized = fdef.SerializeToString()
  c_func = c_api.TF_FunctionImportFunctionDef(serialized)
  result._c_func = c_api_util.ScopedTFFunction(c_func, func_name)
  result._extra_inputs = []
  result._op_def = fdef.signature
  return result
def from_library(lib):
  if not lib.function and not lib.gradient:
    return []
  funcs = {fdef.signature.name: fdef for fdef in lib.function}
  for g in lib.gradient:
    if g.function_name not in funcs:
      raise ValueError(f"FunctionDefLibrary missing '{g.function_name}' "
                       f"FunctionDef\n{lib}")
    if g.gradient_func not in funcs:
      raise ValueError(f"FunctionDefLibrary missing '{g.gradient_func}' "
                       f"FunctionDef\n{lib}")
  func_to_grad = collections.defaultdict(lambda: None)
  grad_to_funcs = collections.defaultdict(list)
  for gdef in lib.gradient:
    func_to_grad[gdef.function_name] = gdef.gradient_func
    grad_to_funcs[gdef.gradient_func].append(gdef.function_name)
  ready = [
      fdef for fdef in lib.function if func_to_grad[fdef.signature.name] is None
  ]
  if not ready:
    raise ValueError(
        f"FunctionDefLibrary contains cyclic gradient functions!\n{lib}")
  initialized = {}
  while ready:
    fdef = ready.pop()
    name = fdef.signature.name
    grad = initialized.get(func_to_grad[name])
    if func_to_grad[name]:
      assert grad
    defined_func = _from_definition(fdef, grad_func=grad)
    initialized[name] = defined_func
    ready.extend(funcs[f] for f in grad_to_funcs[name])
  return initialized.values()
def _get_experimental_kwarg_as_attr(attr_name, value):
  if isinstance(value, bool):
    return attr_value_pb2.AttrValue(b=value)
  elif isinstance(value, int):
    return attr_value_pb2.AttrValue(i=value)
  elif isinstance(value, float):
    return attr_value_pb2.AttrValue(f=value)
  elif isinstance(value, str):
    return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
  else:
    raise ValueError(f"Attribute {attr_name} must be bool, int, float, or "
                     f"str. Got {type(value)}.")
def _get_kwarg_as_str_attr(attr_name, value):
  if isinstance(value, str):
    return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
  else:
    raise ValueError(f"Attribute {attr_name} must be str. Got {type(value)}.")
def _parse_kwargs_as_attrs(func_name, **kwargs):
  attrs = {}
  noinline = kwargs.pop("noinline", None)
  if noinline is not None:
    attrs["_noinline"] = attr_value_pb2.AttrValue(b=bool(noinline))
  attrs["_disable_call_shape_inference"] = attr_value_pb2.AttrValue(b=True)
  compiled = kwargs.pop("compiled", None)
  separate_compiled_gradients = kwargs.pop("separate_compiled_gradients", None)
  if compiled is not None:
    attrs["_XlaCompile"] = attr_value_pb2.AttrValue(b=bool(compiled))
    attrs["_XlaSeparateCompiledGradients"] = attr_value_pb2.AttrValue(
        b=bool(separate_compiled_gradients))
    if "_XlaScope" in ops.get_default_graph()._attr_scope_map:
      attrs["_XlaScope"] = ops.get_default_graph()._attr_scope_map["_XlaScope"]
    else:
      attrs["_XlaScope"] = attr_value_pb2.AttrValue(
          s=("function_%s" % func_name).encode())
  kwargs_keys = list(kwargs.keys())
  for key in kwargs_keys:
    if key.startswith("experimental_"):
      attrs[key] = _get_experimental_kwarg_as_attr(key, kwargs[key])
      del kwargs[key]
    elif key == "_implements" or key == "_reference":
      attrs[key] = _get_kwarg_as_str_attr(key, kwargs[key])
      del kwargs[key]
  if kwargs:
    raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}.")
  return attrs
def get_extra_vars():
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_vars
  else:
    return []
def get_extra_inputs():
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_inputs
  else:
    return []
def get_extra_args():
  """Returns the corresponding function arguments for the captured inputs.
  Returns:
    If the default graph is being used to define a function, the
    returned list of place holders are those used inside the function
    body corresponding those returned by get_extra_inputs(). Otherwise,
    returns an empty list.
  """
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_args
  else:
    return []
def _type_list_to_str(types):
  if any(_ not in _DTYPE_TO_STR for _ in types):
    unsupported_types = [type_ for type_ in types if type_ not in _DTYPE_TO_STR]
    raise ValueError(f"Unsupported dtypes {unsupported_types} in "
                     "`types`. Supported dtypes are "
                     f"{_DTYPE_TO_STR.keys()}.")
  return "".join(_DTYPE_TO_STR[_] for _ in types)
_DTYPE_TO_STR = {
    dtypes.float16: "f16",
    dtypes.float32: "f32",
    dtypes.float64: "f64",
    dtypes.int32: "i32",
    dtypes.uint8: "i8",
    dtypes.uint16: "u16",
    dtypes.uint32: "u32",
    dtypes.uint64: "u64",
    dtypes.int16: "i16",
    dtypes.int8: "i8",
    dtypes.string: "s",
    dtypes.complex64: "c64",
    dtypes.complex128: "c128",
    dtypes.int64: "i64",
    dtypes.bool: "b",
    dtypes.qint8: "qi8",
    dtypes.quint8: "qu8",
    dtypes.qint16: "qi16",
    dtypes.quint16: "qu16",
    dtypes.qint32: "qi32",
    dtypes.bfloat16: "b16"
}
