
import functools
import traceback
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
__all__ = ["make_template"]
@tf_export(v1=["make_template"])
def make_template(name_,
                  func_,
                  create_scope_now_=False,
                  unique_name_=None,
                  custom_getter_=None,
                  **kwargs):
  """Given an arbitrary function, wrap it so that it does variable sharing.
  @compatibility(TF2)
  `tf.compat.v1.make_template` is a legacy API that is only compatible
  with eager execution enabled and `tf.function` if you combine it with
  `tf.compat.v1.keras.utils.track_tf1_style_variables`. See the model mapping
  migration guide section on `make_template` for more info:
  Even if you use legacy apis for `variable_scope`-based variable reuse,
  we recommend using
  `tf.compat.v1.keras.utils.track_tf1_style_variables` directly and not using
  `tf.compat.v1.make_template`, as it interoperates with eager execution in a
  simpler and more predictable fashion than `make_template`.
  The TF2 API approach would be tracking your variables using
  `tf.Module`s or Keras layers and models rather than relying on
  `make_template`.
  @end_compatibility
  This wraps `func_` in a Template and partially evaluates it. Templates are
  functions that create variables the first time they are called and reuse them
  thereafter. In order for `func_` to be compatible with a `Template` it must
  have the following properties:
  * The function should create all trainable variables and any variables that
     should be reused by calling `tf.compat.v1.get_variable`. If a trainable
     variable is
     created using `tf.Variable`, then a ValueError will be thrown. Variables
     that are intended to be locals can be created by specifying
     `tf.Variable(..., trainable=false)`.
  * The function may use variable scopes and other templates internally to
      create and reuse variables, but it shouldn't use
      `tf.compat.v1.global_variables` to
      capture variables that are defined outside of the scope of the function.
  * Internal scopes and variable names should not depend on any arguments that
      are not supplied to `make_template`. In general you will get a ValueError
      telling you that you are trying to reuse a variable that doesn't exist
      if you make a mistake.
  In the following example, both `z` and `w` will be scaled by the same `y`. It
  is important to note that if we didn't assign `scalar_name` and used a
  different name for z and w that a `ValueError` would be thrown because it
  couldn't reuse the variable.
  ```python
  def my_op(x, scalar_name):
    var1 = tf.compat.v1.get_variable(scalar_name,
                           shape=[],
                           initializer=tf.compat.v1.constant_initializer(1))
    return x * var1
  scale_by_y = tf.compat.v1.make_template('scale_by_y', my_op, scalar_name='y')
  z = scale_by_y(input1)
  w = scale_by_y(input2)
  ```
  As a safe-guard, the returned function will raise a `ValueError` after the
  first call if trainable variables are created by calling `tf.Variable`.
  If all of these are true, then 2 properties are enforced by the template:
  1. Calling the same template multiple times will share all non-local
      variables.
  2. Two different templates are guaranteed to be unique, unless you reenter the
      same variable scope as the initial definition of a template and redefine
      it. An examples of this exception:
  ```python
  def my_op(x, scalar_name):
    var1 = tf.compat.v1.get_variable(scalar_name,
                           shape=[],
                           initializer=tf.compat.v1.constant_initializer(1))
    return x * var1
  with tf.compat.v1.variable_scope('scope') as vs:
    scale_by_y = tf.compat.v1.make_template('scale_by_y', my_op,
    scalar_name='y')
    z = scale_by_y(input1)
    w = scale_by_y(input2)
  with tf.compat.v1.variable_scope(vs, reuse=True):
    scale_by_y2 = tf.compat.v1.make_template('scale_by_y', my_op,
    scalar_name='y')
    z2 = scale_by_y2(input1)
    w2 = scale_by_y2(input2)
  ```
  Depending on the value of `create_scope_now_`, the full variable scope may be
  captured either at the time of first call or at the time of construction. If
  this option is set to True, then all Tensors created by repeated calls to the
  template will have an extra trailing _N+1 to their name, as the first time the
  scope is entered in the Template constructor no Tensors are created.
  Note: `name_`, `func_` and `create_scope_now_` have a trailing underscore to
  reduce the likelihood of collisions with kwargs.
  Args:
    name_: A name for the scope created by this template. If necessary, the name
      will be made unique by appending `_N` to the name.
    func_: The function to wrap.
    create_scope_now_: Boolean controlling whether the scope should be created
      when the template is constructed or when the template is called. Default
      is False, meaning the scope is created when the template is called.
    unique_name_: When used, it overrides name_ and is not made unique. If a
      template of the same scope/unique_name already exists and reuse is false,
      an error is raised. Defaults to None.
    custom_getter_: Optional custom getter for variables used in `func_`. See
      the `tf.compat.v1.get_variable` `custom_getter` documentation for more
      information.
    **kwargs: Keyword arguments to apply to `func_`.
  Returns:
    A function to encapsulate a set of variables which should be created once
    and reused. An enclosing scope will be created either when `make_template`
    is called or when the result is called, depending on the value of
    `create_scope_now_`. Regardless of the value, the first time the template
    is called it will enter the scope with no reuse, and call `func_` to create
    variables, which are guaranteed to be unique. All subsequent calls will
    re-enter the scope and reuse those variables.
  Raises:
    ValueError: if `name_` is None.
  """
  return make_template_internal(
      name_,
      func_,
      create_scope_now_,
      unique_name_,
      custom_getter_,
      create_graph_function_=False,
      **kwargs)
def make_template_internal(name_,
                           func_,
                           create_scope_now_=False,
                           unique_name_=None,
                           custom_getter_=None,
                           create_graph_function_=False,
                           **kwargs):
  if kwargs:
    func_ = tf_decorator.make_decorator(func_,
                                        functools.partial(func_, **kwargs))
  if context.executing_eagerly():
    if unique_name_ is not None:
      raise ValueError(
          "unique_name_ cannot be used when eager execution is enabled.")
    return EagerTemplate(
        name_,
        func_,
        create_scope_now=create_scope_now_,
        custom_getter=custom_getter_,
        create_graph_function=create_graph_function_)
  return Template(
      name_,
      func_,
      create_scope_now=create_scope_now_,
      unique_name=unique_name_,
      custom_getter=custom_getter_,
      create_graph_function=create_graph_function_)
def _skip_common_stack_elements(stacktrace, base_case):
  for i, (trace, base) in enumerate(zip(stacktrace, base_case)):
    if trace != base:
      return stacktrace[i:]
  return stacktrace[-1:]
class Template(trackable.Trackable):
  def __init__(self,
               name,
               func,
               create_scope_now=False,
               unique_name=None,
               custom_getter=None,
               create_graph_function=False):
    """Creates a template for the given function.
    Args:
      name: A name for the scope created by this template. The name will be made
        unique by appending `_N` to the it (see how
        `tf.compat.v1.variable_scope` treats the `default_name` for details).
      func: The function to apply each time.
      create_scope_now: Whether to create the scope at Template construction
        time, rather than first call. Defaults to false. Creating the scope at
        construction time may be more convenient if the template is to passed
        through much lower level code, and you want to be sure of the scope name
        without knowing exactly where it will be first called. If set to True,
        the scope will be created in the constructor, and all subsequent times
        in `__call__`, leading to a trailing numeral being added to the names of
        all created Tensors. If set to False, the scope will be created at the
        first call location.
      unique_name: When used, it overrides `name` and is not made unique. If a
        template of the same scope/unique_name already exists and reuse is
        false, an error is raised. Defaults to None.
      custom_getter: optional custom getter to pass to `variable_scope()`
      create_graph_function: When True, `func` will be executed as a graph
        function. Enabling this flag gives the caller access to graph-function
        semantics, i.e., accesses to variables are totally ordered and
        side-effecting ops are not pruned.
    Raises:
      ValueError: if `name` is None.
    """
    if create_graph_function:
      self._func = function.defun(func)
    else:
      self._func = func
    self._stacktrace = traceback.format_stack()[:-2]
    self._name = name
    self._unique_name = unique_name
    self._custom_getter = custom_getter
    if name is None:
      raise ValueError("name cannot be None.")
    if create_scope_now:
          (self._unique_name or
          custom_getter=self._custom_getter) as vs:
        self._variable_scope = vs
    else:
      self._variable_scope = None
    self._variables_created = False
    self._first_call = True
  def _call_func(self, args, kwargs):
    try:
      if self._variables_created:
        vars_at_start = len(
            ops.get_collection_ref(ops.GraphKeys.GLOBAL_VARIABLES))
        trainable_at_start = len(
            ops.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES))
        result = self._func(*args, **kwargs)
        trainable_variables = ops.get_collection_ref(
            ops.GraphKeys.TRAINABLE_VARIABLES)
        if trainable_at_start != len(trainable_variables):
          raise ValueError("Trainable variable created when calling a template "
                           "after the first time, perhaps you used tf.Variable "
                           "when you meant tf.get_variable: %s" %
                           (trainable_variables[trainable_at_start:],))
        variables = ops.get_collection_ref(ops.GraphKeys.GLOBAL_VARIABLES)
        if vars_at_start != len(variables):
          logging.info(
              "New variables created when calling a template after "
              "the first time, perhaps you used tf.Variable when you "
              "meant tf.get_variable: %s", variables[vars_at_start:])
      elif self._first_call:
        self._first_call = False
        try:
          with trackable_util.capture_dependencies(template=self):
            result = self._func(*args, **kwargs)
        except:
          self._first_call = True
          raise
        self._variables_created = True
        result = self._func(*args, **kwargs)
      return result
    except Exception as exc:
      args = exc.args
      if not args:
        arg0 = ""
      else:
        arg0 = args[0]
      trace = "".join(
          _skip_common_stack_elements(self._stacktrace,
                                      traceback.format_stack()))
      arg0 = "%s\n\noriginally defined at:\n%s" % (arg0, trace)
      new_args = [arg0]
      new_args.extend(args[1:])
      exc.args = tuple(new_args)
      raise
  def __call__(self, *args, **kwargs):
    if self._variable_scope:
      with variable_scope.variable_scope(
          self._variable_scope, reuse=not self._first_call):
        return self._call_func(args, kwargs)
    else:
      with variable_scope.variable_scope(
          self._unique_name, self._name,
          custom_getter=self._custom_getter) as vs:
        self._variable_scope = vs
        return self._call_func(args, kwargs)
  @property
  def name(self):
    return self._name
  @property
  def func(self):
    return self._func
  @property
  def variable_scope(self):
    return self._variable_scope
  @property
  def variable_scope_name(self):
    if self._variable_scope:
      name = self._variable_scope.name
      if not name or name[-1] == "/":
        return name
      else:
        return name + "/"
  @property
  def variables(self):
    return self.global_variables + self.local_variables
  @property
  def trainable_variables(self):
    if self._variables_created:
      return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,
                                self.variable_scope_name)
    else:
      return []
  @property
  def non_trainable_variables(self):
    global_variables = self.global_variables
    trainable_variables = set(self.trainable_variables)
    return [x for x in global_variables if x not in trainable_variables]
  @property
  def global_variables(self):
    if self._variables_created:
      return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                self.variable_scope_name)
    else:
      return []
  @property
  def local_variables(self):
    if self._variables_created:
      return ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES,
                                self.variable_scope_name)
    else:
      return []
  @property
  def weights(self):
    return self.variables
  @property
  def trainable_weights(self):
    return self.trainable_variables
  @property
  def non_trainable_weights(self):
    return self.non_trainable_variables
  @property
  @deprecated("2017-02-21",
              "The .var_scope property is deprecated. Please change your "
              "code to use the .variable_scope property")
  def var_scope(self):
    return self._variable_scope
class _EagerTemplateVariableStore:
  def __init__(self, variable_scope_name):
    self._variable_scope_name = variable_scope_name
      self._eager_variable_store = variable_scope.EagerVariableStore(default)
    else:
      self._eager_variable_store = variable_scope.EagerVariableStore()
    self._used_once = False
  def set_variable_scope_name(self, variable_scope_name):
    self._variable_scope_name = variable_scope_name
  @tf_contextlib.contextmanager
  def as_default(self):
    try:
      if not self._used_once:
        self._used_once = True
        yield
    finally:
      if self._variable_scope_name is None:
        raise RuntimeError("A variable scope must be set before an "
                           "_EagerTemplateVariableStore object exits.")
      variable_scope.get_variable_scope_store().close_variable_subscopes(
          self._variable_scope_name)
  def _variables_in_scope(self, variable_list):
    if self._variable_scope_name is None:
      raise RuntimeError(
          "A variable scope must be set before variables can be accessed.")
    return [
        v for v in variable_list
        if v.name.startswith(self._variable_scope_name + "/")
    ]
  def variables(self):
    return self._variables_in_scope(self._eager_variable_store.variables())
  def trainable_variables(self):
    return self._variables_in_scope(
        self._eager_variable_store.trainable_variables())
  def non_trainable_variables(self):
    return self._variables_in_scope(
        self._eager_variable_store.non_trainable_variables())
class EagerTemplate(Template):
  def __init__(self,
               name,
               func,
               create_scope_now=False,
               custom_getter=None,
               create_graph_function=False):
    """Creates a template for the given function.
    Args:
      name: A name for the scope created by this template. The name will be made
        unique by appending `_N` to the it (see how
        `tf.compat.v1.variable_scope` treats the `default_name` for details).
      func: The function to apply each time.
      create_scope_now: Whether to create the scope at Template construction
        time, rather than first call. Defaults to false. Creating the scope at
        construction time may be more convenient if the template is passed
        through much lower level code, and you want to be sure of the scope name
        without knowing exactly where it will be first called. If set to True,
        the scope will be created in the constructor, and all subsequent times
        in `__call__`, leading to a trailing numeral being added to the names of
        all created Tensors. If set to False, the scope will be created at the
        first call location.
      custom_getter: optional custom getter to pass to `variable_scope()`
      create_graph_function: When True, `func` will be executed as a graph
        function. Enabling this flag allows the caller to reap the performance
        benefits associated with executing graphs, at the cost of sacrificing
        debuggability; however, not all Python functions can be compiled into
        graph functions. See the documentation for `function.defun` for details.
    Raises:
      RuntimeError: if eager execution is not enabled.
    """
    if not context.executing_eagerly():
      raise RuntimeError(
          "{} objects can only be used when eager execution is enabled, use "
          "tf.Template for graph construction".format(type(self)))
    super(EagerTemplate, self).__init__(name, func, create_scope_now, None,
                                        custom_getter, create_graph_function)
    if self._variable_scope is not None:
      variable_scope_name = self._variable_scope.name
    else:
      variable_scope_name = None
    self._template_store = _EagerTemplateVariableStore(variable_scope_name)
    self._variable_scope_context_manager = None
  def _call_func(self, args, kwargs):
    try:
      vars_at_start = self._template_store.variables()
      trainable_at_start = self._template_store.trainable_variables()
      if self._variables_created:
        result = self._func(*args, **kwargs)
      else:
        with trackable_util.capture_dependencies(template=self):
          result = self._func(*args, **kwargs)
      if self._variables_created:
        trainable_variables = self._template_store.trainable_variables()
        if len(trainable_at_start) != len(trainable_variables):
          raise ValueError(
              "Trainable variable created when calling a template "
              "after the first time, perhaps you used tf.Variable "
              "when you meant tf.get_variable: %s" % list(
                  object_identity.ObjectIdentitySet(trainable_variables) -
                  object_identity.ObjectIdentitySet(trainable_at_start)))
        variables = self._template_store.variables()
        if len(vars_at_start) != len(variables):
          logging.info(
              "New variables created when calling a template after "
              "the first time, perhaps you used tf.Variable when you "
              "meant tf.get_variable: %s",
              list(
                  object_identity.ObjectIdentitySet(variables) -
                  object_identity.ObjectIdentitySet(vars_at_start)))
      else:
        self._variables_created = True
      return result
    except Exception as exc:
      args = exc.args
      if not args:
        arg0 = ""
      else:
        arg0 = args[0]
      trace = "".join(
          _skip_common_stack_elements(self._stacktrace,
                                      traceback.format_stack()))
      arg0 = "%s\n\noriginally defined at:\n%s" % (arg0, trace)
      new_args = [arg0]
      new_args.extend(args[1:])
      exc.args = tuple(new_args)
      raise
  def __call__(self, *args, **kwargs):
    if self._variable_scope:
      if not self._variable_scope_context_manager:
        self._variable_scope_context_manager = variable_scope.variable_scope(
            self._variable_scope, reuse=variable_scope.AUTO_REUSE)
      with self._variable_scope_context_manager:
        with self._template_store.as_default():
          return self._call_func(args, kwargs)
    else:
      with variable_scope.variable_scope(
          self._unique_name, self._name,
          custom_getter=self._custom_getter) as vs:
        self._variable_scope = vs
        self._template_store.set_variable_scope_name(vs.name)
        with self._template_store.as_default():
          return self._call_func(args, kwargs)
  @property
  def variables(self):
    if not self._variables_created:
      return []
    return self._template_store.variables()
  @property
  def trainable_variables(self):
    if not self._variables_created:
      return []
    return self._template_store.trainable_variables()
  @property
  def non_trainable_variables(self):
    if not self._variables_created:
      return []
    return self._template_store.non_trainable_variables()
  @property
  def global_variables(self):
    if not self._variables_created:
      return []
    return self.variables
  @property
  def local_variables(self):
    return []
