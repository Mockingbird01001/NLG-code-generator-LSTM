
import abc
import contextlib
import functools
import warnings
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
_DEFAULT_VALID_DTYPES = frozenset([
    dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64,
    dtypes.complex64, dtypes.complex128
])
def _deduplicate_indexed_slices(values, indices):
  """Sums `values` associated with any non-unique `indices`.
  Args:
    values: A `Tensor` with rank >= 1.
    indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
  Returns:
    A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
    de-duplicated version of `indices` and `summed_values` contains the sum of
    `values` slices associated with each unique index.
  """
  unique_indices, new_index_positions = array_ops.unique(indices)
  summed_values = math_ops.unsorted_segment_sum(
      values, new_index_positions,
      array_ops.shape(unique_indices)[0])
  return (summed_values, unique_indices)
class NullContextmanager(object):
  def __init__(self, *args, **kwargs):
    pass
  def __enter__(self):
    pass
  def __exit__(self, type_arg, value_arg, traceback_arg):
def name_scope_only_in_function_or_graph(name):
  if not context.executing_eagerly():
    return ops.name_scope_v1(name)
  else:
    return NullContextmanager()
@keras_export("keras.optimizers.Optimizer", metaclass=abc.ABCMeta)
class OptimizerV2(trackable.Trackable):
  """Base class for Keras optimizers.
  You should not use this class directly, but instead instantiate one of its
  subclasses such as `tf.keras.optimizers.SGD`, `tf.keras.optimizers.Adam`, etc.
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
  opt_op = opt.minimize(loss, var_list=[var1, var2])
  opt_op.run()
  opt.minimize(loss, var_list=[var1, var2])
  ```
  In Keras models, sometimes variables are created when the model is first
  called, instead of construction time. Examples include 1) sequential models
  without input shape pre-defined, or 2) subclassed models. Pass var_list as
  callable in these cases.
  Example:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(num_hidden, activation='relu'))
  model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
  loss_fn = lambda: tf.keras.losses.mse(model(input), output)
  var_list_fn = lambda: model.trainable_weights
  for input, output in data:
    opt.minimize(loss_fn, var_list_fn)
  ```
  Calling `minimize()` takes care of both computing the gradients and
  applying them to the variables.  If you want to process the gradients
  before applying them you can instead use the optimizer in three steps:
  1.  Compute the gradients with `tf.GradientTape`.
  2.  Process the gradients as you wish.
  3.  Apply the processed gradients with `apply_gradients()`.
  Example:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  with tf.GradientTape() as tape:
    loss = <call_loss_function>
  vars = <list_of_variables>
  grads = tape.gradient(loss, vars)
  processed_grads = [process_gradient(g) for g in grads]
  opt.apply_gradients(zip(processed_grads, var_list))
  ```
  This optimizer class is `tf.distribute.Strategy` aware, which means it
  automatically sums gradients across all replicas. To average gradients,
  you divide your loss by the global batch size, which is done
  automatically if you use `tf.keras` built-in training or evaluation loops.
  See the `reduction` argument of your loss which should be set to
  `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` for averaging or
  `tf.keras.losses.Reduction.SUM` for not.
  To aggregate gradients yourself, call `apply_gradients` with
  `experimental_aggregate_gradients` set to False. This is useful if you need to
  process aggregated gradients.
  If you are not using these and you want to average gradients, you should use
  `tf.math.reduce_sum` to add up your per-example losses and then divide by the
  global batch size. Note that when using `tf.distribute.Strategy`, the first
  component of a tensor's shape is the *replica-local* batch size, which is off
  by a factor equal to the number of replicas being used to compute a single
  step. As a result, using `tf.math.reduce_mean` will give the wrong answer,
  resulting in gradients that can be many times too big.
  All Keras optimizers respect variable constraints. If constraint function is
  passed to any variable, the constraint will be applied to the variable after
  the gradient has been applied to the variable.
  Important: If gradient is sparse tensor, variable constraint is not supported.
  The entire optimizer is currently thread compatible, not thread-safe. The user
  needs to perform synchronization if necessary.
  Many optimizer subclasses, such as `Adam` and `Adagrad` allocate and manage
  additional variables associated with the variables to train.  These are called
  <i>Slots</i>.  Slots have names and you can ask the optimizer for the names of
  the slots that it uses.  Once you have a slot name you can ask the optimizer
  for the variable it created to hold the slot value.
  This can be useful if you want to log debug a training algorithm, report stats
  about the slots, etc.
  These are arguments passed to the optimizer subclass constructor
  (the `__init__` method), and then passed to `self._set_hyper()`.
  They can be either regular Python values (like 1.0), tensors, or
  callables. If they are callable, the callable will be called during
  `apply_gradients()` to get the value for the hyper parameter.
  Hyperparameters can be overwritten through user code:
  Example:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  loss = lambda: 3 * var1 + 2 * var2
  opt.minimize(loss, var_list=[var1, var2])
  opt.learning_rate = 0.05
  opt.minimize(loss, var_list=[var1, var2])
  ```
  Optimizer accepts a callable learning rate in two ways. The first way is
  through built-in or customized
  `tf.keras.optimizers.schedules.LearningRateSchedule`. The schedule will be
  called on each iteration with `schedule(iteration)`, a `tf.Variable`
  owned by the optimizer.
  Example:
  >>> var = tf.Variable(np.random.random(size=(1,)))
  >>> learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
  ... initial_learning_rate=.01, decay_steps=20, decay_rate=.1)
  >>> opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  >>> loss = lambda: 3 * var
  >>> opt.minimize(loss, var_list=[var])
  <tf.Variable...
  The second way is through a callable function that
  does not accept any arguments.
  Example:
  >>> var = tf.Variable(np.random.random(size=(1,)))
  >>> def lr_callable():
  ...   return .1
  >>> opt = tf.keras.optimizers.SGD(learning_rate=lr_callable)
  >>> loss = lambda: 3 * var
  >>> opt.minimize(loss, var_list=[var])
  <tf.Variable...
  If you intend to create your own optimization algorithm, simply inherit from
  this class and override the following methods:
    - `_resource_apply_dense` (update variable given gradient tensor is a dense
      `tf.Tensor`)
    - `_resource_apply_sparse` (update variable given gradient tensor is a
      sparse `tf.IndexedSlices`. The most common way for this to happen
      is if you are taking the gradient through a `tf.gather`.)
    - `_create_slots`
      (if your optimizer algorithm requires additional variables)
    - `get_config`
      (serialization of the optimizer, include all hyper parameters)
  """
  _HAS_AGGREGATE_GRAD = False
  def __init__(self,
               name,
               gradient_aggregator=None,
               gradient_transformers=None,
               **kwargs):
    """Create a new Optimizer.
    This must be called by the constructors of subclasses.
    Note that Optimizer instances should not bind to a single graph,
    and so shouldn't keep Tensors as member variables. Generally
    you should be able to use the _set_hyper()/state.get_hyper()
    facility instead.
    This class is stateful and thread-compatible.
    Example of custom gradient transformations:
    ```python
    def my_gradient_transformer(grads_and_vars):
      return [(2. * g, v) for g, v in grads_and_vars]
    optimizer = tf.keras.optimizers.SGD(
        1e-3, gradient_transformers=[my_gradient_transformer])
    ```
    Args:
      name: String. The name to use for momentum accumulator weights created
        by the optimizer.
      gradient_aggregator: The function to use to aggregate gradients across
        devices (when using `tf.distribute.Strategy`). If `None`, defaults to
        summing the gradients across devices. The function should accept and
        return a list of `(gradient, variable)` tuples.
      gradient_transformers: Optional. List of functions to use to transform
        gradients before applying updates to Variables. The functions are
        applied after `gradient_aggregator`. The functions should accept and
        return a list of `(gradient, variable)` tuples.
      **kwargs: keyword arguments. Allowed arguments are `clipvalue`,
        `clipnorm`, `global_clipnorm`.
        If `clipvalue` (float) is set, the gradient of each weight
        is clipped to be no higher than this value.
        If `clipnorm` (float) is set, the gradient of each weight
        is individually clipped so that its norm is no higher than this value.
        If `global_clipnorm` (float) is set the gradient of all weights is
        clipped so that their global norm is no higher than this value.
    Raises:
      ValueError: in case of any invalid argument.
    """
    allowed_kwargs = {"clipnorm", "clipvalue", "lr", "decay", "global_clipnorm"}
    for k in kwargs:
      if k not in allowed_kwargs:
        raise TypeError("Unexpected keyword argument "
                        "passed to optimizer: " + str(k))
      if kwargs[k] is not None and kwargs[k] < 0:
        raise ValueError("Expected {} >= 0, received: {}".format(k, kwargs[k]))
      if k == "lr":
        warnings.warn(
            "The `lr` argument is deprecated, use `learning_rate` instead.")
    self._use_locking = True
    self._init_set_name(name)
    self._hyper = {}
    self._slots = {}
    self._slot_names = []
    self._weights = []
    self._iterations = None
    self._deferred_slot_restorations = {}
    decay = kwargs.pop("decay", 0.0)
    if decay < 0.:
      raise ValueError("decay cannot be less than 0: {}".format(decay))
    self._initial_decay = decay
    self._hypers_created = False
    if distribute_ctx.has_strategy():
      self._distribution_strategy = distribute_ctx.get_strategy()
    else:
      self._distribution_strategy = None
    if gradient_aggregator is None:
      gradient_aggregator = optimizer_utils.all_reduce_sum_gradients
    self.gradient_aggregator = gradient_aggregator
    if gradient_transformers is None:
      gradient_transformers = []
    self.gradient_transformers = gradient_transformers
    self.clipnorm = kwargs.pop("clipnorm", None)
    self.global_clipnorm = kwargs.pop("global_clipnorm", None)
    if self.clipnorm is not None and self.global_clipnorm is not None:
      raise ValueError("Cannot accept both `clipnorm` and `global_clipnorm`, "
                       "passed `clipnorm` {}, `global_clipnorm` {}".format(
                           self.clipnorm, self.global_clipnorm))
    self.clipvalue = kwargs.pop("clipvalue", None)
  @property
  def clipnorm(self):
    return self._clipnorm
  @property
  def global_clipnorm(self):
    return self._global_clipnorm
  @clipnorm.setter
  def clipnorm(self, val):
    if val is not None and self.gradient_transformers:
      raise ValueError("`clipnorm` cannot be set when `gradient_transformers` "
                       "is set. Instead, use the `gradient_transformers` to "
                       "specify clipping and other transformations.")
    self._clipnorm = val
    self._clipnorm_fn = optimizer_utils.make_gradient_clipnorm_fn(
        self._clipnorm)
  @global_clipnorm.setter
  def global_clipnorm(self, val):
    if val is not None and self.gradient_transformers:
      raise ValueError("`clipnorm` cannot be set when `gradient_transformers` "
                       "is set. Instead, use the `gradient_transformers` to "
                       "specify clipping and other transformations.")
    self._global_clipnorm = val
    self._global_clipnorm_fn = optimizer_utils.make_global_gradient_clipnorm_fn(
        self._global_clipnorm)
  @property
  def clipvalue(self):
    return self._clipvalue
  @clipvalue.setter
  def clipvalue(self, val):
    if val is not None and self.gradient_transformers:
      raise ValueError("`clipvalue` cannot be set when `gradient_transformers` "
                       "is set. Instead, use the `gradient_transformers` to "
                       "specify clipping and other transformations.")
    self._clipvalue = val
    self._clipvalue_fn = optimizer_utils.make_gradient_clipvalue_fn(
        self._clipvalue)
  def _transform_loss(self, loss):
    return loss
  def _get_gradients(self, tape, loss, var_list, grad_loss=None):
    grads = tape.gradient(loss, var_list, grad_loss)
    return list(zip(grads, var_list))
  def _transform_unaggregated_gradients(self, grads_and_vars):
    return grads_and_vars
  def _aggregate_gradients(self, grads_and_vars):
    """Called in `apply_gradients` to aggregate gradients across devices.
    Note that user subclasses may override this, so the interface should not be
    changed.
    Args:
      grads_and_vars: List of (gradient, variable) pairs.
    Returns:
      A list of (aggregrated_gradient, variable) pairs. By default, this calls
      `self.gradient_aggregator`.
    """
    return self.gradient_aggregator(grads_and_vars)
  def _transform_gradients(self, grads_and_vars):
    if self._clipvalue is not None:
      grads_and_vars = self._clipvalue_fn(grads_and_vars)
    if self._clipnorm is not None:
      grads_and_vars = self._clipnorm_fn(grads_and_vars)
    if self._global_clipnorm is not None:
      grads_and_vars = self._global_clipnorm_fn(grads_and_vars)
    for fn in self.gradient_transformers:
      grads_and_vars = fn(grads_and_vars)
    return grads_and_vars
  def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
    """Minimize `loss` by updating `var_list`.
    This method simply computes gradient using `tf.GradientTape` and calls
    `apply_gradients()`. If you want to process the gradient before applying
    then call `tf.GradientTape` and `apply_gradients()` explicitly instead
    of using this function.
    Args:
      loss: `Tensor` or callable. If a callable, `loss` should take no arguments
        and return the value to minimize. If a `Tensor`, the `tape` argument
        must be passed.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` since the variables are created at the first time `loss` is
        called.
      grad_loss: (Optional). A `Tensor` holding the gradient computed for
        `loss`.
      name: (Optional) str. Name for the returned operation.
      tape: (Optional) `tf.GradientTape`. If `loss` is provided as a `Tensor`,
        the tape that computed the `loss` must be provided.
    Returns:
      An `Operation` that updates the variables in `var_list`. The `iterations`
      will be automatically increased by 1.
    Raises:
      ValueError: If some of the variables are not `Variable` objects.
    """
    grads_and_vars = self._compute_gradients(
        loss, var_list=var_list, grad_loss=grad_loss, tape=tape)
    return self.apply_gradients(grads_and_vars, name=name)
  def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
    """Compute gradients of `loss` for the variables in `var_list`.
    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.
    Args:
      loss: `Tensor` or callable. If a callable, `loss` should take no
        arguments and return the value to minimize. If a `Tensor`, the `tape`
        argument must be passed.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` and the variables are created at the first time when `loss`
        is called.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      tape: (Optional) `tf.GradientTape`. If `loss` is provided as a `Tensor`,
        the tape that computed the `loss` must be provided.
    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.
    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid, or var_list is None.
    """
    if not callable(loss) and tape is None:
      raise ValueError("`tape` is required when a `Tensor` loss is passed.")
    tape = tape if tape is not None else backprop.GradientTape()
    if callable(loss):
      with tape:
        if not callable(var_list):
          tape.watch(var_list)
        loss = loss()
        if callable(var_list):
          var_list = var_list()
    with tape:
      loss = self._transform_loss(loss)
    var_list = nest.flatten(var_list)
    with ops.name_scope_v2(self._name + "/gradients"):
      grads_and_vars = self._get_gradients(tape, loss, var_list, grad_loss)
    self._assert_valid_dtypes([
        v for g, v in grads_and_vars
        if g is not None and v.dtype != dtypes.resource
    ])
    return grads_and_vars
  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    """Apply gradients to variables.
    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.
    The method sums gradients from all replicas in the presence of
    `tf.distribute.Strategy` by default. You can aggregate gradients yourself by
    passing `experimental_aggregate_gradients=False`.
    Example:
    ```python
    grads = tape.gradient(loss, vars)
    grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    optimizer.apply_gradients(zip(grads, vars),
        experimental_aggregate_gradients=False)
    ```
    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      name: Optional name for the returned operation. Default to the name passed
        to the `Optimizer` constructor.
      experimental_aggregate_gradients: Whether to sum gradients from different
        replicas in the presense of `tf.distribute.Strategy`. If False, it's
        user responsibility to aggregate the gradients. Default to True.
    Returns:
      An `Operation` that applies the specified gradients. The `iterations`
      will be automatically increased by 1.
    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
      RuntimeError: If called in a cross-replica context.
    """
    grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]
    with ops.name_scope_v2(self._name):
      with ops.init_scope():
        self._create_all_weights(var_list)
      if not grads_and_vars:
        return control_flow_ops.no_op()
      if distribute_ctx.in_cross_replica_context():
        raise RuntimeError(
            "`apply_gradients() cannot be called in cross-replica context. "
            "Use `tf.distribute.Strategy.run` to enter replica "
            "context.")
      strategy = distribute_ctx.get_strategy()
      if (not experimental_aggregate_gradients and strategy and
          isinstance(strategy,
                     (parameter_server_strategy.ParameterServerStrategyV1,
                      parameter_server_strategy_v2.ParameterServerStrategyV2,
                      central_storage_strategy.CentralStorageStrategy,
                      central_storage_strategy.CentralStorageStrategyV1))):
        raise NotImplementedError(
            "`experimental_aggregate_gradients=False is not supported for "
            "ParameterServerStrategy and CentralStorageStrategy")
      apply_state = self._prepare(var_list)
      if experimental_aggregate_gradients:
        grads_and_vars = self._transform_unaggregated_gradients(grads_and_vars)
        grads_and_vars = self._aggregate_gradients(grads_and_vars)
      grads_and_vars = self._transform_gradients(grads_and_vars)
      if optimizer_utils.strategy_supports_no_merge_call():
        return self._distributed_apply(strategy, grads_and_vars, name,
                                       apply_state)
      else:
        return distribute_ctx.get_replica_context().merge_call(
            functools.partial(self._distributed_apply, apply_state=apply_state),
            args=(grads_and_vars,),
            kwargs={
                "name": name,
            })
  def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
    def apply_grad_to_update_var(var, grad):
      if isinstance(var, ops.Tensor):
        raise NotImplementedError("Trying to update a Tensor ", var)
      apply_kwargs = {}
      if isinstance(grad, indexed_slices.IndexedSlices):
        if var.constraint is not None:
          raise RuntimeError(
              "Cannot use a constraint function on a sparse variable.")
        if "apply_state" in self._sparse_apply_args:
          apply_kwargs["apply_state"] = apply_state
        return self._resource_apply_sparse_duplicate_indices(
            grad.values, var, grad.indices, **apply_kwargs)
      if "apply_state" in self._dense_apply_args:
        apply_kwargs["apply_state"] = apply_state
      update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
      if var.constraint is not None:
        with ops.control_dependencies([update_op]):
          return var.assign(var.constraint(var))
      else:
        return update_op
    eagerly_outside_functions = ops.executing_eagerly_outside_functions()
    update_ops = []
    with name_scope_only_in_function_or_graph(name or self._name):
      for grad, var in grads_and_vars:
        with distribution.extended.colocate_vars_with(var):
          with name_scope_only_in_function_or_graph(
              "update" if eagerly_outside_functions else "update_" +
              var.op.name):
            update_op = distribution.extended.update(
                var, apply_grad_to_update_var, args=(grad,), group=False)
            if distribute_ctx.in_cross_replica_context():
              update_ops.extend(update_op)
            else:
              update_ops.append(update_op)
      any_symbolic = any(isinstance(i, ops.Operation) or
                         tf_utils.is_symbolic_tensor(i) for i in update_ops)
      if not context.executing_eagerly() or any_symbolic:
          with ops.control_dependencies([control_flow_ops.group(update_ops)]):
            return self._iterations.assign_add(1, read_value=False)
      return self._iterations.assign_add(1)
  def get_gradients(self, loss, params):
    """Returns gradients of `loss` with respect to `params`.
    Should be used only in legacy v1 graph mode.
    Args:
      loss: Loss tensor.
      params: List of variables.
    Returns:
      List of gradient tensors.
    Raises:
      ValueError: In case any gradient cannot be computed (e.g. if gradient
        function not implemented).
    """
    params = nest.flatten(params)
    with backend.get_graph().as_default(), backend.name_scope(self._name +
                                                              "/gradients"):
      grads = gradients.gradients(loss, params)
      for grad, param in zip(grads, params):
        if grad is None:
          raise ValueError("Variable {} has `None` for gradient. "
                           "Please make sure that all of your ops have a "
                           "gradient defined (i.e. are differentiable). "
                           "Common ops without gradient: "
                           "K.argmax, K.round, K.eval.".format(param))
    return grads
  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    grads_and_vars = list(zip(grads, params))
    self._assert_valid_dtypes([
        v for g, v in grads_and_vars
        if g is not None and v.dtype != dtypes.resource
    ])
    return [self.apply_gradients(grads_and_vars)]
  def _set_hyper(self, name, value):
    if isinstance(value, trackable.Trackable):
      self._track_trackable(value, name, overwrite=True)
    if name not in self._hyper:
      self._hyper[name] = value
    else:
      prev_value = self._hyper[name]
      if (callable(prev_value)
          or isinstance(prev_value,
                        (ops.Tensor, int, float,
                         learning_rate_schedule.LearningRateSchedule))
          or isinstance(value, learning_rate_schedule.LearningRateSchedule)):
        self._hyper[name] = value
      else:
        backend.set_value(self._hyper[name], value)
  def _get_hyper(self, name, dtype=None):
    if not self._hypers_created:
      self._create_hypers()
    value = self._hyper[name]
    if isinstance(value, learning_rate_schedule.LearningRateSchedule):
      return value
    if callable(value):
      value = value()
    if dtype:
      return math_ops.cast(value, dtype)
    else:
      return value
  def _create_slots(self, var_list):
    pass
  def _create_all_weights(self, var_list):
    _ = self.iterations
    self._create_hypers()
    self._create_slots(var_list)
  def __getattribute__(self, name):
    try:
      return super(OptimizerV2, self).__getattribute__(name)
    except AttributeError as e:
      if name == "_hyper":
        raise e
      if name == "lr":
        name = "learning_rate"
      if name in self._hyper:
        return self._get_hyper(name)
      raise e
  def __dir__(self):
    result = set(super(OptimizerV2, self).__dir__())
    if "_hyper" in result:
      result |= self._hyper.keys()
      if "learning_rate" in self._hyper.keys():
        result.add("lr")
    return list(result)
  def __setattr__(self, name, value):
    if name == "lr":
      name = "learning_rate"
    if hasattr(self, "_hyper") and name in self._hyper:
      self._set_hyper(name, value)
    else:
      super(OptimizerV2, self).__setattr__(name, value)
  def get_slot_names(self):
    return self._slot_names
  def add_slot(self, var, slot_name, initializer="zeros", shape=None):
    """Add a new slot variable for `var`.
    A slot variable is an additional variable associated with `var` to train.
    It is allocated and managed by optimizers, e.g. `Adam`.
    Args:
      var: a `Variable` object.
      slot_name: name of the slot variable.
      initializer: initializer of the slot variable
      shape: (Optional) shape of the slot variable. If not set, it will default
      to the shape of `var`.
    Returns:
      A slot variable.
    """
    if slot_name not in self._slot_names:
      self._slot_names.append(slot_name)
    var_key = _var_key(var)
    slot_dict = self._slots.setdefault(var_key, {})
    weight = slot_dict.get(slot_name, None)
    if weight is None:
      if isinstance(initializer, str) or callable(initializer):
        initializer = initializers.get(initializer)
        if isinstance(
            initializer,
            trackable.CheckpointInitialValueCallable) or (shape is not None):
          slot_shape = shape
        else:
          slot_shape = var.shape
        initial_value = functools.partial(
            initializer, shape=slot_shape, dtype=var.dtype)
      else:
        initial_value = initializer
      with self._distribution_strategy_scope():
        strategy = distribute_ctx.get_strategy()
        if not strategy.extended.variable_created_in_scope(var):
          raise ValueError(
              "Trying to create optimizer slot variable under the scope for "
              "tf.distribute.Strategy ({}), which is different from the scope "
              "used for the original variable ({}). Make sure the slot "
              "variables are created under the same strategy scope. This may "
              "happen if you're restoring from a checkpoint outside the scope"
              .format(strategy, var))
        with strategy.extended.colocate_vars_with(var):
          weight = tf_variables.Variable(
              dtype=var.dtype,
              trainable=False,
              initial_value=initial_value)
      backend.track_variable(weight)
      slot_dict[slot_name] = weight
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=weight)
      self._weights.append(weight)
    return weight
  def get_slot(self, var, slot_name):
    var_key = _var_key(var)
    slot_dict = self._slots[var_key]
    return slot_dict[slot_name]
  def _prepare(self, var_list):
    keys = set()
    for var in var_list:
      if isinstance(var, ds_values.DistributedValues):
      else:
        var_devices = [var.device]
      var_dtype = var.dtype.base_dtype
      for var_device in var_devices:
        keys.add((var_device, var_dtype))
    apply_state = {}
    for var_device, var_dtype in keys:
      apply_state[(var_device, var_dtype)] = {}
      with ops.device(var_device):
        self._prepare_local(var_device, var_dtype, apply_state)
    return apply_state
  def _prepare_local(self, var_device, var_dtype, apply_state):
    if "learning_rate" in self._hyper:
      lr_t = array_ops.identity(self._decayed_lr(var_dtype))
      apply_state[(var_device, var_dtype)]["lr_t"] = lr_t
  def _fallback_apply_state(self, var_device, var_dtype):
    apply_state = {(var_device, var_dtype): {}}
    self._prepare_local(var_device, var_dtype, apply_state)
    return apply_state[(var_device, var_dtype)]
  def _create_hypers(self):
    if self._hypers_created:
      return
    with self._distribution_strategy_scope():
      for name, value in sorted(self._hyper.items()):
        if isinstance(value,
                      (ops.Tensor, tf_variables.Variable)) or callable(value):
          continue
        else:
          self._hyper[name] = self.add_weight(
              name,
              shape=[],
              trainable=False,
              initializer=value,
              aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
    self._hypers_created = True
  @property
  def iterations(self):
    if self._iterations is None:
      with self._distribution_strategy_scope():
        self._iterations = self.add_weight(
            "iter",
            shape=[],
            dtype=dtypes.int64,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
      self._weights.append(self._iterations)
    return self._iterations
  @iterations.setter
  def iterations(self, variable):
    if self._iterations is not None:
      raise RuntimeError("Cannot set `iterations` to a new Variable after "
                         "the Optimizer weights have been created")
    self._iterations = variable
    self._weights.append(self._iterations)
  def _decayed_lr(self, var_dtype):
    lr_t = self._get_hyper("learning_rate", var_dtype)
    if isinstance(lr_t, learning_rate_schedule.LearningRateSchedule):
      local_step = math_ops.cast(self.iterations, var_dtype)
      lr_t = math_ops.cast(lr_t(local_step), var_dtype)
    if self._initial_decay > 0.:
      local_step = math_ops.cast(self.iterations, var_dtype)
      decay_t = math_ops.cast(self._initial_decay, var_dtype)
      lr_t = lr_t / (1. + decay_t * local_step)
    return lr_t
  @abc.abstractmethod
  def get_config(self):
    """Returns the config of the optimizer.
    An optimizer config is a Python dictionary (serializable)
    containing the configuration of an optimizer.
    The same optimizer can be reinstantiated later
    (without any saved state) from this configuration.
    Returns:
        Python dictionary.
    """
    config = {"name": self._name}
    if self.clipnorm is not None:
      config["clipnorm"] = self.clipnorm
    if self.clipvalue is not None:
      config["clipvalue"] = self.clipvalue
    if self.global_clipnorm is not None:
      config["global_clipnorm"] = self.global_clipnorm
    return config
  @classmethod
  def from_config(cls, config, custom_objects=None):
    if "lr" in config:
      config["learning_rate"] = config.pop("lr")
    if "learning_rate" in config:
      if isinstance(config["learning_rate"], dict):
        config["learning_rate"] = learning_rate_schedule.deserialize(
            config["learning_rate"], custom_objects=custom_objects)
    return cls(**config)
  def _serialize_hyperparameter(self, hyperparameter_name):
    value = self._hyper[hyperparameter_name]
    if isinstance(value, learning_rate_schedule.LearningRateSchedule):
      return learning_rate_schedule.serialize(value)
    if callable(value):
      return value()
    if tensor_util.is_tf_type(value):
      return backend.get_value(value)
    return value
  def variables(self):
    return self._weights
  @property
  def weights(self):
    return self._weights
  def get_weights(self):
    """Returns the current weights of the optimizer.
    The weights of an optimizer are its state (ie, variables).
    This function returns the weight values associated with this
    optimizer as a list of Numpy arrays. The first value is always the
    iterations count of the optimizer, followed by the optimizer's state
    variables in the order they were created. The returned list can in turn
    be used to load state into similarly parameterized optimizers.
    For example, the RMSprop optimizer for this simple model returns a list of
    three values-- the iteration count, followed by the root-mean-square value
    of the kernel and bias of the single Dense layer:
    >>> opt = tf.keras.optimizers.RMSprop()
    >>> m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> m.compile(opt, loss='mse')
    >>> data = np.arange(100).reshape(5, 20)
    >>> labels = np.zeros(5)
    >>> len(opt.get_weights())
    3
    Returns:
        Weights values as a list of numpy arrays.
    """
    params = self.weights
    return backend.batch_get_value(params)
  def set_weights(self, weights):
    """Set the weights of the optimizer.
    The weights of an optimizer are its state (ie, variables).
    This function takes the weight values associated with this
    optimizer as a list of Numpy arrays. The first value is always the
    iterations count of the optimizer, followed by the optimizer's state
    variables in the order they are created. The passed values are used to set
    the new state of the optimizer.
    For example, the RMSprop optimizer for this simple model takes a list of
    three values-- the iteration count, followed by the root-mean-square value
    of the kernel and bias of the single Dense layer:
    >>> opt = tf.keras.optimizers.RMSprop()
    >>> m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> m.compile(opt, loss='mse')
    >>> data = np.arange(100).reshape(5, 20)
    >>> labels = np.zeros(5)
    >>> new_weights = [np.array(10), np.ones([20, 10]), np.zeros([10])]
    >>> opt.set_weights(new_weights)
    >>> opt.iterations
    <tf.Variable 'RMSprop/iter:0' shape=() dtype=int64, numpy=10>
    Args:
        weights: weight values as a list of numpy arrays.
    """
    params = self.weights
    if len(params) != len(weights):
      raise ValueError(
          "You called `set_weights(weights)` on optimizer " + self._name +
          " with a  weight list of length " + str(len(weights)) +
          ", but the optimizer was expecting " + str(len(params)) +
          " weights. Provided weights: " + str(weights)[:50] + "...")
    if not params:
      return
    weight_value_tuples = []
    param_values = backend.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError("Optimizer weight shape " + str(pv.shape) +
                         " not compatible with "
                         "provided weight shape " + str(w.shape))
      weight_value_tuples.append((p, w))
    backend.batch_set_value(weight_value_tuples)
  def add_weight(self,
                 name,
                 shape,
                 dtype=None,
                 initializer="zeros",
                 trainable=None,
                 synchronization=tf_variables.VariableSynchronization.AUTO,
                 aggregation=tf_variables.VariableAggregation.NONE):
    if dtype is None:
      dtype = dtypes.float32
    if isinstance(initializer, str) or callable(initializer):
      initializer = initializers.get(initializer)
    if synchronization == tf_variables.VariableSynchronization.ON_READ:
      if trainable:
        raise ValueError(
            "Synchronization value can be set to "
            "VariableSynchronization.ON_READ only for non-trainable variables. "
            "You have specified trainable=True and "
            "synchronization=VariableSynchronization.ON_READ.")
      else:
        trainable = False
    elif trainable is None:
      trainable = True
    variable = self._add_variable_with_custom_getter(
        name=name,
        shape=shape,
        getter=base_layer_utils.make_variable,
        overwrite=True,
        initializer=initializer,
        dtype=dtype,
        trainable=trainable,
        use_resource=True,
        synchronization=synchronization,
        aggregation=aggregation)
    backend.track_variable(variable)
    return variable
  def _init_set_name(self, name, zero_based=True):
    if not name:
      self._name = backend.unique_object_name(
          generic_utils.to_snake_case(self.__class__.__name__),
          zero_based=zero_based)
    else:
      self._name = name
  def _assert_valid_dtypes(self, tensors):
    """Asserts tensors are all valid types (see `_valid_dtypes`).
    Args:
      tensors: Tensors to check.
    Raises:
      ValueError: If any tensor is not a valid type.
    """
    valid_dtypes = self._valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype
      if dtype not in valid_dtypes:
        raise ValueError("Invalid type %r for %s, expected: %s." %
                         (dtype, t.name, [v for v in valid_dtypes]))
  def _valid_dtypes(self):
    return _DEFAULT_VALID_DTYPES
  def _call_if_callable(self, param):
    return param() if callable(param) else param
  def _resource_apply_dense(self, grad, handle, apply_state):
    raise NotImplementedError("Must be implemented in subclasses.")
  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices,
                                               **kwargs):
    summed_grad, unique_indices = _deduplicate_indexed_slices(
        values=grad, indices=indices)
    return self._resource_apply_sparse(summed_grad, handle, unique_indices,
                                       **kwargs)
  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    raise NotImplementedError("Must be implemented in subclasses.")
  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies([
        gen_resource_variable_ops.ResourceScatterAdd(
            resource=x.handle, indices=i, updates=v)
    ]):
      return x.value()
  def _resource_scatter_update(self, x, i, v):
    with ops.control_dependencies(
        [gen_resource_variable_ops.ResourceScatterUpdate(
            resource=x.handle, indices=i, updates=v)]):
      return x.value()
  @property
  @layer_utils.cached_per_instance
  def _dense_apply_args(self):
    return tf_inspect.getfullargspec(self._resource_apply_dense).args
  @property
  @layer_utils.cached_per_instance
  def _sparse_apply_args(self):
    return tf_inspect.getfullargspec(self._resource_apply_sparse).args
  def _restore_slot_variable(self, slot_name, variable, slot_variable):
    variable_key = _var_key(variable)
    deferred_restorations = self._deferred_slot_restorations.get(
        slot_name, {}).pop(variable_key, [])
    deferred_restorations.sort(key=lambda position: position.restore_uid,
                               reverse=True)
    for checkpoint_position in deferred_restorations:
      checkpoint_position.restore(slot_variable)
  def _create_or_restore_slot_variable(
      self, slot_variable_position, slot_name, variable):
    """Restore a slot variable's value, possibly creating it.
    Called when a variable which has an associated slot variable is created or
    restored. When executing eagerly, we create the slot variable with a
    restoring initializer.
    No new variables are created when graph building. Instead,
    _restore_slot_variable catches these after normal creation and adds restore
    ops to the graph. This method is nonetheless important when graph building
    for the case when a slot variable has already been created but `variable`
    has just been added to a dependency graph (causing us to realize that the
    slot variable needs to be restored).
    Args:
      slot_variable_position: A `trackable._CheckpointPosition` object
        indicating the slot variable `Trackable` object to be restored.
      slot_name: The name of this `Optimizer`'s slot to restore into.
      variable: The variable object this slot is being created for.
    """
    variable_key = _var_key(variable)
    slot_dict = self._slots.get(variable_key, {})
    slot_variable = slot_dict.get(slot_name, None)
    if (slot_variable is None and context.executing_eagerly() and
        slot_variable_position.is_simple_variable()
             self._distribution_strategy)):
      initializer = trackable.CheckpointInitialValueCallable(
          checkpoint_position=slot_variable_position)
      slot_variable = self.add_slot(
          var=variable,
          initializer=initializer,
          slot_name=slot_name,
          shape=slot_variable_position.value_shape())
    if slot_variable is not None:
      slot_variable_position.restore(slot_variable)
    else:
      self._deferred_slot_restorations.setdefault(
          slot_name, {}).setdefault(variable_key, []).append(
              slot_variable_position)
  @contextlib.contextmanager
  def _distribution_strategy_scope(self):
    if self._distribution_strategy and not distribute_ctx.has_strategy():
      with self._distribution_strategy.scope():
        yield self._distribution_strategy.scope()
    else:
      yield
def _var_key(var):
  if hasattr(var, "_distributed_container"):
    var = var._distributed_container()
  if var._in_graph_mode:
    return var._shared_name
  return var._unique_id
def _get_slot_key_from_var(var, slot_name):
  name = _var_key(var)
  return name + "/" + slot_name
class RestoredOptimizer(OptimizerV2):
  """A non-functional Optimizer implementation for checkpoint compatibility.
  Holds slot variables and hyperparameters when an optimizer is restored from a
  SavedModel. These variables may be referenced in functions along with ops
  created by the original optimizer, but currently we do not support using the
  optimizer object iself (e.g. through `apply_gradients`).
  """
  def __init__(self):
    super(RestoredOptimizer, self).__init__("RestoredOptimizer")
    self._hypers_created = True
  def get_config(self):
    raise NotImplementedError(
        "Restoring functional Optimizers from SavedModels is not currently "
        "supported. Please file a feature request if this limitation bothers "
        "you.")
revived_types.register_revived_type(
    "tf_deprecated_optimizer",
    lambda obj: isinstance(obj, OptimizerV2),
    versions=[revived_types.VersionedTypeRegistration(
        object_factory=lambda proto: RestoredOptimizer(),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
    )])