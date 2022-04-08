
import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.training.tracking import base as tracking
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
_call_context = threading.local()
def create_mean_metric(value, name=None):
  metric_obj = metrics_module.Mean(name=name, dtype=value.dtype)
  return metric_obj, metric_obj(value)
def make_variable(name,
                  shape=None,
                  dtype=dtypes.float32,
                  initializer=None,
                  trainable=None,
                  caching_device=None,
                  validate_shape=True,
                  constraint=None,
                  use_resource=None,
                  collections=None,
                  synchronization=tf_variables.VariableSynchronization.AUTO,
                  aggregation=tf_variables.VariableAggregation.NONE,
  """Temporary util to create a variable (relies on `variable_scope.variable`).
  Some reuse-related technicalities prevent us from using
  `variable_scope.get_variable()` directly, so we use a subcomponent
  that has fewer constraints (`variable_scope.variable()`).
  In the longer term, it seems like a similar "default variable creator" method
  should exist in `Trackable` instead. When this happens, we can get
  rid of this temporary solution.
  TODO(fchollet): remove this method when no longer needed.
  Args:
    name: Variable name.
    shape: Variable shape.
    dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
    initializer: Initializer instance (callable).
    trainable: Whether the variable should be part of the layer's
      "trainable_variables" (e.g. variables, biases)
      or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
      Note, if the current variable scope is marked as non-trainable
      then this parameter is ignored and any added variables are also
      marked as non-trainable. `trainable` defaults to `True` unless
      `synchronization` is set to `ON_READ`.
    caching_device: Passed to `tf.Variable`.
    validate_shape: Passed to `tf.Variable`.
    constraint: Constraint instance (callable).
    use_resource: Whether to use a `ResourceVariable`.
    collections: List of graph collections keys. The new variable is added to
      these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
    synchronization: Indicates when a distributed a variable will be
      aggregated. Accepted values are constants defined in the class
      `tf.VariableSynchronization`. By default the synchronization is set to
      `AUTO` and the current `DistributionStrategy` chooses
      when to synchronize. If `synchronization` is set to `ON_READ`,
      `trainable` must not be set to `True`.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableAggregation`.
    partitioner: Not handled at this time.
  Returns:
    Variable instance.
  """
  initializing_from_value = False
  if initializer is not None and not callable(initializer):
    initializing_from_value = True
  if initializing_from_value:
    init_val = initializer
    variable_dtype = None
  else:
    if tf_inspect.isclass(initializer):
      initializer = initializer()
    init_val = functools.partial(initializer, shape, dtype=dtype)
    variable_dtype = dtype.base_dtype
  if use_resource is None:
    use_resource = True
  variable_shape = tensor_shape.TensorShape(shape)
  return tf_variables.VariableV1(
      initial_value=init_val,
      name=name,
      trainable=trainable,
      caching_device=caching_device,
      dtype=variable_dtype,
      validate_shape=validate_shape,
      constraint=constraint,
      use_resource=use_resource,
      collections=collections,
      synchronization=synchronization,
      aggregation=aggregation,
      shape=variable_shape if variable_shape else None)
def collect_previous_mask(input_tensors):
  """Retrieves the output mask(s) of the previous node.
  Args:
      input_tensors: An arbitrary structure of Tensors.
  Returns:
      A mask tensor or list of mask tensors.
  """
  def _collect_previous_mask(x):
    return getattr(x, '_keras_mask', None)
  return nest.map_structure(_collect_previous_mask, input_tensors)
def have_all_keras_metadata(tensors):
  return all(hasattr(x, '_keras_history') for x in nest.flatten(tensors))
def generate_placeholders_from_shape(shape):
  return array_ops.placeholder(shape=shape, dtype=backend.floatx())
def create_keras_history(tensors):
  _, created_layers = _create_keras_history_helper(tensors, set(), [])
  return created_layers
_UNSAFE_GRAPH_OP_LAYER_CREATION = False
def _create_keras_history_helper(tensors, processed_ops, created_layers):
  if ops.executing_eagerly_outside_functions():
    raise ValueError(
        '`create_keras_history` should only be called if eager is disabled!')
  tensor_list = nest.flatten(tensors)
  sparse_ops = []
  ragged_tensors = []
  for tensor in tensor_list:
    if getattr(tensor, '_keras_history', None) is not None:
      continue
    if isinstance(
        tensor, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
      sparse_ops.append(tensor.op)
      continue
    if tf_utils.is_ragged(tensor):
      ragged_tensors.append(tensor)
      continue
    if op not in processed_ops:
      op_inputs = list(op.inputs)
      constants = {}
      layer_inputs = []
      for i, op_input in enumerate(op_inputs):
        if uses_keras_history(op_input):
          layer_inputs.append(op_input)
        else:
          ds_with_session = (
              distribution_strategy_context.in_cross_replica_context() and
              not ops.executing_eagerly_outside_functions())
          using_xla = control_flow_util.GraphOrParentsInXlaContext(
              ops.get_default_graph())
          if ds_with_session or using_xla or _UNSAFE_GRAPH_OP_LAYER_CREATION:
            constants[i] = op_input
          else:
            with ops.init_scope():
              constants[i] = backend.function([], op_input)([])
      layer_inputs = unnest_if_single_tensor(layer_inputs)
      processed_ops, created_layers = _create_keras_history_helper(
          layer_inputs, processed_ops, created_layers)
      name = op.name
      node_def = op.node_def.SerializeToString()
      op_layer = base_layer.TensorFlowOpLayer(
          node_def, constants=constants, name=name)
      created_layers.append(op_layer)
          args=(layer_inputs,),
          kwargs={},
          outputs=op.outputs)
      processed_ops.update([op])
  if sparse_ops or ragged_tensors:
    lambda_example = """
    weights_mult = lambda x: tf.sparse.sparse_dense_matmul(x, weights)
    output = tf.keras.layers.Lambda(weights_mult)(input)
    """
    raise ValueError(
        'Tensorflow ops that generate ragged or sparse tensor '
        'outputs are currently not supported by Keras automatic '
        'op wrapping. Please wrap these ops in a Lambda layer: '
        '\n\n```\n{example}\n```\n'
        'Sparse ops encountered: {sparse_ops}\n'
        'Ragged tensors encountered: {ragged_tensors}\n'.format(
            example=lambda_example,
            sparse_ops=str(sparse_ops),
            ragged_tensors=str(ragged_tensors)))
  return processed_ops, created_layers
def unnest_if_single_tensor(input_tensors):
  flat_input_tensors = nest.flatten(input_tensors)
  if not isinstance(input_tensors, dict) and len(flat_input_tensors) == 1:
    input_tensors = flat_input_tensors[0]
  return input_tensors
def needs_keras_history(tensors, ignore_call_context=False):
  input_tensors = nest.flatten(tensors)
  if call_context().in_call and not ignore_call_context:
    return False
  if all(
      getattr(tensor, '_keras_history', None) is not None
      for tensor in input_tensors):
    return False
  return uses_keras_history(tensors)
def is_in_keras_graph():
  return call_context().in_keras_graph
def is_in_eager_or_tf_function():
  return context.executing_eagerly() or is_in_tf_function()
def is_in_tf_function():
  if not ops.executing_eagerly_outside_functions():
    return False
  if not ops.inside_function():
    return False
  if is_in_keras_graph():
    return False
  graph = ops.get_default_graph()
  if (getattr(graph, 'name', False) and
      graph.name.startswith('wrapped_function')):
    return False
  return True
def uses_keras_history(tensors):
  checked_tensors = set()
  tensors_to_check = nest.flatten(tensors)
  while tensors_to_check:
    new_tensors_to_check = []
    for tensor in tensors_to_check:
      if id(tensor) in checked_tensors:
        continue
      checked_tensors.add(id(tensor))
      if getattr(tensor, '_keras_history_checked', None) is not None:
        continue
      if getattr(tensor, '_keras_history', None) is not None:
        return True
      try:
        new_tensors_to_check.extend(tensor.op.inputs)
      except AttributeError:
        pass
    tensors_to_check = new_tensors_to_check
  mark_checked(tensors)
  return False
def mark_checked(tensors):
  def _mark_checked(tensor):
  nest.map_structure(_mark_checked, tensors)
def call_context():
  call_ctx = getattr(_call_context, 'call_context', None)
  if call_ctx is None:
    call_ctx = CallContext()
    _call_context.call_context = call_ctx
  return call_ctx
keras_deps.register_call_context_function(call_context)
class CallContext(object):
  def __init__(self):
    self.in_call = False
    self._state = {
        'layer': None,
        'inputs': None,
        'build_graph': False,
        'training': None,
        'saving': None
    }
    self._in_keras_graph = False
  def enter(self, layer, inputs, build_graph, training, saving=None):
    state = {
        'layer': layer,
        'inputs': inputs,
        'build_graph': build_graph,
        'training': training,
        'saving': saving
    }
    return CallContextManager(self, state)
  @property
  def layer(self):
    return self._state['layer']
  @property
  def inputs(self):
    return self._state['inputs']
  @property
  def build_graph(self):
    return self._state['build_graph']
  @property
  def training(self):
    return self._state['training']
  @property
  def saving(self):
    return self._state['saving']
  @property
  def frozen(self):
    layer = self._state['layer']
    if not layer:
      return False
    return not layer.trainable
  @property
  def in_keras_graph(self):
    if context.executing_eagerly():
      return False
    return (self._in_keras_graph or
            getattr(backend.get_graph(), 'name', None) == 'keras_graph')
class CallContextManager(object):
  def __init__(self, call_ctx, state):
    self._call_ctx = call_ctx
    self._state = state
    self._build_graph = state['build_graph']
  def __enter__(self):
    call_ctx = self._call_ctx
    self._prev_in_call = call_ctx.in_call
    self._prev_state = call_ctx._state
    call_ctx.in_call = True
    call_ctx._state = self._state
    if self._build_graph:
      self._prev_in_keras_graph = call_ctx._in_keras_graph
      call_ctx._in_keras_graph = (
          call_ctx._in_keras_graph or
          getattr(backend.get_graph(), 'name', None) == 'keras_graph')
  def __exit__(self, *exc_info):
    call_ctx = self._call_ctx
    call_ctx.in_call = self._prev_in_call
    call_ctx._state = self._prev_state
    if self._build_graph:
      call_ctx._in_keras_graph = self._prev_in_keras_graph
def training_arg_passed_to_call(argspec, args, kwargs):
  full_args = dict(zip(argspec.args[2:], args))
  full_args.update(kwargs)
  return 'training' in full_args and full_args['training'] is not None
def is_subclassed(layer):
  return (layer.__module__.find('keras.engine') == -1 and
          layer.__module__.find('keras.layers') == -1)
def from_saved_model(layer):
  return layer.__module__.find('keras.saving.saved_model') != -1
def check_graph_consistency(tensor=None, method='add_loss', force_raise=False):
  if (force_raise or
      (ops.executing_eagerly_outside_functions() and
       hasattr(tensor, 'graph') and tensor.graph.is_control_flow_graph)):
    if method == 'activity_regularizer':
      bad_example = """
      class TestModel(tf.keras.Model):
        def __init__(self):
          super(TestModel, self).__init__(name='test_model')
          self.dense = tf.keras.layers.Dense(2, activity_regularizer='l2')
        def call(self, x, training=None):
          if training:
            return self.dense(x)
          else:
            return self.dense(x)
      class TestModel(tf.keras.Model):
        def __init__(self):
          super(TestModel, self).__init__(name='test_model')
          self.dense = tf.keras.layers.Dense(2, activity_regularizer='l2')
        def call(self, x, training=None):
          return self.dense(x)
      """
      raise RuntimeError(
          'You are using a layer with `activity_regularizer` in a control flow '
          'branch, e.g.:\n{bad_example}\nThis is currently not supported. '
          'Please move your call to the layer with `activity_regularizer` out '
          'of the control flow branch, e.g.:\n{correct_example}\n'
          'You can also resolve this by marking your outer model/layer dynamic'
          ' (eager-only) by passing `dynamic=True` to the layer constructor. '
          'Any kind of control flow is supported with dynamic layers. '
          'Note that using `dynamic=True` requires you to implement static '
          'shape inference in the `compute_output_shape(input_shape)` '
          'method.'.format(
              bad_example=bad_example, correct_example=correct_example))
    if method == 'add_metric':
      bad_example = """
      def call(self, inputs, training=None):
        if training:
          metric = compute_metric(inputs)
          self.add_metric(metric, name='my_metric', aggregation='mean')
        return inputs
      def call(self, inputs, training=None):
        if training:
          metric = compute_metric(inputs)
        else:
          metric = 0.
        self.add_metric(metric, name='my_metric', aggregation='mean')
        return inputs
      def call(self, inputs, training=None):
        if training:
          loss = compute_loss(inputs)
          self.add_loss(loss)
        return inputs
      def call(self, inputs, training=None):
        if training:
          loss = compute_loss(inputs)
        else:
          loss = 0.
        self.add_loss(loss)
        return inputs
      def call(self, inputs, training=None):
        if training:
          self.add_update(self.w.assign_add(1))
        return inputs
      def call(self, inputs, training=None):
        if training:
          increment = 1
        else:
          increment = 0
        self.add_update(self.w.assign_add(increment))
        return inputs
      """
    raise RuntimeError(
        'You are using the method `{method}` in a control flow branch '
        'in your layer, e.g.:\n{bad_example}\n'
        'This is not currently supported. '
        'Please move your call to {method} out of the control flow branch, '
        'e.g.:\n{correct_example}\n'
        'You can also resolve this by marking your layer '
        'as dynamic (eager-only) by passing '
        '`dynamic=True` to the layer constructor. '
        'Any kind of control flow is supported with dynamic layers. '
        'Note that using `dynamic=True` requires you '
        'to implement static shape inference '
        'in the `compute_output_shape(input_shape)` method.'.format(
            method=method,
            bad_example=bad_example,
            correct_example=correct_example))
def mark_as_return(outputs, acd):
  def _mark_as_return(tensor):
    if not tensor_util.is_tf_type(tensor):
      return tensor
    return_tensor = acd.mark_as_return(tensor)
    if getattr(tensor, '_keras_mask', None) is not None:
      return_tensor._keras_mask = acd.mark_as_return(tensor._keras_mask)
    else:
      return_tensor._keras_mask = None
    if getattr(tensor, '_tfp_distribution', None) is not None:
      return_tensor._tfp_distribution = tensor._tfp_distribution
    return return_tensor
  return nest.map_structure(_mark_as_return, outputs)
V2_DTYPE_BEHAVIOR = None
@keras_export(v1=['keras.layers.enable_v2_dtype_behavior'])
def enable_v2_dtype_behavior():
  """Enable the V2 dtype behavior for Keras layers.
  By default, the V2 dtype behavior is enabled in TensorFlow 2, so this function
  is only useful if `tf.compat.v1.disable_v2_behavior` has been called. Since
  mixed precision requires V2 dtype behavior to be enabled, this function allows
  you to use mixed precision in Keras layers if `disable_v2_behavior` has been
  called.
  When enabled, the dtype of Keras layers defaults to floatx (which is typically
  float32) instead of None. In addition, layers will automatically cast
  floating-point inputs to the layer's dtype.
  >>> x = tf.ones((4, 4, 4, 4), dtype='float64')
  >>> layer = tf.keras.layers.Conv2D(filters=4, kernel_size=2)
  float32
  >>> print(y.dtype.name)
  float32
  A layer author can opt-out their layer from the automatic input casting by
  passing `autocast=False` to the base Layer's constructor. This disables the
  autocasting part of the V2 behavior for that layer, but not the defaulting to
  floatx part of the V2 behavior.
  When a global `tf.keras.mixed_precision.Policy` is set, a Keras layer's dtype
  will default to the global policy instead of floatx. Layers will automatically
  cast inputs to the policy's compute_dtype.
  """
  global V2_DTYPE_BEHAVIOR
  V2_DTYPE_BEHAVIOR = True
@keras_export(v1=['keras.layers.disable_v2_dtype_behavior'])
def disable_v2_dtype_behavior():
  global V2_DTYPE_BEHAVIOR
  V2_DTYPE_BEHAVIOR = False
def v2_dtype_behavior_enabled():
  if V2_DTYPE_BEHAVIOR is None:
    return tf2.enabled()
  return V2_DTYPE_BEHAVIOR
class TrackableWeightHandler(object):
  def __init__(self, trackable):
    if not isinstance(trackable, tracking.Trackable):
      raise ValueError('%s is not a Trackable object.' % (trackable,))
    self._trackable = trackable
    self._distribute_strategy = distribution_strategy_context.get_strategy()
    if not saveables:
      self._num_tensors = 0
      self._setter = lambda weights: None
      self._getter = lambda: []
    elif len(saveables) == 1:
      saveable = list(saveables)[0]
      if ops.executing_eagerly_outside_functions():
        self._saveable = saveable
        self._num_tensors = len(self._saveable().specs)
        self._setter = lambda weights: self._saveable().restore(weights, None)
        self._getter = lambda: [spec.tensor for spec in self._saveable().specs]
      else:
        self._placeholder_tensors = []
        self._saveable = saveable()
        self._num_tensors = len(self._saveable.specs)
        for spec in self._saveable.specs:
          tensor = spec.tensor
          self._placeholder_tensors.append(
              array_ops.placeholder(tensor.dtype, tensor.shape))
        self._assign_op = self._saveable.restore(self._placeholder_tensors,
                                                 None)
        self._setter = self._set_weights_v1
        self._getter = lambda: [spec.tensor for spec in self._saveable.specs]
    else:
      raise ValueError('Only Trackables with one Saveable are supported. '
                       'The Trackable %s has %d Saveables.' %
                       (trackable, len(saveables)))
  @property
  def num_tensors(self):
    return self._num_tensors
  def set_weights(self, weights):
    if len(weights) != self._num_tensors:
      raise ValueError(
          ('Weight handler for trackable %s received the wrong number of ' +
           'weights: expected %s, got %s.') %
          (self._trackable, self._num_tensors, len(weights)))
    self._setter(weights)
  def get_tensors(self):
    return self._getter()
  def _set_weights_v1(self, weights):
    feed_dict = {}
    for idx, tensor in enumerate(weights):
      feed_dict[self._placeholder_tensors[idx]] = tensor
    backend.get_session().run(self._assign_op, feed_dict)
class StaticTableHandler(TrackableWeightHandler):
    self._num_tensors = 2
    self._getter = getter_lambda
    self._distribute_strategy = distribution_strategy_context.get_strategy()
    def raise_error(_):
      raise RuntimeError('This layer contains a static lookup table, which '
                         'cannot be changed via set_weights().')
    self._setter = raise_error
def no_ragged_support(inputs, layer_name):
  input_list = nest.flatten(inputs)
  if any(isinstance(x, ragged_tensor.RaggedTensor) for x in input_list):
    raise ValueError('Layer %s does not support RaggedTensors as input. '
                     'Inputs received: %s. You can try converting your '
                     'input to an uniform tensor.' % (layer_name, inputs))
def is_split_variable(v):
  return hasattr(v, '_variable_list') or hasattr(v, '_variables')
def has_weights(obj):
  obj_type = type(obj)
  return (hasattr(obj_type, 'trainable_weights') and
          hasattr(obj_type, 'non_trainable_weights') and
          not isinstance(obj, type))
REVIVED_LOSS_PLACEHOLDER = (
    'This layer\'s losses have been added to the parent layer.')
