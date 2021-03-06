
import functools
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def set_weights(distribution_strategy, dist_model, weights):
  assign_ops = []
  for layer in dist_model.layers:
    num_param = len(layer.weights)
    layer_weights = weights[:num_param]
    for sw, w in zip(layer.weights, layer_weights):
      if ops.executing_eagerly_outside_functions():
        sw.assign(w)
      else:
        assign_ops.append(distribution_strategy.unwrap(sw.assign(w)))
    weights = weights[num_param:]
  if not ops.executing_eagerly_outside_functions():
    backend.get_session(assign_ops).run(assign_ops)
def unwrap_values(distribution_strategy, grouped_inputs, grouped_outputs,
                  grouped_updates=None, grouped_session_args=None,
                  with_loss_tensor=False):
  all_inputs = flatten_per_replica_values(distribution_strategy,
                                          grouped_inputs)
  all_outputs = unwrap_outputs(distribution_strategy, grouped_outputs,
                               with_loss_tensor)
  if grouped_updates:
    all_updates = flatten_per_replica_values(distribution_strategy,
                                             grouped_updates)
  else:
    all_updates = None
  all_session_args = {}
  if grouped_session_args:
    grouped_feed_dict = grouped_session_args.get('feed_dict')
    if grouped_feed_dict:
      all_session_args['feed_dict'] = flatten_per_replica_values(
          distribution_strategy, grouped_feed_dict)
    grouped_fetches = grouped_session_args.get('fetches')
    if grouped_fetches:
      all_session_args['fetches'] = flatten_per_replica_values(
          distribution_strategy, grouped_fetches)
  return all_inputs, all_outputs, all_updates, all_session_args
def unwrap_output_dict(strategy, grouped_outputs, mode):
  if mode == ModeKeys.PREDICT:
    return flatten_per_replica_values(strategy, grouped_outputs)
  total_loss = strategy.reduce(reduce_util.ReduceOp.SUM,
                               grouped_outputs['total_loss'][0], axis=None)
  output_losses = flatten_per_replica_values(strategy,
                                             grouped_outputs['output_losses'])
  metrics = flatten_per_replica_values(strategy,
                                       grouped_outputs['metrics'])
  batch_size = strategy.reduce(reduce_util.ReduceOp.SUM,
                               grouped_outputs['batch_size'], axis=None)
  if (backend.is_tpu_strategy(strategy) and
      ops.executing_eagerly_outside_functions()):
    output_losses = output_losses[::strategy.num_replicas_in_sync]
    metrics = metrics[::strategy.num_replicas_in_sync]
  return {'total_loss': [total_loss],
          'output_losses': output_losses,
          'metrics': metrics,
          'batch_size': batch_size}
def unwrap_outputs(distribution_strategy, grouped_outputs,
                   with_loss_tensor=False):
  if not with_loss_tensor:
    return flatten_per_replica_values(distribution_strategy,
                                      grouped_outputs)
  if not isinstance(grouped_outputs, list):
    grouped_outputs = [grouped_outputs]
  loss = distribution_strategy.reduce(reduce_util.ReduceOp.SUM,
                                      grouped_outputs[0], axis=None)
  all_outputs = flatten_per_replica_values(distribution_strategy,
                                           grouped_outputs[1:])
  if (backend.is_tpu_strategy(distribution_strategy) and
      ops.executing_eagerly_outside_functions()):
    all_outputs = all_outputs[::distribution_strategy.num_replicas_in_sync]
  return [loss] + all_outputs
def flatten_per_replica_values(distribution_strategy, per_replica_values):
  return [e for flattened in nest.flatten(per_replica_values)
          for e in distribution_strategy.unwrap(flattened)]
def validate_callbacks(input_callbacks, optimizer):
  if input_callbacks:
    for callback in input_callbacks:
      if isinstance(callback, (callbacks.LearningRateScheduler,
                               callbacks.ReduceLROnPlateau)):
        if not isinstance(optimizer, optimizer_v2.OptimizerV2):
          raise ValueError('You must specify a Keras Optimizer V2 when using '
                           '%s callback with DistributionStrategy.' % callback)
      if isinstance(callback, callbacks.TensorBoard):
        if getattr(callback, 'write_grads', False):
          logging.warning(
              UserWarning(
                  '`write_grads` in the TensorBoard callback is not supported '
                  'when using DistributionStrategy. Setting `write_grads` '
                  'to `False`.'))
          callback.write_grads = False
def validate_distributed_dataset_inputs(distribution_strategy, x, y,
                                        sample_weights=None):
  x_values_list = validate_per_replica_inputs(distribution_strategy, x)
  if y is not None:
    y_values_list = validate_per_replica_inputs(distribution_strategy, y)
  else:
    y_values_list = None
  if sample_weights is not None:
    sample_weights_list = validate_per_replica_inputs(distribution_strategy,
                                                      sample_weights)
  else:
    sample_weights_list = None
  return x_values_list, y_values_list, sample_weights_list
def validate_per_replica_inputs(distribution_strategy, x):
  per_replica_list = nest.flatten(x, expand_composites=True)
  x_values_list = []
  for x in per_replica_list:
    x_values = distribution_strategy.unwrap(x)
    for value in x_values:
      if not tensor_util.is_tf_type(value):
        raise ValueError('Dataset input to the model should be tensors instead '
                         'they are of type {}'.format(type(value)))
    if not context.executing_eagerly():
      validate_all_tensor_shapes(x, x_values)
    validate_all_tensor_types(x, x_values)
    x_values_list.append(x_values[0])
  return x_values_list
def validate_all_tensor_types(x, x_values):
  x_dtype = x_values[0].dtype
  for i in range(1, len(x_values)):
    if x_dtype != x_values[i].dtype:
      raise ValueError('Input tensor dtypes do not match for distributed tensor'
                       ' inputs {}'.format(x))
def validate_all_tensor_shapes(x, x_values):
  x_shape = x_values[0].shape.as_list()
  for i in range(1, len(x_values)):
    if x_shape != x_values[i].shape.as_list():
      raise ValueError('Input tensor shapes do not match for distributed tensor'
                       ' inputs {}'.format(x))
def _wait_for_variable_initialization(session):
  candidate_vars = []
  for v in all_variables:
    if not getattr(v, '_keras_initialized', False):
      candidate_vars.append(v)
  if not candidate_vars:
    return
  while True:
    is_initialized = session.run(
        [variables.is_variable_initialized(v) for v in candidate_vars])
    uninitialized_vars = []
    for flag, v in zip(is_initialized, candidate_vars):
      if not flag:
        uninitialized_vars.append(v)
    if not uninitialized_vars:
      break
def init_restore_or_wait_for_variables():
def validate_inputs(x, y):
  """Validate inputs when using DistributionStrategy.
  Args:
    x: Model Inputs.
    y: Model Targets.
  Raises:
    ValueError: if input is not a Dataset or a numpy array(when we use
      MirroredStrategy).
  """
  if (isinstance(x, iterator_ops.Iterator) or
      isinstance(y, iterator_ops.Iterator)):
    raise ValueError('`DistributionStrategy` does not support inputs of type '
                     'Iterator. You must pass a `tf.data.Dataset` object or a '
                     'numpy array as input.')
def is_dataset_shape_fully_defined(dataset):
  shapes = nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))
  unknown_shapes = [s for s in shapes if not s.is_fully_defined()]
  return not unknown_shapes
def process_batch_and_step_size(strategy,
                                inputs,
                                batch_size,
                                steps_per_epoch,
                                mode,
                                validation_split=0.):
  first_x_value = nest.flatten(inputs)[0]
  if isinstance(first_x_value, np.ndarray):
    num_samples = first_x_value.shape[0]
    if validation_split and 0. < validation_split < 1.:
      num_samples = int(num_samples * (1 - validation_split))
    steps_per_epoch, batch_size = get_input_params(
        strategy, num_samples, steps_per_epoch, batch_size, mode=mode)
  return batch_size, steps_per_epoch
def get_input_params(distribution_strategy,
                     num_samples,
                     steps,
                     batch_size,
                     mode=None):
  use_per_replica_batch = not dist_utils.global_batch_size_supported(
      distribution_strategy)
  if context.executing_eagerly():
    allow_partial_batch = (
        mode != ModeKeys.TRAIN or
        not backend.is_tpu_strategy(distribution_strategy))
  else:
    allow_partial_batch = (
        mode == ModeKeys.TRAIN or
        ((mode == ModeKeys.PREDICT or mode == ModeKeys.TEST) and
         backend.is_tpu_strategy(distribution_strategy)))
  if steps is None:
    if batch_size is None:
      global_batch_size = min(num_samples, 32)
    else:
      global_batch_size = batch_size
      if use_per_replica_batch:
        global_batch_size *= distribution_strategy.num_replicas_in_sync
    if allow_partial_batch:
      steps = np.ceil(num_samples / global_batch_size).astype(int)
    else:
      if num_samples % global_batch_size:
        raise ValueError('The number of samples %s is not divisible by '
                         'batch size %s.' % (num_samples, global_batch_size))
      steps = num_samples // global_batch_size
  else:
    if batch_size is None:
      if num_samples % steps:
        raise ValueError('The number of samples %s is not divisible by '
                         'steps %s. Please change the number of steps to a '
                         'value that can consume all the samples' % (
                             num_samples, steps))
      global_batch_size = num_samples // steps
    else:
      global_batch_size = batch_size
      if use_per_replica_batch:
        global_batch_size *= distribution_strategy.num_replicas_in_sync
      min_num_samples = global_batch_size * steps
      if allow_partial_batch:
        min_num_samples = global_batch_size * (steps-1) + 1 if steps > 1 else 0
      if num_samples < min_num_samples:
        raise ValueError('Number of samples %s is less than samples required '
                         'for specified batch_size %s and steps %s' % (
                             num_samples, global_batch_size, steps))
  if use_per_replica_batch:
    if global_batch_size % distribution_strategy.num_replicas_in_sync:
      raise ValueError(
          'The batch size (%s) could not be sharded evenly across the sync '
          'replicas (%s) in the distribution strategy.' % (
              global_batch_size, distribution_strategy.num_replicas_in_sync))
    batch_size = global_batch_size // distribution_strategy.num_replicas_in_sync
  else:
    batch_size = global_batch_size
  return steps, batch_size
def get_batch_dimension(iterator):
  shapes = nest.flatten(dataset_ops.get_legacy_output_shapes(iterator))
  dims = shapes[0].dims
  return dims[0] if dims else None
def get_iterator(dataset, distribution_strategy):
  with distribution_strategy.scope():
    iterator = distribution_strategy.make_dataset_iterator(dataset)
  initialize_iterator(iterator, distribution_strategy)
  return iterator
def initialize_iterator(iterator, distribution_strategy):
  with distribution_strategy.scope():
    init_op = control_flow_ops.group(iterator.initializer)
    if not context.executing_eagerly():
      backend.get_session((init_op,)).run(init_op)
def _get_input_from_iterator(iterator, model):
  next_element = iterator.get_next()
  if len(nest.flatten(next_element)) == len(model.inputs):
    x = next_element
    y = None
    sample_weights = None
  elif len(nest.flatten(next_element)) == (len(model.inputs) +
                                           len(model.outputs)):
    x, y = next_element
    sample_weights = None
  else:
    x, y, sample_weights = next_element
  validate_distributed_dataset_inputs(
      model._distribution_strategy, x, y, sample_weights)
  return x, y, sample_weights
def _prepare_feed_values(model, inputs, targets, sample_weights, mode):
  strategy = model._distribution_strategy
  inputs, targets, sample_weights = _get_input_from_iterator(inputs, model)
  if backend.is_tpu_strategy(strategy):
    if sample_weights is not None:
      raise ValueError('TPUStrategy does not support sample weights.')
  if isinstance(inputs, dict):
    inputs = [inputs[key] for key in model._feed_input_names]
  if is_distributing_by_cloning(model):
    inputs = flatten_per_replica_values(strategy, inputs)
    targets = flatten_per_replica_values(strategy, targets)
    inputs, targets = nest.map_structure(
        training_utils_v1.standardize_single_array, (inputs, targets))
  else:
    inputs = training_utils_v1.ModelInputs(inputs).as_list()
  if mode == ModeKeys.PREDICT:
    sample_weights = []
    targets = []
  elif sample_weights is not None and is_distributing_by_cloning(model):
    if context.executing_eagerly() and not model._compile_distribution:
      raise NotImplementedError('`sample_weight` is not supported when using '
                                'tf.distribute.Strategy in eager mode and '
                                'cloning=True.')
    sample_weights = flatten_per_replica_values(strategy, sample_weights)
  ins = [inputs, targets, sample_weights]
  return tuple(ins)
def is_distributing_by_cloning(model):
  if (backend.is_tpu_strategy(model._distribution_strategy) and
    return False
  elif ops.executing_eagerly_outside_functions():
    return bool(model._compile_distribution)
  return True
def _custom_compile_for_predict(model):
  if not model.built:
    return
  model._is_compiled = True
  model.total_loss = None
  model.train_function = None
  model.test_function = None
  model.predict_function = None
def _build_network_on_replica(model, mode, inputs=None, targets=None):
  """Build an updated model on replicas.
  We create a new Keras model while sharing the variables from the old graph.
  Building a new sub-graph is required since the original keras model creates
  placeholders for the input and the output that are not accessible till we
  call iterator.get_next() inside the step_fn for `fit`/`evaluate`/`predict`.
  The sharing of weights and layers between the old and the new model guarantee
  that we're using Strategy variables and any updates on either model are
  reflected correctly in callbacks and loop iterations.
  We need to make sure we share the optimizers between the old and the new model
  as well so that optimizer state is not lost if the user is running fit
  multiple times.
  Args:
    model: Model to be replicated across Replicas
    mode: Which of fit/eval/predict is building the distributed network
    inputs: Input variables to be passed to the model
    targets: Target tensor to be passed to model.compile
  Returns:
    A new model with shared layers with the old model.
  """
  if isinstance(model, sequential.Sequential):
    updated_model = models._clone_sequential_model(
        model, input_tensors=inputs, layer_fn=models.share_weights)
  else:
    updated_model = models._clone_functional_model(
        model, input_tensors=inputs, layer_fn=models.share_weights)
    updated_model._callable_losses = model._callable_losses
  def _upcast_low_precision_outputs(output):
    if output.dtype == dtypes.bfloat16:
      return math_ops.cast(output, dtypes.float32)
    else:
      return output
  updated_model.outputs = [_upcast_low_precision_outputs(o)
                           for o in updated_model.outputs]
  if isinstance(targets, tuple):
    targets = nest.flatten(targets)
    _custom_compile_for_predict(updated_model)
  else:
    updated_model.compile(
        model.optimizer,
        model.loss,
        metrics=metrics_module.clone_metrics(model._compile_metrics),
        loss_weights=model.loss_weights,
        sample_weight_mode=model.sample_weight_mode,
        weighted_metrics=metrics_module.clone_metrics(
            model._compile_weighted_metrics),
        target_tensors=targets)
  return updated_model
def _build_distributed_network(model, strategy, mode, inputs=None,
                               targets=None):
  with backend.get_graph().as_default(), strategy.scope():
    distributed_model = strategy.extended.call_for_each_replica(
        _build_network_on_replica,
        args=(model, mode, inputs, targets))
    set_distributed_model(model, mode, distributed_model)
def _clone_and_build_model(model, mode, inputs=None, targets=None):
  cloned_model = models.clone_model(model, input_tensors=inputs)
  if isinstance(model.optimizer, optimizers.TFOptimizer):
    optimizer = model.optimizer
  else:
    optimizer_config = model.optimizer.get_config()
    optimizer = model.optimizer.__class__.from_config(optimizer_config)
  def _upcast_low_precision_outputs(output):
    if output.dtype == dtypes.bfloat16:
      return math_ops.cast(output, dtypes.float32)
    else:
      return output
  cloned_model.outputs = [_upcast_low_precision_outputs(o)
                          for o in cloned_model.outputs]
  if isinstance(targets, tuple):
    targets = nest.flatten(targets)
    _custom_compile_for_predict(cloned_model)
  else:
    cloned_model.compile(
        optimizer,
        model.loss,
        metrics=metrics_module.clone_metrics(model._compile_metrics),
        loss_weights=model.loss_weights,
        sample_weight_mode=model.sample_weight_mode,
        weighted_metrics=metrics_module.clone_metrics(
            model._compile_weighted_metrics),
        target_tensors=targets)
  return cloned_model
def clone_model_on_replicas(model, strategy, mode, inputs=None, targets=None):
  with backend.get_graph().as_default(), strategy.scope():
    distributed_model = strategy.extended.call_for_each_replica(
        _clone_and_build_model, args=(model, mode, inputs, targets))
    set_distributed_model(model, mode, distributed_model)
  if mode == ModeKeys.TRAIN:
    model._make_callback_model(distributed_model)
def _make_execution_function(model, mode):
  if is_distributing_by_cloning(model):
    return _make_execution_function_with_cloning(model, mode)
  distributed_function = get_distributed_function(model, mode)
  if distributed_function:
    return distributed_function
  distribution_function = _make_execution_function_without_cloning(model, mode)
  set_distributed_function(model, mode, distribution_function)
  return distribution_function
def _make_execution_function_without_cloning(model, mode):
  strategy = model._distribution_strategy
  with strategy.scope():
    per_replica_function = _make_replica_execution_function(model, mode)
    def distributed_function(input_fn):
      x, y, sample_weights = input_fn()
      outputs = strategy.run(per_replica_function, args=(x, y, sample_weights))
      all_outputs = unwrap_outputs(
          strategy, outputs, with_loss_tensor=(mode != ModeKeys.PREDICT))
      return all_outputs
    if not model.run_eagerly:
      distributed_function = def_function.function(distributed_function)
      def execution_function(input_fn):
        return [out.numpy() for out in distributed_function(input_fn)]
    else:
      execution_function = distributed_function
    return execution_function
def _make_replica_execution_function(model, mode):
  if mode == ModeKeys.TRAIN:
    func = model.train_on_batch
  elif mode == ModeKeys.TEST:
    func = model.test_on_batch
  else:
    def predict_on_batch(x, y=None, sample_weights=None):
      del y, sample_weights
      return model.predict_on_batch(x)
    func = predict_on_batch
  if mode != ModeKeys.PREDICT:
    func = functools.partial(func, reset_metrics=False)
  return func
def _make_replicated_models_with_cloning(model, mode):
  strategy = model._distribution_strategy
  if model._compile_distribution:
    clone_model_on_replicas(model, strategy, mode)
  else:
    _build_distributed_network(model, strategy, mode)
def _make_execution_function_with_cloning(model, mode):
  distributed_model = get_distributed_model(model, mode)
  if (distributed_model and hasattr(distributed_model, '_distribution_function')
      and not (hasattr(distributed_model, '_recompile_exec_function') and
               distributed_model._recompile_exec_function)):
    return distributed_model._distributed_function
  if not distributed_model:
    _make_replicated_models_with_cloning(model, mode)
    distributed_model = get_distributed_model(model, mode)
  assert distributed_model
  if context.executing_eagerly():
    distributed_function = _make_eager_execution_function(model, mode)
  else:
    distributed_function = _make_graph_execution_function(model, mode)
  distributed_model._distributed_function = distributed_function
  distributed_model._recompile_exec_function = False
  return distributed_function
def _make_graph_execution_function(model, mode):
  def _per_replica_function(model):
    f = model._make_execution_function(mode)
    return (f.inputs, f.outputs, f.updates_op, f.session_kwargs)
  strategy = model._distribution_strategy
  with strategy.scope():
    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = strategy.extended.call_for_each_replica(
         _per_replica_function, args=(get_distributed_model(model, mode),))
    init_restore_or_wait_for_variables()
    (all_inputs, all_outputs, all_updates, all_session_args) = unwrap_values(
        strategy,
        grouped_inputs,
        grouped_outputs,
        grouped_updates,
        grouped_session_args,
        with_loss_tensor=(mode != ModeKeys.PREDICT))
    return backend.function(
        all_inputs,
        all_outputs,
        updates=all_updates,
        name='distributed_{}_function'.format(mode),
        **all_session_args)
def _make_eager_execution_function(model, mode):
  def _per_replica_function(model):
    f = model._make_execution_function(mode)
    return (f.inputs, f.outputs)
  strategy = model._distribution_strategy
  global_graph = backend.get_graph()
  with global_graph.as_default(), strategy.scope():
    with backend._scratch_graph(global_graph):
      grouped = strategy.extended.call_for_each_replica(
          _per_replica_function, args=(get_distributed_model(model, mode),))
      grouped_inputs, grouped_outputs = grouped
      (all_inputs, all_outputs, _, _) = unwrap_values(
          strategy,
          grouped_inputs,
          grouped_outputs,
          with_loss_tensor=(mode != ModeKeys.PREDICT))
    return backend.function(
        all_inputs,
        all_outputs,
        name='eager_distributed_{}_function'.format(mode))
def _copy_weights_to_distributed_model(original_model, mode):
  strategy = original_model._distribution_strategy
  distributed_model = get_distributed_model(original_model, mode)
  if strategy:
    orig_model_weights = original_model.get_weights()
    first_model = strategy.unwrap(distributed_model)[0]
    set_weights(strategy, first_model, orig_model_weights)
def _copy_weights_to_original_model(model, mode):
  if model._distribution_strategy and mode == ModeKeys.TRAIN:
    distributed_model = get_distributed_model(model, mode)
    updated_weights = model._distribution_strategy.unwrap(
        distributed_model)[0].get_weights()
    model.set_weights(updated_weights)
def _per_replica_aggregate_batch(strategy, batch_outs, model, mode):
  if strategy is not None and mode == ModeKeys.PREDICT:
    total_batch_outs = []
    for i in range(len(model.outputs)):
      num_replicas = strategy.num_replicas_in_sync
      nested_outs = batch_outs[i * num_replicas:i * num_replicas + num_replicas]
      total_batch_outs.append(
          concat_along_batch_dimension(nest.flatten(nested_outs)))
    return total_batch_outs
  return batch_outs
def _reset_metrics(model):
  if model._distribution_strategy:
    for mode in [ModeKeys.TRAIN, ModeKeys.TEST, ModeKeys.PREDICT]:
      distributed_model = get_distributed_model(model, mode)
      if distributed_model:
        first_model = model._distribution_strategy.unwrap(distributed_model)[0]
        first_model.reset_metrics()
def get_distributed_model(model, mode):
  key = _generate_cache_key(mode)
  return model._distributed_model_cache.get(key, None)
def set_distributed_model(model, mode, distributed_model):
  key = _generate_cache_key(mode)
  model._distributed_model_cache[key] = distributed_model
def get_distributed_function(model, mode):
  key = _generate_cache_key(mode)
  return model._distributed_function_cache.get(key, None)
def set_distributed_function(model, mode, distributed_function):
  key = _generate_cache_key(mode)
  model._distributed_function_cache[key] = distributed_function
def _generate_cache_key(mode):
  key = hash(mode)
  return key
@tf_contextlib.contextmanager
def distributed_scope(strategy, learning_phase):
  with strategy.scope(), backend.learning_phase_scope(learning_phase):
    yield
def is_current_worker_chief():
  return dc.get_current_worker_context().is_chief
def filter_distributed_callbacks(callbacks_list, model):
  if not model._in_multi_worker_mode():
    raise ValueError(
        'filter_distributed_callbacks() should only be called when Keras '
        'is in multi worker mode.')
  callbacks_list = callbacks_list or []
  if not [
      c for c in callbacks_list if isinstance(c, callbacks.ModelCheckpoint)
  ]:
    logging.warning('ModelCheckpoint callback is not provided. '
                    'Workers will need to restart training if any fails.')
  if callbacks_list is None or is_current_worker_chief():
    return callbacks_list
  return [
      callback for callback in callbacks_list if not callback._chief_worker_only
def _update_sample_weight_modes(model, mode, sample_weights):
  if is_distributing_by_cloning(model):
    distributed_model = get_distributed_model(model, mode)
    if not distributed_model:
      _make_replicated_models_with_cloning(model, mode)
      distributed_model = get_distributed_model(model, mode)
    distributed_model._recompile_exec_function = any(
        [e.sample_weights_mismatch() for e in model._training_endpoints])
    if sample_weights:
      distributed_models = flatten_per_replica_values(
          model._distribution_strategy, distributed_model)
      sample_weights = sample_weights[0]
      if sample_weights and None not in sample_weights:
        for m, sw in zip(distributed_models, sample_weights):
          m._update_sample_weight_modes(sample_weights=[sw])
def concat_along_batch_dimension(outputs):
  if isinstance(outputs[0], sparse_tensor.SparseTensor):
    return sparse_ops.sparse_concat_v2(axis=0, sp_inputs=outputs)
  if isinstance(outputs[0], ragged_tensor.RaggedTensor):
    return array_ops.concat(outputs, axis=0)
  return np.concatenate(outputs)
