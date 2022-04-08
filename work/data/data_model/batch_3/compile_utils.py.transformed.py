
import copy
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
class Container(object):
  def __init__(self, output_names=None):
    self._output_names = output_names
  def build(self, y_pred):
    if self._output_names is None:
      self._output_names = create_pseudo_output_names(y_pred)
  def _conform_to_outputs(self, outputs, struct):
    """Convenience method to conform `struct` to `outputs` structure.
    Mappings performed:
    (1) Map a dict to a list of outputs, using the output names.
    (2) Fill missing keys in a dict w/ `None`s.
    (3) Map a single item to all outputs.
    Args:
      outputs: Model predictions.
      struct: Arbitrary nested structure (e.g. of labels, sample_weights,
        losses, or metrics).
    Returns:
      Mapping of `struct` to `outputs` structure.
    """
    struct = map_to_output_names(outputs, self._output_names, struct)
    struct = map_missing_dict_keys(outputs, struct)
    if not nest.is_nested(struct) and nest.is_nested(outputs):
      struct = nest.map_structure(lambda _: struct, outputs)
    return struct
  def _maybe_broadcast_to_outputs(self, outputs, objects):
    """Determines if losses / metrics should be applied to all outputs.
    NOTE: This method should only be called for Metrics / Losses, not for
    y_true / sample_weight.
    Args:
      outputs: Model predictions.
      objects: Arbitrary nested structure (e.g. of losses or metrics)
    Returns:
      Arbitrary nested structure of objects, maybe copied to each output.
    Applies a Loss / Metric to all outputs.
    """
    if not self._should_broadcast(objects):
      return objects
    should_copy_objects = len(nest.flatten(outputs)) > 1
    def _broadcast_fn():
      if should_copy_objects:
        return nest.map_structure(self._copy_object, objects)
      return objects
    return nest.map_structure(lambda _: _broadcast_fn(), outputs)
  def _should_broadcast(self, objects):
    raise NotImplementedError
  def _copy_object(self, obj):
    raise NotImplementedError
class LossesContainer(Container):
  def __init__(self, losses, loss_weights=None, output_names=None):
    super(LossesContainer, self).__init__(output_names=output_names)
    self._user_losses = losses
    self._user_loss_weights = loss_weights
    self._losses = losses
    self._loss_weights = loss_weights
    self._built = False
  @property
  def metrics(self):
    if not self._built:
      return []
    per_output_metrics = [
        metric_obj for metric_obj in nest.flatten(self._per_output_metrics)
        if metric_obj is not None
    ]
    return [self._loss_metric] + per_output_metrics
  def build(self, y_pred):
    super(LossesContainer, self).build(y_pred)
    self._losses = self._maybe_broadcast_to_outputs(y_pred, self._losses)
    self._losses = self._conform_to_outputs(y_pred, self._losses)
    self._losses = nest.map_structure(self._get_loss_object, self._losses)
    self._losses = nest.flatten(self._losses)
    self._loss_weights = self._maybe_broadcast_to_outputs(
        y_pred, self._loss_weights)
    self._loss_weights = self._conform_to_outputs(y_pred, self._loss_weights)
    self._loss_weights = nest.flatten(self._loss_weights)
    self._create_metrics()
    self._built = True
  @property
  def built(self):
    return self._built
  def _create_metrics(self):
    if len(self._output_names) == 1:
      self._per_output_metrics = [None]
    else:
      self._per_output_metrics = []
      for loss_obj, output_name in zip(self._losses, self._output_names):
        if loss_obj is None:
          self._per_output_metrics.append(None)
        else:
          self._per_output_metrics.append(
              metrics_mod.Mean(output_name + '_loss'))
  def __call__(self,
               y_true,
               y_pred,
               sample_weight=None,
               regularization_losses=None):
    """Computes the overall loss.
    Args:
      y_true: An arbitrary structure of Tensors representing the ground truth.
      y_pred: An arbitrary structure of Tensors representing a Model's outputs.
      sample_weight: An arbitrary structure of Tensors representing the
        per-sample loss weights. If one Tensor is passed, it is used for all
        losses. If multiple Tensors are passed, the structure should match
        `y_pred`.
      regularization_losses: Additional losses to be added to the total loss.
    Returns:
      Tuple of `(total_loss, per_output_loss_list)`
    """
    y_true = self._conform_to_outputs(y_pred, y_true)
    sample_weight = self._conform_to_outputs(y_pred, sample_weight)
    if not self._built:
      self.build(y_pred)
    y_pred = nest.flatten(y_pred)
    y_true = nest.flatten(y_true)
    sample_weight = nest.flatten(sample_weight)
    batch_dim = None
    zip_args = (y_true, y_pred, sample_weight, self._losses, self._loss_weights,
                self._per_output_metrics)
    for y_t, y_p, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
        continue
      y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
      sw = apply_mask(y_p, sw, get_mask(y_p))
      loss_value = loss_obj(y_t, y_p, sample_weight=sw)
      loss_metric_value = loss_value
      if loss_obj.reduction == losses_utils.ReductionV2.SUM:
        loss_metric_value *= ds_context.get_strategy().num_replicas_in_sync
      if batch_dim is None:
        if tf_utils.is_ragged(y_t):
          batch_dim = y_t.nrows()
        else:
          batch_dim = array_ops.shape(y_t)[0]
      if metric_obj is not None:
        metric_obj.update_state(loss_metric_value, sample_weight=batch_dim)
      if loss_weight is not None:
        loss_value *= loss_weight
        loss_metric_value *= loss_weight
      if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
          loss_obj.reduction == losses_utils.ReductionV2.AUTO):
        loss_value = losses_utils.scale_loss_for_distribution(loss_value)
      loss_values.append(loss_value)
      loss_metric_values.append(loss_metric_value)
    if regularization_losses:
      regularization_losses = losses_utils.cast_losses_to_common_dtype(
          regularization_losses)
      reg_loss = math_ops.add_n(regularization_losses)
      loss_metric_values.append(reg_loss)
      loss_values.append(losses_utils.scale_loss_for_distribution(reg_loss))
    if loss_values:
      loss_metric_values = losses_utils.cast_losses_to_common_dtype(
          loss_metric_values)
      total_loss_metric_value = math_ops.add_n(loss_metric_values)
      self._loss_metric.update_state(
          total_loss_metric_value, sample_weight=batch_dim)
      loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
      total_loss = math_ops.add_n(loss_values)
      return total_loss
    else:
      return array_ops.zeros(shape=())
  def reset_state(self):
    if not self._built:
      return
    metrics = [self._loss_metric] + nest.flatten(self._per_output_metrics)
    for metric_obj in metrics:
      if metric_obj is not None:
        metric_obj.reset_state()
  def _get_loss_object(self, loss):
    if loss is None:
    loss = losses_mod.get(loss)
    if not isinstance(loss, losses_mod.Loss):
      loss_name = get_custom_object_name(loss)
      if loss_name is None:
        raise ValueError('Loss should be a callable, found: {}'.format(loss))
      loss = losses_mod.LossFunctionWrapper(loss, name=loss_name)
    return loss
  def _should_broadcast(self, obj):
    return not nest.is_nested(obj)
  def _copy_object(self, obj):
class MetricsContainer(Container):
  def __init__(self, metrics=None, weighted_metrics=None, output_names=None,
               from_serialized=False):
    super(MetricsContainer, self).__init__(output_names=output_names)
    self._user_metrics = metrics
    self._user_weighted_metrics = weighted_metrics
    self._metrics = metrics
    self._weighted_metrics = weighted_metrics
    self._built = False
    self._from_serialized = from_serialized
  @property
  def metrics(self):
    if not self._built:
      return []
    return self._metrics_in_order
  @property
  def unweighted_metrics(self):
    if not self._built:
      return None
    return nest.flatten(self._metrics)
  @property
  def weighted_metrics(self):
    if not self._built:
      return None
    return nest.flatten(self._weighted_metrics)
  def build(self, y_pred, y_true):
    super(MetricsContainer, self).build(y_pred)
    self._metrics = self._maybe_broadcast_to_outputs(y_pred, self._metrics)
    self._metrics = self._conform_to_outputs(y_pred, self._metrics)
    self._weighted_metrics = self._maybe_broadcast_to_outputs(
        y_pred, self._weighted_metrics)
    self._weighted_metrics = self._conform_to_outputs(y_pred,
                                                      self._weighted_metrics)
    y_pred = nest.list_to_tuple(y_pred)
    y_true = nest.list_to_tuple(y_true)
    self._metrics = nest.list_to_tuple(self._metrics)
    self._weighted_metrics = nest.list_to_tuple(self._weighted_metrics)
    self._metrics = nest.map_structure_up_to(y_pred, self._get_metric_objects,
                                             self._metrics, y_true, y_pred)
    self._weighted_metrics = nest.map_structure_up_to(y_pred,
                                                      self._get_metric_objects,
                                                      self._weighted_metrics,
                                                      y_true, y_pred)
    self._metrics = nest.flatten_up_to(y_pred, self._metrics, check_types=False)
    self._weighted_metrics = nest.flatten_up_to(
        y_pred, self._weighted_metrics, check_types=False)
    if not self._from_serialized:
      self._set_metric_names()
    self._create_ordered_metrics()
    self._built = True
  @property
  def built(self):
    return self._built
  def _set_metric_names(self):
    metric_names = set()
    is_multi_output = len(self._output_names) > 1
    zip_args = (self._output_names, self._metrics, self._weighted_metrics)
    for output_name, output_metrics, weighted_output_metrics in zip(*zip_args):
      for m in output_metrics:
        if m is None:
          continue
        if is_multi_output:
          m._name = output_name + '_' + m._name
        if m._name in metric_names:
          raise ValueError('Found two metrics with the same name: {}'.format(
              m._name))
        metric_names.add(m._name)
      for wm in weighted_output_metrics:
        if wm is None:
          continue
        if is_multi_output:
          if output_name + '_' + wm._name in metric_names:
            wm._name = output_name + '_weighted_' + wm._name
          else:
            wm._name = output_name + '_' + wm._name
        elif wm._name in metric_names:
          wm._name = 'weighted_' + wm._name
        if wm._name in metric_names:
          raise ValueError('Found two metrics with the same name: {}'.format(
              wm._name))
        metric_names.add(wm._name)
  def _create_ordered_metrics(self):
    self._metrics_in_order = []
    for output_metrics, output_weighted_metrics in zip(self._metrics,
                                                       self._weighted_metrics):
      for m in nest.flatten(output_metrics):
        if m is not None:
          self._metrics_in_order.append(m)
      for wm in nest.flatten(output_weighted_metrics):
        if wm is not None:
          self._metrics_in_order.append(wm)
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = self._conform_to_outputs(y_pred, y_true)
    sample_weight = self._conform_to_outputs(y_pred, sample_weight)
    if not self._built:
      self.build(y_pred, y_true)
    y_pred = nest.flatten(y_pred)
    y_true = nest.flatten(y_true) if y_true is not None else []
    sample_weight = nest.flatten(sample_weight)
    zip_args = (y_true, y_pred, sample_weight, self._metrics,
                self._weighted_metrics)
    for y_t, y_p, sw, metric_objs, weighted_metric_objs in zip(*zip_args):
      if (y_t is None or (all(m is None for m in metric_objs) and
                          all(wm is None for wm in weighted_metric_objs))):
        continue
      y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
      mask = get_mask(y_p)
      sw = apply_mask(y_p, sw, mask)
      for metric_obj in metric_objs:
        if metric_obj is None:
          continue
        metric_obj.update_state(y_t, y_p, sample_weight=mask)
      for weighted_metric_obj in weighted_metric_objs:
        if weighted_metric_obj is None:
          continue
        weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw)
  def reset_state(self):
    if self._built:
      metrics = self._metrics_in_order
    else:
      metrics = nest.flatten(self._user_metrics) + nest.flatten(
          self._user_weighted_metrics)
    for metric_obj in metrics:
      if isinstance(metric_obj, metrics_mod.Metric):
        metric_obj.reset_state()
  def _get_metric_objects(self, metrics, y_t, y_p):
    metrics = nest.flatten(metrics)
    return [self._get_metric_object(m, y_t, y_p) for m in metrics]
  def _get_metric_object(self, metric, y_t, y_p):
    if metric is None:
    if str(metric).lower() not in ['accuracy', 'acc', 'crossentropy', 'ce']:
      metric_obj = metrics_mod.get(metric)
    else:
      y_t_rank = len(y_t.shape.as_list())
      y_p_rank = len(y_p.shape.as_list())
      y_t_last_dim = y_t.shape.as_list()[-1]
      y_p_last_dim = y_p.shape.as_list()[-1]
      is_binary = y_p_last_dim == 1
      is_sparse_categorical = (
          y_t_rank < y_p_rank or y_t_last_dim == 1 and y_p_last_dim > 1)
      if str(metric).lower() in ['accuracy', 'acc']:
        if is_binary:
          metric_obj = metrics_mod.binary_accuracy
        elif is_sparse_categorical:
          metric_obj = metrics_mod.sparse_categorical_accuracy
        else:
          metric_obj = metrics_mod.categorical_accuracy
      else:
        if is_binary:
          metric_obj = metrics_mod.binary_crossentropy
        elif is_sparse_categorical:
          metric_obj = metrics_mod.sparse_categorical_crossentropy
        else:
          metric_obj = metrics_mod.categorical_crossentropy
    if isinstance(metric_obj, losses_mod.Loss):
    if not isinstance(metric_obj, metrics_mod.Metric):
      if isinstance(metric, str):
        metric_name = metric
      else:
        metric_name = get_custom_object_name(metric)
        if metric_name is None:
          raise ValueError(
              'Metric should be a callable, found: {}'.format(metric))
      metric_obj = metrics_mod.MeanMetricWrapper(metric_obj, name=metric_name)
    return metric_obj
  def _should_broadcast(self, obj):
    if not nest.is_nested(obj):
      return True
    return (isinstance(obj, (list, tuple)) and
            not any(nest.is_nested(o) for o in obj))
  def _copy_object(self, obj):
    if isinstance(obj, metrics_mod.Metric):
      return obj.__class__.from_config(obj.get_config())
def create_pseudo_output_names(outputs):
  return _create_pseudo_names(outputs, prefix='output_')
def create_pseudo_input_names(inputs):
  return _create_pseudo_names(inputs, prefix='input_')
def _create_pseudo_names(tensors, prefix):
  def one_index(ele):
    if isinstance(ele, int):
      return ele + 1
    return ele
  flat_paths = list(nest.yield_flat_paths(tensors))
  flat_paths = nest.map_structure(one_index, flat_paths)
  names = []
  for path in flat_paths:
    if not path:
    else:
      name = '_'.join(str(p) for p in path)
      if isinstance(path[0], int):
        name = prefix + name
    names.append(name)
  return names
def map_to_output_names(y_pred, output_names, struct):
  """Maps a dict to a list using `output_names` as keys.
  This is a convenience feature only. When a `Model`'s outputs
  are a list, you can specify per-output losses and metrics as
  a dict, where the keys are the output names. If you specify
  per-output losses and metrics via the same structure as the
  `Model`'s outputs (recommended), no mapping is performed.
  For the Functional API, the output names are the names of the
  last layer of each output. For the Subclass API, the output names
  are determined by `create_pseudo_output_names` (For example:
  `['output_1', 'output_2']` for a list of outputs).
  This mapping preserves backwards compatibility for `compile` and
  `fit`.
  Args:
    y_pred: Sample outputs of the Model, to determine if this convenience
      feature should be applied (`struct` is returned unmodified if `y_pred`
      isn't a flat list).
    output_names: List. The names of the outputs of the Model.
    struct: The structure to map.
  Returns:
    `struct` mapped to a list in same order as `output_names`.
  """
  single_output = not nest.is_nested(y_pred)
  outputs_are_flat_list = (not single_output and
                           isinstance(y_pred, (list, tuple)) and
                           not any(nest.is_nested(y_p) for y_p in y_pred))
  if (single_output or outputs_are_flat_list) and isinstance(struct, dict):
    output_names = output_names or create_pseudo_output_names(y_pred)
    struct = copy.copy(struct)
    new_struct = [struct.pop(name, None) for name in output_names]
    if struct:
      raise ValueError('Found unexpected keys that do not correspond '
                       'to any Model output: {}. Expected: {}'.format(
                           struct.keys(), output_names))
    if len(new_struct) == 1:
      return new_struct[0]
    return new_struct
  else:
    return struct
def map_missing_dict_keys(y_pred, struct):
  if not isinstance(y_pred, dict) or not isinstance(struct, dict):
    return struct
  for k in y_pred.keys():
    if k not in struct:
      struct[k] = None
  return struct
def match_dtype_and_rank(y_t, y_p, sw):
  if y_t.shape.rank == 1 and y_p.shape.rank == 2:
    y_t = array_ops.expand_dims_v2(y_t, axis=-1)
  if sw is not None:
    if sw.shape.rank == 1 and y_p.shape.rank == 2:
      sw = array_ops.expand_dims_v2(sw, axis=-1)
  if ((y_t.dtype.is_floating and y_p.dtype.is_floating) or
      (y_t.dtype.is_integer and y_p.dtype.is_integer)):
    y_t = math_ops.cast(y_t, y_p.dtype)
  if sw is not None:
    sw = math_ops.cast(sw, y_p.dtype)
  return y_t, y_p, sw
def get_mask(y_p):
  return getattr(y_p, '_keras_mask', None)
def apply_mask(y_p, sw, mask):
  if mask is not None:
    mask = math_ops.cast(mask, y_p.dtype)
    if sw is not None:
      mask, _, sw = (
          losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=sw))
      sw *= mask
    else:
      sw = mask
  return sw
def get_custom_object_name(obj):
    return obj.name
    return obj.__name__
    return generic_utils.to_snake_case(obj.__class__.__name__)
    return None
