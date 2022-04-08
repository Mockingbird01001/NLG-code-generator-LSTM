
import abc
import six
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@six.add_metaclass(abc.ABCMeta)
@deprecation.deprecated_endpoints('mixed_precision.experimental.LossScale',
                                  'train.experimental.LossScale')
@tf_export(
    v1=[
        'mixed_precision.LossScale',
        'mixed_precision.experimental.LossScale',
        'train.experimental.LossScale'
    ])
class LossScale(trackable.Trackable):
  """Base class for all TF1 loss scales.
  This is an abstract base class, so you cannot instantiate it directly.
  Instead, use one of its concrete subclasses:
    * `tf.compat.v1.mixed_precision.DynamicLossScale`
    * `tf.compat.v1.mixed_precision.FixedLossScale`
  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:
  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```
  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used for mixed
  precision training. By multiplying the loss, each intermediate gradient will
  have the same multiplier applied.
  Instances of this class represent a loss scale. Calling instances of this
  class returns the loss scale as a scalar float32 tensor, while method
  `update()` updates the loss scale depending on the values of the gradients.
  Optimizers use instances of this class to scale loss and gradients.
  In most functions that accept a LossScale, you can also pass an int (such as
  8) to create a `FixedLossScale` or the string `"dynamic"` to create a dynamic
  loss scale.
  """
  def __init__(self):
    self._weights = {}
  @abc.abstractmethod
  def __call__(self):
    pass
  @abc.abstractmethod
  def update(self, grads):
    pass
  def _add_weight(self, name, initial_value, dtype=None):
    variable = variable_scope.variable(
        initial_value=initial_value,
        name=name,
        dtype=dtype,
        trainable=False,
        use_resource=True,
        synchronization=variables.VariableSynchronization.AUTO,
        aggregation=variables.VariableAggregation.NONE)
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
    key = (name, graph_key)
    if self._weights.get(key, None) is not None:
      raise RuntimeError('Duplicate variables detected. {}'.format(key))
    self._weights[key] = variable
    self._handle_deferred_dependencies(name=name, trackable=variable)
    return variable
  def _trackable_children(self,
                          save_type=trackable.SaveType.CHECKPOINT,
                          **kwargs):
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
    weights = {}
    for (name, g), v in sorted(self._weights.items(), key=lambda i: i[0][0]):
      if g == graph_key:
        weights[name] = v
    weights.update(
        super(LossScale, self)._trackable_children(save_type, **kwargs))
    return weights
  def _lookup_dependency(self, name):
    unconditional = super(LossScale, self)._lookup_dependency(name)
    if unconditional is not None:
      return unconditional
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
    return self._weights.get((name, graph_key), None)
  @abc.abstractmethod
  def get_config(self):
    pass
  @classmethod
  def from_config(cls, config):
    return cls(**config)
@deprecation.deprecated_endpoints('mixed_precision.experimental.FixedLossScale',
                                  'train.experimental.FixedLossScale')
@tf_export(
    v1=[
        'mixed_precision.FixedLossScale',
        'mixed_precision.experimental.FixedLossScale',
        'train.experimental.FixedLossScale'
    ])
class FixedLossScale(LossScale):
  @deprecation.deprecated(
      None, 'Use tf.keras.mixed_precision.LossScaleOptimizer instead. '
            'LossScaleOptimizer now has all the functionality of '
            'FixedLossScale')
  def __init__(self, loss_scale_value):
    super(FixedLossScale, self).__init__()
    if not isinstance(loss_scale_value, six.integer_types + (float,)):
      raise ValueError('loss_scale_value must be a Python int or float.')
    if loss_scale_value < 1:
      raise ValueError('loss_scale_value must be at least 1.')
    self._loss_scale_value = float(loss_scale_value)
  def __call__(self):
    return ops.convert_to_tensor(self._loss_scale_value)
  def update(self, grads):
    del grads
    return control_flow_ops.no_op(), True
  def __repr__(self):
    return 'FixedLossScale(%s)' % self._loss_scale_value
  def get_config(self):
    return {'loss_scale_value': self._loss_scale_value}
def _is_all_finite(grads):
  is_finite_per_grad = [
      math_ops.reduce_all(math_ops.is_finite(g)) for g in grads if g is not None
  ]
  return math_ops.reduce_all(is_finite_per_grad)
def _op_in_graph_mode(tensor):
  if context.executing_eagerly():
    return tensor
  return tensor.op
def _assign_if_finite(var, value):
  return control_flow_ops.cond(
      math_ops.is_finite(value), lambda: _op_in_graph_mode(var.assign(value)),
      control_flow_ops.no_op)
@deprecation.deprecated_endpoints(
    'mixed_precision.experimental.DynamicLossScale',
    'train.experimental.DynamicLossScale')
@tf_export(
    v1=[
        'mixed_precision.DynamicLossScale',
        'mixed_precision.experimental.DynamicLossScale',
        'train.experimental.DynamicLossScale'
    ])
class DynamicLossScale(LossScale):
  @deprecation.deprecated(
      None, 'Use tf.keras.mixed_precision.LossScaleOptimizer instead. '
            'LossScaleOptimizer now has all the functionality of '
            'DynamicLossScale')
  def __init__(self,
               increment_period=2000,
               multiplier=2.):
    super(DynamicLossScale, self).__init__()
    self._initial_loss_scale = float(initial_loss_scale)
    self._increment_period = int(increment_period)
    self._multiplier = float(multiplier)
    self._current_loss_scale = self._add_weight(
        name='current_loss_scale',
        dtype=dtypes.float32,
        initial_value=self._initial_loss_scale)
    self._num_good_steps = self._add_weight(
        name='good_steps', dtype=dtypes.int64, initial_value=0)
  @property
  def initial_loss_scale(self):
    return self._initial_loss_scale
  @property
  def increment_period(self):
    return self._increment_period
  @property
  def multiplier(self):
    return self._multiplier
  def __call__(self):
    return ops.convert_to_tensor(self._current_loss_scale)
  def update(self, grads):
    grads = nest.flatten(grads)
    if distribution_strategy_context.has_strategy():
      distribution = distribution_strategy_context.get_cross_replica_context()
      def get_is_finite(grads):
        is_finite = _is_all_finite(grads)
        return math_ops.cast(is_finite, dtypes.float32)
      is_finite_float = distribution.extended.call_for_each_replica(
          get_is_finite, args=(grads,))
      reduced_is_finite_float = distribution.reduce(reduce_util.ReduceOp.SUM,
                                                    is_finite_float, axis=None)
      is_finite = math_ops.equal(reduced_is_finite_float,
                                 distribution.num_replicas_in_sync)
    else:
      is_finite = _is_all_finite(grads)
    def update_if_finite_grads():
      def incr_loss_scale():
        new_loss_scale = self._current_loss_scale * self._multiplier
        return control_flow_ops.group(
            _assign_if_finite(self._current_loss_scale, new_loss_scale),
            self._num_good_steps.assign(0))
      return control_flow_ops.cond(
          self._num_good_steps + 1 >= self._increment_period,
          incr_loss_scale, lambda: _op_in_graph_mode(
              self._num_good_steps.assign_add(1)))
    def update_if_not_finite_grads():
      new_loss_scale = math_ops.maximum(
          self._current_loss_scale / self._multiplier, 1)
      return control_flow_ops.group(
          self._num_good_steps.assign(0),
          self._current_loss_scale.assign(new_loss_scale))
    update_op = control_flow_ops.cond(is_finite, update_if_finite_grads,
                                      update_if_not_finite_grads)
    should_apply_gradients = is_finite
    return update_op, should_apply_gradients
  def __repr__(self):
    if context.executing_eagerly():
      return ('DynamicLossScale(current_loss_scale=%s, num_good_steps=%s, '
              'initial_loss_scale=%s, increment_period=%s, multiplier=%s)' %
              (self._current_loss_scale.numpy(), self._num_good_steps.numpy(),
               self.initial_loss_scale, self.increment_period, self.multiplier))
    else:
      return ('DynamicLossScale(initial_loss_scale=%s, increment_period=%s, '
              'multiplier=%s)' %
              (self.initial_loss_scale, self.increment_period, self.multiplier))
  def get_config(self):
    return {
        'initial_loss_scale': self.initial_loss_scale,
        'increment_period': self.increment_period,
        'multiplier': self.multiplier,
    }
def get(identifier):
  if isinstance(identifier, six.integer_types + (float,)):
    return FixedLossScale(identifier)
  if identifier == 'dynamic':
    return DynamicLossScale()
  if isinstance(identifier, LossScale):
    return identifier
  elif identifier is None:
    return None
  else:
    raise ValueError('Could not interpret loss scale identifier: %s' %
                     identifier)
