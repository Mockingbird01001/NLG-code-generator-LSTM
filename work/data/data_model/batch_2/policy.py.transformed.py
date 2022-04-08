
import contextlib
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.mixed_precision import device_compatibility_check
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.mixed_precision.Policy', v1=[])
class Policy(object):
  """A dtype policy for a Keras layer.
  A dtype policy determines a layer's computation and variable dtypes. Each
  layer has a policy. Policies can be passed to the `dtype` argument of layer
  constructors, or a global policy can be set with
  `tf.keras.mixed_precision.set_global_policy`.
  Args:
    name: The policy name, which determines the compute and variable dtypes. Can
      be any dtype name, such as `'float32'` or `'float64'`, which causes both
      the compute and variable dtypes will be that dtype. Can also be the string
      `'mixed_float16'` or `'mixed_bfloat16'`, which causes the compute dtype to
      be float16 or bfloat16 and the variable dtype to be float32.
  Typically you only need to interact with dtype policies when using mixed
  precision, which is the use of float16 or bfloat16 for computations and
  float32 for variables. This is why the term `mixed_precision` appears in the
  API name. Mixed precision can be enabled by passing `'mixed_float16'` or
  `'mixed_bfloat16'` to `tf.keras.mixed_precision.set_global_policy`. See [the
  mixed precision guide](https://www.tensorflow.org/guide/keras/mixed_precision)
  for more information on how to use mixed precision.
  >>> tf.keras.mixed_precision.set_global_policy('mixed_float16')
  >>> layer1 = tf.keras.layers.Dense(10)
  <Policy "mixed_float16">
  >>> layer2 = tf.keras.layers.Dense(10, dtype='float32')
  >>> layer2.dtype_policy
  <Policy "float32">
  >>> tf.keras.mixed_precision.set_global_policy('float32')
  In the example above, passing `dtype='float32'` to the layer is equivalent to
  passing `dtype=tf.keras.mixed_precision.Policy('float32')`. In general,
  passing a dtype policy name to a layer is equivalent to passing the
  corresponding policy, so it is never necessary to explicitly construct a
  `Policy` object.
  Note: `Model.compile` will automatically wrap an optimizer with a
  `tf.keras.mixed_precision.LossScaleOptimizer` if you use the `'mixed_float16'`
  policy. If you use a custom training loop instead of calling `Model.compile`,
  you should explicitly use a `tf.keras.mixed_precision.LossScaleOptimizer` to
  avoid numeric underflow with float16.
  A layer casts its inputs to its compute dtype. This causes the layer's
  computations and output to also be in the compute dtype. For example:
  >>> x = tf.ones((4, 4, 4, 4), dtype='float64')
  >>> layer = tf.keras.layers.Conv2D(filters=4, kernel_size=2)
  'float32'
  >>> y = layer(x)
  >>> y.dtype
  tf.float32
  Note that the base `tf.keras.layers.Layer` class inserts the casts. If
  subclassing your own layer, you do not have to insert any casts.
  Currently, only tensors in the first argument to the layer's `call` method are
  casted (although this will likely be changed in a future minor release). For
  example:
  >>> class MyLayer(tf.keras.layers.Layer):
  ...   def call(self, a, b):
  ...     return a + 1., b + 1.
  >>> a = tf.constant(1., dtype="float32")
  >>> b = tf.constant(1., dtype="float32")
  >>> layer = MyLayer(dtype="float64")
  >>> x, y = layer(a, b)
  >>> x.dtype
  tf.float64
  >>> y.dtype
  tf.float32
  If writing your own layer with multiple inputs, you should either explicitly
  cast other tensors to `self.compute_dtype` in `call` or accept all tensors in
  the first argument as a list.
  The casting only occurs in TensorFlow 2. If
  `tf.compat.v1.disable_v2_behavior()` has been called, you can enable the
  casting behavior with `tf.compat.v1.keras.layers.enable_v2_dtype_behavior()`.
  The default dtype of variables created by `tf.keras.layers.Layer.add_weight`
  is the layer's policy's variable dtype.
  If a layer's compute and variable dtypes differ, `add_weight` will wrap
  floating-point variables with a special wrapper called an `AutoCastVariable`.
  `AutoCastVariable` is identical to the original variable except it casts
  itself to the layer's compute dtype when used within `Layer.call`. This means
  if you are writing a layer, you do not have to explicitly cast the variables
  to the layer's compute dtype. For example:
  >>> class SimpleDense(tf.keras.layers.Layer):
  ...
  ...   def build(self, input_shape):
  ...     self.kernel = self.add_weight('kernel', (input_shape[-1], 10))
  ...
  ...   def call(self, inputs):
  ...     return tf.linalg.matmul(inputs, self.kernel)
  ...
  >>> layer = SimpleDense(dtype='mixed_float16')
  >>> y = layer(tf.ones((10, 10)))
  >>> y.dtype
  tf.float16
  >>> layer.kernel.dtype
  tf.float32
  A layer author can prevent a variable from being wrapped with an
  `AutoCastVariable` by passing `experimental_autocast=False` to `add_weight`,
  which is useful if the float32 value of the variable must be accessed within
  the layer.
  For the most part, layers will automatically support mixed precision and
  float64 without any additional work, due to the fact the base layer
  automatically casts inputs, creates variables of the correct type, and in the
  case of mixed precision, wraps variables with `AutoCastVariables`.
  The primary case where you need extra work to support mixed precision or
  float64 is when you create a new tensor, such as with `tf.ones` or
  `tf.random.normal`, In such cases, you must create the tensor of the correct
  dtype. For example, if you call `tf.random.normal`, you must pass the compute
  dtype, which is the dtype the inputs have been casted to:
  >>> class AddRandom(tf.keras.layers.Layer):
  ...
  ...   def call(self, inputs):
  ...     rand = tf.random.normal(shape=inputs.shape, dtype=inputs.dtype)
  ...     return inputs + rand
  >>> layer = AddRandom(dtype='mixed_float16')
  >>> y = layer(x)
  >>> y.dtype
  tf.float16
  If you did not pass `dtype=inputs.dtype` to `tf.random.normal`, a
  `TypeError` would have occurred. This is because the `tf.random.normal`'s
  dtype defaults to `"float32"`, but the input dtype is float16. You cannot add
  a float32 tensor with a float16 tensor.
  """
  def __init__(self, name):
    if isinstance(name, dtypes.DType):
      raise TypeError("'name' must be a string, not a DType. "
                      "Instead, pass DType.name. Got: %s" % (name.name,))
    elif not isinstance(name, str):
      raise TypeError("'name' must be a string, but got: %s" % (name,))
    self._name = name
    self._compute_dtype, self._variable_dtype = self._parse_name(name)
    if name in ('mixed_float16', 'mixed_bloat16'):
      device_compatibility_check.log_device_compatibility_check(name)
  def _parse_name(self, name):
    """Parses a Policy name into a compute and variable dtype.
    Args:
      name: The name of the policy:
    Returns:
      The (compute_dtype, variable_dtype) pair.
    """
    if name.endswith('_float32_vars'):
      error_msg = ('Policies ending in \'_float32_vars\' have been removed '
                   'from TensorFlow.')
      if name in ('infer_float32_vars', 'infer_with_float32_vars'):
        error_msg += (' Please use the \'mixed_float16\' or \'mixed_bfloat16\' '
                      'policy instead.')
      elif name == 'float16_with_float32_vars':
        error_msg += (' Please use the \'mixed_float16\' policy instead.')
      elif name == 'bfloat16_with_float32_vars':
        error_msg += (' Please use the \'mixed_bfloat16\' policy instead.')
      error_msg += ' Got policy name: \'%s\'' % name
      raise ValueError(error_msg)
    if name == 'mixed_float16':
      return 'float16', 'float32'
    elif name == 'mixed_bfloat16':
      return 'bfloat16', 'float32'
    elif name == '_infer':
      return None, None
    try:
      dtype = dtypes.as_dtype(name).name
    except TypeError:
      error = ("Cannot convert value %s to a mixed precision Policy. "
               "Valid policies include 'mixed_float16', 'mixed_bfloat16', "
               "and the name of any dtype such as 'float32'." % (name,))
      raise ValueError(error)
    return dtype, dtype
  @property
  def variable_dtype(self):
    return self._variable_dtype
  @property
  def compute_dtype(self):
    return self._compute_dtype
  @property
  def name(self):
    return self._name
  def __repr__(self):
    return '<Policy "%s">' % self._name
  def get_config(self):
    return {'name': self.name}
  @classmethod
  def from_config(cls, config, custom_objects=None):
    del custom_objects
    if 'loss_scale' in config:
      config = config.copy()
      del config['loss_scale']
    return cls(**config)
@keras_export('keras.mixed_precision.experimental.Policy', v1=[])
class PolicyV1(Policy):
  def __init__(self, name, loss_scale='auto'):
    """Constructs the policy.
    The `name` argument determines the compute and variable dtype, the default
    loss scale, and has no additional effect on the Policy. The compute and
    variable dtypes can only be specified through `name`, and cannot be
    specified directly.
    Args:
      name: A string. Can be one of the following values:
        * Any dtype name, such as 'float32' or 'float64'. Both the variable and
          compute dtypes will be that dtype.
        * 'mixed_float16' or 'mixed_bfloat16': The compute dtype is float16 or
          bfloat16, while the variable dtype is float32. With 'mixed_float16',
          a dynamic loss scale is used. These policies are used for mixed
          precision training.
      loss_scale: A `tf.compat.v1.mixed_precision.LossScale`, an int (which
        uses a `FixedLossScale`), the string "dynamic" (which uses a
        `DynamicLossScale`), or None (which uses no loss scale). Defaults to
        `"auto"`. In the `"auto"` case: 1) if `name` is `"mixed_float16"`, then
        use `loss_scale="dynamic"`. 2) otherwise, do not use a loss scale. Only
        `tf.keras.Model`s, not layers, use the loss scale, and it is only used
        during `Model.fit`, `Model.train_on_batch`, and other similar methods.
    """
    super(PolicyV1, self).__init__(name)
    if loss_scale == 'auto':
      loss_scale = 'dynamic' if name == 'mixed_float16' else None
      self._using_default_loss_scale = True
    else:
      self._using_default_loss_scale = False
    if loss_scale and self._compute_dtype not in (None, 'float16'):
      tf_logging.warning(
          'Creating a Policy with a loss scale is only useful for '
          'float16 policies. You passed loss_scale=%r for policy '
          '%s. Consider not passing any loss_scale instead.' %
          (loss_scale, name))
    self._loss_scale = keras_loss_scale_module.get(loss_scale)
  @property
  def loss_scale(self):
    return self._loss_scale
  def __repr__(self):
    return '<PolicyV1 "%s", loss_scale=%s>' % (self._name, self.loss_scale)
  def get_config(self):
    config = {
        'name': self.name
    }
    if not self._using_default_loss_scale:
      config['loss_scale'] = keras_loss_scale_module.serialize(self.loss_scale)
    return config
  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'loss_scale' in config and isinstance(config['loss_scale'], dict):
      config = config.copy()
      config['loss_scale'] = keras_loss_scale_module.deserialize(
          config['loss_scale'], custom_objects=custom_objects)
    return cls(**config)
_global_policy = None
@keras_export('keras.mixed_precision.global_policy',
              'keras.mixed_precision.experimental.global_policy', v1=[])
def global_policy():
  """Returns the global dtype policy.
  The global policy is the default `tf.keras.mixed_precision.Policy` used for
  layers, if no policy is passed to the layer constructor. If no policy has been
  set with `keras.mixed_precision.set_global_policy`, this will return a policy
  constructed from `tf.keras.backend.floatx()` (floatx defaults to float32).
  >>> tf.keras.mixed_precision.global_policy()
  <Policy "float32">
  <Policy "float32">
  If TensorFlow 2 behavior has been disabled with
  `tf.compat.v1.disable_v2_behavior()`, this will instead return a special
  "_infer" policy which infers the dtype from the dtype of the first input the
  first time the layer is called. This behavior matches the behavior that
  existed in TensorFlow 1.
  See `tf.keras.mixed_precision.Policy` for more information on policies.
  Returns:
    The global Policy.
  """
  if _global_policy is None:
    if base_layer_utils.v2_dtype_behavior_enabled():
      return Policy(backend.floatx())
    else:
      return Policy('_infer')
  return _global_policy
def _check_if_mixed_precision_graph_rewrite_is_enabled(policy):
  if mixed_precision_global_state.is_mixed_precision_graph_rewrite_enabled():
    raise ValueError(
        'The global dtype policy cannot be set to "{policy.name}", because the '
        'mixed precision graph rewrite has already been enabled.\n'
        'At most, one of the following can be called:\n\n'
        '  1. tf.compat.v1.train.enable_mixed_precision_graph_rewrite() '
        '(You called this first)\n'
        '  2. tf.keras.mixed_precision.experimental.set_global_policy() with a '
        'mixed precision policy (You called this second)\n\n'
        'You called both functions, which is an error, because both functions '
        'enable you to use mixed precision. If in doubt which function to use, '
        'use the second, as it supports Eager execution and is more '
        'customizable.'.format(policy=policy))
@keras_export('keras.mixed_precision.set_global_policy',
              'keras.mixed_precision.experimental.set_global_policy', v1=[])
def set_global_policy(policy):
  """Sets the global dtype policy.
  The global policy is the default `tf.keras.mixed_precision.Policy` used for
  layers, if no policy is passed to the layer constructor.
  >>> tf.keras.mixed_precision.set_global_policy('mixed_float16')
  >>> tf.keras.mixed_precision.global_policy()
  <Policy "mixed_float16">
  >>> tf.keras.layers.Dense(10).dtype_policy
  <Policy "mixed_float16">
  >>> tf.keras.layers.Dense(10, dtype='float64').dtype_policy
  <Policy "float64">
  >>> tf.keras.mixed_precision.set_global_policy('float32')
  If no global policy is set, layers will instead default to a Policy
  constructed from `tf.keras.backend.floatx()`.
  To use mixed precision, the global policy should be set to `'mixed_float16'`
  or `'mixed_bfloat16'`, so that every layer uses a 16-bit compute dtype and
  float32 variable dtype by default.
  Only floating point policies can be set as the global policy, such as
  `'float32'` and `'mixed_float16'`. Non-floating point policies such as
  `'int32'` and `'complex64'` cannot be set as the global policy because most
  layers do not support such policies.
  See `tf.keras.mixed_precision.Policy` for more information.
  Args:
    policy: A Policy, or a string that will be converted to a Policy. Can also
      be None, in which case the global policy will be constructed from
      `tf.keras.backend.floatx()`
  """
  global _global_policy
  if not base_layer_utils.v2_dtype_behavior_enabled():
    raise ValueError('The global policy can only be set in TensorFlow 2 or if '
                     'V2 dtype behavior has been set. To enable V2 dtype '
                     'behavior, call '
                     '"tf.compat.v1.keras.layers.enable_v2_dtype_behavior()"')
  if policy is not None and not isinstance(policy, Policy):
    policy = Policy(policy)
  is_mixed_policy = (policy is not None and
                     policy.compute_dtype != policy.variable_dtype)
  if is_mixed_policy:
    _check_if_mixed_precision_graph_rewrite_is_enabled(policy)
  if (policy is not None and policy.compute_dtype is not None and
      not dtypes.as_dtype(policy.compute_dtype).is_floating):
    raise ValueError('set_global_policy can only be used to set the global '
                     'policy to floating-point policies, such as "float32" and '
                     '"mixed_float16", but got policy: %s'
                     % (policy.name,))
  _global_policy = policy
  mixed_precision_global_state.set_using_mixed_precision_policy(is_mixed_policy)
@contextlib.contextmanager
def policy_scope(policy):
  old_policy = _global_policy
  try:
    set_global_policy(policy)
    yield
  finally:
    set_global_policy(old_policy)
def _is_convertible_to_dtype(dtype):
  try:
    dtypes.as_dtype(dtype)
    return True
  except TypeError:
    return False
def _policy_equivalent_to_dtype(policy):
  """Returns True if the Policy is equivalent to a single dtype.
  A policy is equivalent to a single dtype if the policy's compute and variable
  dtypes are the same and the policy's type is Policy and not a subclass of
  Policy (such as PolicyV1).
  The "_infer" policy is considered equivalent to a single dtype.
  Args:
    policy: A Policy.
  Returns:
    True, if the policy is equivalent to a single dtype.
  """
          list(policy.get_config().keys()) == ['name'] and
          (policy.name == '_infer' or _is_convertible_to_dtype(policy.name)))
def serialize(policy):
  if _policy_equivalent_to_dtype(policy):
    return None if policy.name == '_infer' else policy.name
  return generic_utils.serialize_keras_object(policy)
def deserialize(config, custom_objects=None):
  if isinstance(config, str) and _is_convertible_to_dtype(config):
    return Policy(config)
  if config is None:
    return Policy('_infer')
  module_objects = {'Policy': Policy, 'PolicyV1': Policy}
  return generic_utils.deserialize_keras_object(
      config,
      module_objects=module_objects,
      custom_objects=custom_objects,
      printable_module_name='dtype policy')
