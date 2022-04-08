
import math
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export
def _check_penalty_number(x):
  if not isinstance(x, (float, int)):
    raise ValueError(('Value: {} is not a valid regularization penalty number, '
                      'expected an int or float value').format(x))
  if math.isinf(x) or math.isnan(x):
    raise ValueError(
        ('Value: {} is not a valid regularization penalty number, '
         'a positive/negative infinity or NaN is not a property value'
        ).format(x))
def _none_to_default(inputs, default):
  return default if inputs is None else default
@keras_export('keras.regularizers.Regularizer')
class Regularizer(object):
  """Regularizer base class.
  Regularizers allow you to apply penalties on layer parameters or layer
  activity during optimization. These penalties are summed into the loss
  function that the network optimizes.
  Regularization penalties are applied on a per-layer basis. The exact API will
  depend on the layer, but many layers (e.g. `Dense`, `Conv1D`, `Conv2D` and
  `Conv3D`) have a unified API.
  These layers expose 3 keyword arguments:
  - `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
  - `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
  - `activity_regularizer`: Regularizer to apply a penalty on the layer's output
  All layers (including custom layers) expose `activity_regularizer` as a
  settable property, whether or not it is in the constructor arguments.
  The value returned by the `activity_regularizer` is divided by the input
  batch size so that the relative weighting between the weight regularizers and
  the activity regularizers does not change with the batch size.
  You can access a layer's regularization penalties by calling `layer.losses`
  after calling the layer on inputs.
  >>> layer = tf.keras.layers.Dense(
  ...     5, input_dim=5,
  ...     kernel_initializer='ones',
  ...     kernel_regularizer=tf.keras.regularizers.L1(0.01),
  ...     activity_regularizer=tf.keras.regularizers.L2(0.01))
  >>> tensor = tf.ones(shape=(5, 5)) * 2.0
  >>> out = layer(tensor)
  >>> tf.math.reduce_sum(layer.losses)
  <tf.Tensor: shape=(), dtype=float32, numpy=5.25>
  ```python
  ```
  Compute a regularization loss on a tensor by directly calling a regularizer
  as if it is a one-argument function.
  E.g.
  >>> regularizer = tf.keras.regularizers.L2(2.)
  >>> tensor = tf.ones(shape=(5, 5))
  >>> regularizer(tensor)
  <tf.Tensor: shape=(), dtype=float32, numpy=50.0>
  Any function that takes in a weight matrix and returns a scalar
  tensor can be used as a regularizer, e.g.:
  >>> @tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
  ... def l1_reg(weight_matrix):
  ...    return 0.01 * tf.math.reduce_sum(tf.math.abs(weight_matrix))
  ...
  >>> layer = tf.keras.layers.Dense(5, input_dim=5,
  ...     kernel_initializer='ones', kernel_regularizer=l1_reg)
  >>> tensor = tf.ones(shape=(5, 5))
  >>> out = layer(tensor)
  >>> layer.losses
  [<tf.Tensor: shape=(), dtype=float32, numpy=0.25>]
  Alternatively, you can write your custom regularizers in an
  object-oriented way by extending this regularizer base class, e.g.:
  >>> @tf.keras.utils.register_keras_serializable(package='Custom', name='l2')
  ... class L2Regularizer(tf.keras.regularizers.Regularizer):
  ...     self.l2 = l2
  ...
  ...   def __call__(self, x):
  ...     return self.l2 * tf.math.reduce_sum(tf.math.square(x))
  ...
  ...   def get_config(self):
  ...     return {'l2': float(self.l2)}
  ...
  >>> layer = tf.keras.layers.Dense(
  ...   5, input_dim=5, kernel_initializer='ones',
  ...   kernel_regularizer=L2Regularizer(l2=0.5))
  >>> tensor = tf.ones(shape=(5, 5))
  >>> out = layer(tensor)
  >>> layer.losses
  [<tf.Tensor: shape=(), dtype=float32, numpy=12.5>]
  Registering the regularizers as serializable is optional if you are just
  training and executing models, exporting to and from SavedModels, or saving
  and loading weight checkpoints.
  Registration is required for Keras `model_to_estimator`, saving and
  loading models to HDF5 formats, Keras model cloning, some visualization
  utilities, and exporting models to and from JSON. If using this functionality,
  you must make sure any python process running your model has also defined
  and registered your custom regularizer.
  `tf.keras.utils.register_keras_serializable` is only available in TF 2.1 and
  beyond. In earlier versions of TensorFlow you must pass your custom
  regularizer to the `custom_objects` argument of methods that expect custom
  regularizers to be registered as serializable.
  """
  def __call__(self, x):
    return 0.
  @classmethod
  def from_config(cls, config):
    return cls(**config)
  def get_config(self):
    """Returns the config of the regularizer.
    An regularizer config is a Python dictionary (serializable)
    containing all configuration parameters of the regularizer.
    The same regularizer can be reinstantiated later
    (without any saved state) from this configuration.
    This method is optional if you are just training and executing models,
    exporting to and from SavedModels, or using weight checkpoints.
    This method is required for Keras `model_to_estimator`, saving and
    loading models to HDF5 formats, Keras model cloning, some visualization
    utilities, and exporting models to and from JSON.
    Returns:
        Python dictionary.
    """
    raise NotImplementedError(str(self) + ' does not implement get_config()')
@keras_export('keras.regularizers.L1L2')
class L1L2(Regularizer):
  """A regularizer that applies both L1 and L2 regularization penalties.
  The L1 regularization penalty is computed as:
  `loss = l1 * reduce_sum(abs(x))`
  The L2 regularization penalty is computed as
  `loss = l2 * reduce_sum(square(x))`
  L1L2 may be passed to a layer as a string identifier:
  >>> dense = tf.keras.layers.Dense(3, kernel_regularizer='l1_l2')
  In this case, the default values used are `l1=0.01` and `l2=0.01`.
  Attributes:
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.
  """
    l1 = 0. if l1 is None else l1
    l2 = 0. if l2 is None else l2
    _check_penalty_number(l1)
    _check_penalty_number(l2)
    self.l1 = backend.cast_to_floatx(l1)
    self.l2 = backend.cast_to_floatx(l2)
  def __call__(self, x):
    regularization = backend.constant(0., dtype=x.dtype)
    if self.l1:
      regularization += self.l1 * math_ops.reduce_sum(math_ops.abs(x))
    if self.l2:
      regularization += self.l2 * math_ops.reduce_sum(math_ops.square(x))
    return regularization
  def get_config(self):
    return {'l1': float(self.l1), 'l2': float(self.l2)}
@keras_export('keras.regularizers.L1', 'keras.regularizers.l1')
class L1(Regularizer):
  """A regularizer that applies a L1 regularization penalty.
  The L1 regularization penalty is computed as:
  `loss = l1 * reduce_sum(abs(x))`
  L1 may be passed to a layer as a string identifier:
  >>> dense = tf.keras.layers.Dense(3, kernel_regularizer='l1')
  In this case, the default value used is `l1=0.01`.
  Attributes:
      l1: Float; L1 regularization factor.
  """
    if kwargs:
      raise TypeError('Argument(s) not recognized: %s' % (kwargs,))
    l1 = 0.01 if l1 is None else l1
    _check_penalty_number(l1)
    self.l1 = backend.cast_to_floatx(l1)
  def __call__(self, x):
    return self.l1 * math_ops.reduce_sum(math_ops.abs(x))
  def get_config(self):
    return {'l1': float(self.l1)}
@keras_export('keras.regularizers.L2', 'keras.regularizers.l2')
class L2(Regularizer):
  """A regularizer that applies a L2 regularization penalty.
  The L2 regularization penalty is computed as:
  `loss = l2 * reduce_sum(square(x))`
  L2 may be passed to a layer as a string identifier:
  >>> dense = tf.keras.layers.Dense(3, kernel_regularizer='l2')
  In this case, the default value used is `l2=0.01`.
  Attributes:
      l2: Float; L2 regularization factor.
  """
    if kwargs:
      raise TypeError('Argument(s) not recognized: %s' % (kwargs,))
    l2 = 0.01 if l2 is None else l2
    _check_penalty_number(l2)
    self.l2 = backend.cast_to_floatx(l2)
  def __call__(self, x):
    return self.l2 * math_ops.reduce_sum(math_ops.square(x))
  def get_config(self):
    return {'l2': float(self.l2)}
@keras_export('keras.regularizers.l1_l2')
  r"""Create a regularizer that applies both L1 and L2 penalties.
  The L1 regularization penalty is computed as:
  `loss = l1 * reduce_sum(abs(x))`
  The L2 regularization penalty is computed as:
  `loss = l2 * reduce_sum(square(x))`
  Args:
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.
  Returns:
    An L1L2 Regularizer with the given regularization factors.
  """
  return L1L2(l1=l1, l2=l2)
l1 = L1
l2 = L2
@keras_export('keras.regularizers.serialize')
def serialize(regularizer):
  return serialize_keras_object(regularizer)
@keras_export('keras.regularizers.deserialize')
def deserialize(config, custom_objects=None):
  if config == 'l1_l2':
    return L1L2(l1=0.01, l2=0.01)
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='regularizer')
@keras_export('keras.regularizers.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(
        'Could not interpret regularizer identifier: {}'.format(identifier))
