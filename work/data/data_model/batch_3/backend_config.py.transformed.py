
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
_FLOATX = 'float32'
_EPSILON = 1e-7
_IMAGE_DATA_FORMAT = 'channels_last'
@keras_export('keras.backend.epsilon')
@dispatch.add_dispatch_support
def epsilon():
  """Returns the value of the fuzz factor used in numeric expressions.
  Returns:
      A float.
  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  """
  return _EPSILON
@keras_export('keras.backend.set_epsilon')
def set_epsilon(value):
  """Sets the value of the fuzz factor used in numeric expressions.
  Args:
      value: float. New value of epsilon.
  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  >>> tf.keras.backend.set_epsilon(1e-5)
  >>> tf.keras.backend.epsilon()
  1e-05
   >>> tf.keras.backend.set_epsilon(1e-7)
  """
  global _EPSILON
  _EPSILON = value
@keras_export('keras.backend.floatx')
def floatx():
  """Returns the default float type, as a string.
  E.g. `'float16'`, `'float32'`, `'float64'`.
  Returns:
      String, the current default float type.
  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  """
  return _FLOATX
@keras_export('keras.backend.set_floatx')
def set_floatx(value):
  """Sets the default float type.
  Note: It is not recommended to set this to float16 for training, as this will
  likely cause numeric stability issues. Instead, mixed precision, which is
  using a mix of float16 and float32, can be used by calling
  `tf.keras.mixed_precision.set_global_policy('mixed_float16')`. See the
  [mixed precision guide](
    https://www.tensorflow.org/guide/keras/mixed_precision) for details.
  Args:
      value: String; `'float16'`, `'float32'`, or `'float64'`.
  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  >>> tf.keras.backend.set_floatx('float64')
  >>> tf.keras.backend.floatx()
  'float64'
  >>> tf.keras.backend.set_floatx('float32')
  Raises:
      ValueError: In case of invalid value.
  """
  global _FLOATX
  if value not in {'float16', 'float32', 'float64'}:
    raise ValueError('Unknown floatx type: ' + str(value))
  _FLOATX = str(value)
@keras_export('keras.backend.image_data_format')
@dispatch.add_dispatch_support
def image_data_format():
  """Returns the default image data format convention.
  Returns:
      A string, either `'channels_first'` or `'channels_last'`
  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  """
  return _IMAGE_DATA_FORMAT
@keras_export('keras.backend.set_image_data_format')
def set_image_data_format(data_format):
  """Sets the value of the image data format convention.
  Args:
      data_format: string. `'channels_first'` or `'channels_last'`.
  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  >>> tf.keras.backend.set_image_data_format('channels_first')
  >>> tf.keras.backend.image_data_format()
  'channels_first'
  >>> tf.keras.backend.set_image_data_format('channels_last')
  Raises:
      ValueError: In case of invalid `data_format` value.
  """
  global _IMAGE_DATA_FORMAT
  if data_format not in {'channels_last', 'channels_first'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  _IMAGE_DATA_FORMAT = str(data_format)
