
"""Compatibility functions.
The `tf.compat` module contains two sets of compatibility functions.
The `compat.v1` and `compat.v2` submodules provide a complete copy of both the
`v1` and `v2` APIs for backwards and forwards compatibility across TensorFlow
versions 1.x and 2.x. See the
[migration guide](https://www.tensorflow.org/guide/migrate) for details.
Aside from the `compat.v1` and `compat.v2` submodules, `tf.compat` also contains
a set of helper functions for writing code that works in both:
* TensorFlow 1.x and 2.x
* Python 2 and 3
The compatibility module also provides the following aliases for common
sets of python types:
* `bytes_or_text_types`
* `complex_types`
* `integral_types`
* `real_types`
"""
import numbers as _numbers
import numpy as _np
import six as _six
import codecs
from tensorflow.python.util.tf_export import tf_export
try:
except ImportError:
def as_bytes(bytes_or_text, encoding='utf-8'):
  encoding = codecs.lookup(encoding).name
  if isinstance(bytes_or_text, bytearray):
    return bytes(bytes_or_text)
  elif isinstance(bytes_or_text, _six.text_type):
    return bytes_or_text.encode(encoding)
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text
  else:
    raise TypeError('Expected binary or unicode string, got %r' %
                    (bytes_or_text,))
def as_text(bytes_or_text, encoding='utf-8'):
  """Converts any string-like python input types to unicode.
  Returns the input as a unicode string. Uses utf-8 encoding for text
  by default.
  Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for decoding unicode.
  Returns:
    A `unicode` (Python 2) or `str` (Python 3) object.
  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
  encoding = codecs.lookup(encoding).name
  if isinstance(bytes_or_text, _six.text_type):
    return bytes_or_text
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text.decode(encoding)
  else:
    raise TypeError('Expected binary or unicode string, got %r' % bytes_or_text)
def as_str(bytes_or_text, encoding='utf-8'):
  return as_text(bytes_or_text, encoding)
tf_export('compat.as_text')(as_text)
tf_export('compat.as_bytes')(as_bytes)
tf_export('compat.as_str')(as_str)
@tf_export('compat.as_str_any')
def as_str_any(value):
  """Converts input to `str` type.
     Uses `str(value)`, except for `bytes` typed inputs, which are converted
     using `as_str`.
  Args:
    value: A object that can be converted to `str`.
  Returns:
    A `str` object.
  """
  if isinstance(value, bytes):
    return as_str(value)
  else:
    return str(value)
@tf_export('compat.path_to_str')
def path_to_str(path):
  r"""Converts input which is a `PathLike` object to `str` type.
  Converts from any python constant representation of a `PathLike` object to
  a string. If the input is not a `PathLike` object, simply returns the input.
  Args:
    path: An object that can be converted to path representation.
  Returns:
    A `str` object.
  Usage:
    In case a simplified `str` version of the path is needed from an
    `os.PathLike` object
  Examples:
  ```python
  $ tf.compat.path_to_str('C:\XYZ\tensorflow\./.././tensorflow')
  $ tf.compat.path_to_str(Path('C:\XYZ\tensorflow\./.././tensorflow'))
  $ tf.compat.path_to_str(Path('./corpus'))
  $ tf.compat.path_to_str('./.././Corpus')
  $ tf.compat.path_to_str(Path('./.././Corpus'))
  $ tf.compat.path_to_str(Path('./..////../'))
  ```
  """
  if hasattr(path, '__fspath__'):
    path = as_str_any(path.__fspath__())
  return path
def path_to_bytes(path):
  if hasattr(path, '__fspath__'):
    path = path.__fspath__()
  return as_bytes(path)
integral_types = (_numbers.Integral, _np.integer)
tf_export('compat.integral_types').export_constant(__name__, 'integral_types')
real_types = (_numbers.Real, _np.integer, _np.floating)
tf_export('compat.real_types').export_constant(__name__, 'real_types')
complex_types = (_numbers.Complex, _np.number)
tf_export('compat.complex_types').export_constant(__name__, 'complex_types')
bytes_or_text_types = (bytes, _six.text_type)
tf_export('compat.bytes_or_text_types').export_constant(__name__,
                                                        'bytes_or_text_types')
