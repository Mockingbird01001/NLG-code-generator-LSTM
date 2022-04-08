
import logging as _logging
import sys as _sys
import six as _six
from tensorflow.python.util import tf_decorator
_RENAMED_ARGUMENTS = {
    'flag_name': 'name',
    'default_value': 'default',
    'docstring': 'help',
}
def _wrap_define_function(original_function):
  def wrapper(*args, **kwargs):
    has_old_names = False
    for old_name, new_name in _six.iteritems(_RENAMED_ARGUMENTS):
      if old_name in kwargs:
        has_old_names = True
        value = kwargs.pop(old_name)
        kwargs[new_name] = value
    if has_old_names:
      _logging.warning(
          'Use of the keyword argument names (flag_name, default_value, '
          'docstring) is deprecated, please use (name, default, help) instead.')
    return original_function(*args, **kwargs)
  return tf_decorator.make_decorator(original_function, wrapper)
class _FlagValuesWrapper(object):
  def __init__(self, flags_object):
    self.__dict__['__wrapped'] = flags_object
  def __getattribute__(self, name):
    if name == '__dict__':
      return super(_FlagValuesWrapper, self).__getattribute__(name)
    return self.__dict__['__wrapped'].__getattribute__(name)
  def __getattr__(self, name):
    wrapped = self.__dict__['__wrapped']
    if not wrapped.is_parsed():
      wrapped(_sys.argv)
    return wrapped.__getattr__(name)
  def __setattr__(self, name, value):
    return self.__dict__['__wrapped'].__setattr__(name, value)
  def __delattr__(self, name):
    return self.__dict__['__wrapped'].__delattr__(name)
  def __dir__(self):
    return self.__dict__['__wrapped'].__dir__()
  def __getitem__(self, name):
    return self.__dict__['__wrapped'].__getitem__(name)
  def __setitem__(self, name, flag):
    return self.__dict__['__wrapped'].__setitem__(name, flag)
  def __len__(self):
    return self.__dict__['__wrapped'].__len__()
  def __iter__(self):
    return self.__dict__['__wrapped'].__iter__()
  def __str__(self):
    return self.__dict__['__wrapped'].__str__()
  def __call__(self, *args, **kwargs):
    return self.__dict__['__wrapped'].__call__(*args, **kwargs)
DEFINE_string = _wrap_define_function(DEFINE_string)
DEFINE_boolean = _wrap_define_function(DEFINE_boolean)
DEFINE_bool = DEFINE_boolean
DEFINE_float = _wrap_define_function(DEFINE_float)
DEFINE_integer = _wrap_define_function(DEFINE_integer)
