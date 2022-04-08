
__author__ = 'rabsatt@google.com (Kevin Rabsatt)'
import six
class EnumTypeWrapper(object):
  DESCRIPTOR = None
  def __init__(self, enum_type):
    self._enum_type = enum_type
    self.DESCRIPTOR = enum_type
  def Name(self, number):
    try:
      return self._enum_type.values_by_number[number].name
    except KeyError:
      pass
    if not isinstance(number, six.integer_types):
      raise TypeError(
          'Enum value for {} must be an int, but got {} {!r}.'.format(
              self._enum_type.name, type(number), number))
    else:
      raise ValueError('Enum {} has no name defined for value {!r}'.format(
          self._enum_type.name, number))
  def Value(self, name):
    try:
      return self._enum_type.values_by_name[name].number
    except KeyError:
      pass
    raise ValueError('Enum {} has no value defined for name {!r}'.format(
        self._enum_type.name, name))
  def keys(self):
    return [value_descriptor.name
            for value_descriptor in self._enum_type.values]
  def values(self):
    return [value_descriptor.number
            for value_descriptor in self._enum_type.values]
  def items(self):
    return [(value_descriptor.name, value_descriptor.number)
            for value_descriptor in self._enum_type.values]
  def __getattr__(self, name):
    try:
      return super(
          EnumTypeWrapper,
          self).__getattribute__('_enum_type').values_by_name[name].number
    except KeyError:
      pass
    raise AttributeError('Enum {} has no value defined for name {!r}'.format(
        self._enum_type.name, name))
