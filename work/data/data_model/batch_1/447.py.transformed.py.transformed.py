
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import functools
from absl._collections_abc import abc
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _helpers
import six
@functools.total_ordering
class Flag(object):
  def __init__(self, parser, serializer, name, default, help_string,
               short_name=None, boolean=False, allow_override=False,
               allow_override_cpp=False, allow_hide_cpp=False,
               allow_overwrite=True, allow_using_method_names=False):
    self.name = name
    if not help_string:
      help_string = '(no help available)'
    self.help = help_string
    self.short_name = short_name
    self.boolean = boolean
    self.present = 0
    self.parser = parser
    self.serializer = serializer
    self.allow_override = allow_override
    self.allow_override_cpp = allow_override_cpp
    self.allow_hide_cpp = allow_hide_cpp
    self.allow_overwrite = allow_overwrite
    self.allow_using_method_names = allow_using_method_names
    self.using_default_value = True
    self._value = None
    self.validators = []
    if self.allow_hide_cpp and self.allow_override_cpp:
      raise _exceptions.Error(
          "Can't have both allow_hide_cpp (means use Python flag) and "
          'allow_override_cpp (means use C++ flag after InitGoogle)')
    self._set_default(default)
  @property
  def value(self):
    return self._value
  @value.setter
  def value(self, value):
    self._value = value
  def __hash__(self):
    return hash(id(self))
  def __eq__(self, other):
    return self is other
  def __lt__(self, other):
    if isinstance(other, Flag):
      return id(self) < id(other)
    return NotImplemented
  def __getstate__(self):
    raise TypeError("can't pickle Flag objects")
  def __copy__(self):
    raise TypeError('%s does not support shallow copies. '
                    'Use copy.deepcopy instead.' % type(self).__name__)
  def __deepcopy__(self, memo):
    result = object.__new__(type(self))
    result.__dict__ = copy.deepcopy(self.__dict__, memo)
    return result
  def _get_parsed_value_as_string(self, value):
    if value is None:
      return None
    if self.serializer:
      return repr(self.serializer.serialize(value))
    if self.boolean:
      if value:
        return repr('true')
      else:
        return repr('false')
    return repr(_helpers.str_or_unicode(value))
  def parse(self, argument):
    if self.present and not self.allow_overwrite:
      raise _exceptions.IllegalFlagValueError(
          'flag --%s=%s: already defined as %s' % (
              self.name, argument, self.value))
    self.value = self._parse(argument)
    self.present += 1
  def _parse(self, argument):
    try:
      return self.parser.parse(argument)
    except (TypeError, ValueError) as e:
      raise _exceptions.IllegalFlagValueError(
          'flag --%s=%s: %s' % (self.name, argument, e))
  def unparse(self):
    self.value = self.default
    self.using_default_value = True
    self.present = 0
  def serialize(self):
    return self._serialize(self.value)
  def _serialize(self, value):
    if value is None:
      return ''
    if self.boolean:
      if value:
        return '--%s' % self.name
      else:
        return '--no%s' % self.name
    else:
      if not self.serializer:
        raise _exceptions.Error(
            'Serializer not present for flag %s' % self.name)
      return '--%s=%s' % (self.name, self.serializer.serialize(value))
  def _set_default(self, value):
    self.default_unparsed = value
    if value is None:
      self.default = None
    else:
      self.default = self._parse_from_default(value)
    self.default_as_str = self._get_parsed_value_as_string(self.default)
    if self.using_default_value:
      self.value = self.default
  def _parse_from_default(self, value):
    return self._parse(value)
  def flag_type(self):
    return self.parser.flag_type()
  def _create_xml_dom_element(self, doc, module_name, is_key=False):
    element = doc.createElement('flag')
    if is_key:
      element.appendChild(_helpers.create_xml_dom_element(doc, 'key', 'yes'))
    element.appendChild(_helpers.create_xml_dom_element(
        doc, 'file', module_name))
    element.appendChild(_helpers.create_xml_dom_element(doc, 'name', self.name))
    if self.short_name:
      element.appendChild(_helpers.create_xml_dom_element(
          doc, 'short_name', self.short_name))
    if self.help:
      element.appendChild(_helpers.create_xml_dom_element(
          doc, 'meaning', self.help))
    if self.serializer and not isinstance(self.default, str):
      if self.default is not None:
        default_serialized = self.serializer.serialize(self.default)
      else:
        default_serialized = ''
    else:
      default_serialized = self.default
    element.appendChild(_helpers.create_xml_dom_element(
        doc, 'default', default_serialized))
    value_serialized = self._serialize_value_for_xml(self.value)
    element.appendChild(_helpers.create_xml_dom_element(
        doc, 'current', value_serialized))
    element.appendChild(_helpers.create_xml_dom_element(
        doc, 'type', self.flag_type()))
    for e in self._extra_xml_dom_elements(doc):
      element.appendChild(e)
    return element
  def _serialize_value_for_xml(self, value):
    return value
  def _extra_xml_dom_elements(self, doc):
    return self.parser._custom_xml_dom_elements(doc)
class BooleanFlag(Flag):
  def __init__(self, name, default, help, short_name=None, **args):
    p = _argument_parser.BooleanParser()
    super(BooleanFlag, self).__init__(
        p, None, name, default, help, short_name, 1, **args)
class EnumFlag(Flag):
  def __init__(self, name, default, help, enum_values,
               short_name=None, case_sensitive=True, **args):
    p = _argument_parser.EnumParser(enum_values, case_sensitive)
    g = _argument_parser.ArgumentSerializer()
    super(EnumFlag, self).__init__(
        p, g, name, default, help, short_name, **args)
    self.help = '<%s>: %s' % ('|'.join(enum_values), self.help)
  def _extra_xml_dom_elements(self, doc):
    elements = []
    for enum_value in self.parser.enum_values:
      elements.append(_helpers.create_xml_dom_element(
          doc, 'enum_value', enum_value))
    return elements
class EnumClassFlag(Flag):
  def __init__(
      self,
      name,
      default,
      help,
      enum_class,
      short_name=None,
      case_sensitive=False,
      **args):
    p = _argument_parser.EnumClassParser(
        enum_class, case_sensitive=case_sensitive)
    g = _argument_parser.EnumClassSerializer(lowercase=not case_sensitive)
    super(EnumClassFlag, self).__init__(
        p, g, name, default, help, short_name, **args)
    self.help = '<%s>: %s' % ('|'.join(p.member_names), self.help)
  def _extra_xml_dom_elements(self, doc):
    elements = []
    for enum_value in self.parser.enum_class.__members__.keys():
      elements.append(_helpers.create_xml_dom_element(
          doc, 'enum_value', enum_value))
    return elements
class MultiFlag(Flag):
  def __init__(self, *args, **kwargs):
    super(MultiFlag, self).__init__(*args, **kwargs)
    self.help += ';\n    repeat this option to specify a list of values'
  def parse(self, arguments):
    new_values = self._parse(arguments)
    if self.present:
      self.value.extend(new_values)
    else:
      self.value = new_values
    self.present += len(new_values)
  def _parse(self, arguments):
    if (isinstance(arguments, abc.Iterable) and
        not isinstance(arguments, six.string_types)):
      arguments = list(arguments)
    if not isinstance(arguments, list):
      arguments = [arguments]
    return [super(MultiFlag, self)._parse(item) for item in arguments]
  def _serialize(self, value):
    if not self.serializer:
      raise _exceptions.Error(
          'Serializer not present for flag %s' % self.name)
    if value is None:
      return ''
    serialized_items = [
        super(MultiFlag, self)._serialize(value_item) for value_item in value
    ]
    return '\n'.join(serialized_items)
  def flag_type(self):
    return 'multi ' + self.parser.flag_type()
  def _extra_xml_dom_elements(self, doc):
    elements = []
    if hasattr(self.parser, 'enum_values'):
      for enum_value in self.parser.enum_values:
        elements.append(_helpers.create_xml_dom_element(
            doc, 'enum_value', enum_value))
    return elements
class MultiEnumClassFlag(MultiFlag):
  def __init__(self,
               name,
               default,
               help_string,
               enum_class,
               case_sensitive=False,
               **args):
    p = _argument_parser.EnumClassParser(
        enum_class, case_sensitive=case_sensitive)
    g = _argument_parser.EnumClassListSerializer(
        list_sep=',', lowercase=not case_sensitive)
    super(MultiEnumClassFlag, self).__init__(
        p, g, name, default, help_string, **args)
    self.help = (
        '<%s>: %s;\n    repeat this option to specify a list of values' %
        ('|'.join(p.member_names), help_string or '(no help available)'))
  def _extra_xml_dom_elements(self, doc):
    elements = []
    for enum_value in self.parser.enum_class.__members__.keys():
      elements.append(_helpers.create_xml_dom_element(
          doc, 'enum_value', enum_value))
    return elements
  def _serialize_value_for_xml(self, value):
    if value is not None:
      value_serialized = self.serializer.serialize(value)
    else:
      value_serialized = ''
    return value_serialized
