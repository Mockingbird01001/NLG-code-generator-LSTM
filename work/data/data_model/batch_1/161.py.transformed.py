
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import types
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _flagvalues
from absl.flags import _helpers
from absl.flags import _validators
try:
  from typing import Text, List, Any
except ImportError:
  pass
try:
  import enum
except ImportError:
  pass
_helpers.disclaim_module_ids.add(id(sys.modules[__name__]))
def _register_bounds_validator_if_needed(parser, name, flag_values):
  if parser.lower_bound is not None or parser.upper_bound is not None:
    def checker(value):
      if value is not None and parser.is_outside_bounds(value):
        message = '%s is not %s' % (value, parser.syntactic_help)
        raise _exceptions.ValidationError(message)
      return True
    _validators.register_validator(name, checker, flag_values=flag_values)
def DEFINE(
    parser,
    name,
    default,
    help,
    flag_values=_flagvalues.FLAGS,
    serializer=None,
    module_name=None,
    required=False,
    **args):
  return DEFINE_flag(
      _flag.Flag(parser, serializer, name, default, help, **args), flag_values,
      module_name, required)
def DEFINE_flag(
    flag,
    flag_values=_flagvalues.FLAGS,
    module_name=None,
    required=False):
  if required and flag.default is not None:
    raise ValueError('Required flag --%s cannot have a non-None default' %
                     flag.name)
  fv = flag_values
  fv[flag.name] = flag
  if module_name:
    module = sys.modules.get(module_name)
  else:
    module, module_name = _helpers.get_calling_module_object_and_name()
  flag_values.register_flag_by_module(module_name, flag)
  flag_values.register_flag_by_module_id(id(module), flag)
  if required:
    _validators.mark_flag_as_required(flag.name, fv)
  ensure_non_none_value = (flag.default is not None) or required
  return _flagvalues.FlagHolder(
      fv, flag, ensure_non_none_value=ensure_non_none_value)
def _internal_declare_key_flags(flag_names,
                                flag_values=_flagvalues.FLAGS,
                                key_flag_values=None):
  key_flag_values = key_flag_values or flag_values
  module = _helpers.get_calling_module()
  for flag_name in flag_names:
    flag = flag_values[flag_name]
    key_flag_values.register_key_flag_for_module(module, flag)
def declare_key_flag(flag_name, flag_values=_flagvalues.FLAGS):
  if flag_name in _helpers.SPECIAL_FLAGS:
    _internal_declare_key_flags([flag_name],
                                flag_values=_helpers.SPECIAL_FLAGS,
                                key_flag_values=flag_values)
    return
  try:
    _internal_declare_key_flags([flag_name], flag_values=flag_values)
  except KeyError:
    raise ValueError('Flag --%s is undefined. To set a flag as a key flag '
                     'first define it in Python.' % flag_name)
def adopt_module_key_flags(module, flag_values=_flagvalues.FLAGS):
  if not isinstance(module, types.ModuleType):
    raise _exceptions.Error('Expected a module object, not %r.' % (module,))
  _internal_declare_key_flags(
      [f.name for f in flag_values.get_key_flags_for_module(module.__name__)],
      flag_values=flag_values)
  if module == _helpers.FLAGS_MODULE:
    _internal_declare_key_flags(
        [_helpers.SPECIAL_FLAGS[name].name for name in _helpers.SPECIAL_FLAGS],
        flag_values=_helpers.SPECIAL_FLAGS,
        key_flag_values=flag_values)
def disclaim_key_flags():
  globals_for_caller = sys._getframe(1).f_globals
  module, _ = _helpers.get_module_object_and_name(globals_for_caller)
  _helpers.disclaim_module_ids.add(id(module))
def DEFINE_string(
    name,
    default,
    help,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args):
  parser = _argument_parser.ArgumentParser()
  serializer = _argument_parser.ArgumentSerializer()
  return DEFINE(
      parser,
      name,
      default,
      help,
      flag_values,
      serializer,
      required=required,
      **args)
def DEFINE_boolean(
    name,
    default,
    help,
    flag_values=_flagvalues.FLAGS,
    module_name=None,
    required=False,
    **args):
  return DEFINE_flag(
      _flag.BooleanFlag(name, default, help, **args), flag_values, module_name,
      required)
def DEFINE_float(
    name,
    default,
    help,
    lower_bound=None,
    upper_bound=None,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args):
  parser = _argument_parser.FloatParser(lower_bound, upper_bound)
  serializer = _argument_parser.ArgumentSerializer()
  result = DEFINE(
      parser,
      name,
      default,
      help,
      flag_values,
      serializer,
      required=required,
      **args)
  _register_bounds_validator_if_needed(parser, name, flag_values=flag_values)
  return result
def DEFINE_integer(
    name,
    default,
    help,
    lower_bound=None,
    upper_bound=None,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args):
  parser = _argument_parser.IntegerParser(lower_bound, upper_bound)
  serializer = _argument_parser.ArgumentSerializer()
  result = DEFINE(
      parser,
      name,
      default,
      help,
      flag_values,
      serializer,
      required=required,
      **args)
  _register_bounds_validator_if_needed(parser, name, flag_values=flag_values)
  return result
def DEFINE_enum(
    name,
    default,
    enum_values,
    help,
    flag_values=_flagvalues.FLAGS,
    module_name=None,
    required=False,
    **args):
  return DEFINE_flag(
      _flag.EnumFlag(name, default, help, enum_values, **args), flag_values,
      module_name, required)
def DEFINE_enum_class(
    name,
    default,
    enum_class,
    help,
    flag_values=_flagvalues.FLAGS,
    module_name=None,
    case_sensitive=False,
    required=False,
    **args):
  return DEFINE_flag(
      _flag.EnumClassFlag(
          name,
          default,
          help,
          enum_class,
          case_sensitive=case_sensitive,
          **args), flag_values, module_name, required)
def DEFINE_list(
    name,
    default,
    help,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args):
  parser = _argument_parser.ListParser()
  serializer = _argument_parser.CsvListSerializer(',')
  return DEFINE(
      parser,
      name,
      default,
      help,
      flag_values,
      serializer,
      required=required,
      **args)
def DEFINE_spaceseplist(
    name,
    default,
    help,
    comma_compat=False,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args):
  parser = _argument_parser.WhitespaceSeparatedListParser(
      comma_compat=comma_compat)
  serializer = _argument_parser.ListSerializer(' ')
  return DEFINE(
      parser,
      name,
      default,
      help,
      flag_values,
      serializer,
      required=required,
      **args)
def DEFINE_multi(
    parser,
    serializer,
    name,
    default,
    help,
    flag_values=_flagvalues.FLAGS,
    module_name=None,
    required=False,
    **args):
  return DEFINE_flag(
      _flag.MultiFlag(parser, serializer, name, default, help, **args),
      flag_values, module_name, required)
def DEFINE_multi_string(
    name,
    default,
    help,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args):
  parser = _argument_parser.ArgumentParser()
  serializer = _argument_parser.ArgumentSerializer()
  return DEFINE_multi(
      parser,
      serializer,
      name,
      default,
      help,
      flag_values,
      required=required,
      **args)
def DEFINE_multi_integer(
    name,
    default,
    help,
    lower_bound=None,
    upper_bound=None,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args):
  parser = _argument_parser.IntegerParser(lower_bound, upper_bound)
  serializer = _argument_parser.ArgumentSerializer()
  return DEFINE_multi(
      parser,
      serializer,
      name,
      default,
      help,
      flag_values,
      required=required,
      **args)
def DEFINE_multi_float(
    name,
    default,
    help,
    lower_bound=None,
    upper_bound=None,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args):
  parser = _argument_parser.FloatParser(lower_bound, upper_bound)
  serializer = _argument_parser.ArgumentSerializer()
  return DEFINE_multi(
      parser,
      serializer,
      name,
      default,
      help,
      flag_values,
      required=required,
      **args)
def DEFINE_multi_enum(
    name,
    default,
    enum_values,
    help,
    flag_values=_flagvalues.FLAGS,
    case_sensitive=True,
    required=False,
    **args):
  parser = _argument_parser.EnumParser(enum_values, case_sensitive)
  serializer = _argument_parser.ArgumentSerializer()
  return DEFINE_multi(
      parser,
      serializer,
      name,
      default,
      help,
      flag_values,
      required=required,
      **args)
def DEFINE_multi_enum_class(
    name,
    default,
    enum_class,
    help,
    flag_values=_flagvalues.FLAGS,
    module_name=None,
    case_sensitive=False,
    required=False,
    **args):
  return DEFINE_flag(
      _flag.MultiEnumClassFlag(
          name, default, help, enum_class, case_sensitive=case_sensitive),
      flag_values,
      module_name,
      required=required,
      **args)
def DEFINE_alias(
    name,
    original_name,
    flag_values=_flagvalues.FLAGS,
    module_name=None):
  if original_name not in flag_values:
    raise _exceptions.UnrecognizedFlagError(original_name)
  flag = flag_values[original_name]
  class _FlagAlias(_flag.Flag):
    def parse(self, argument):
      flag.parse(argument)
      self.present += 1
    def _parse_from_default(self, value):
      return value
    @property
    def value(self):
      return flag.value
    @value.setter
    def value(self, value):
      flag.value = value
  help_msg = 'Alias for --%s.' % flag.name
  return DEFINE_flag(
      _FlagAlias(
          flag.parser,
          flag.serializer,
          name,
          flag.default,
          help_msg,
          boolean=flag.boolean), flag_values, module_name)
