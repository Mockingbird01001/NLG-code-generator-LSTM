
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
import six
try:
  import typing
  from typing import Text, Optional
except ImportError:
  typing = None
_helpers.disclaim_module_ids.add(id(sys.modules[__name__]))
class FlagValues(object):
  def __init__(self):
    self.__dict__['__flags'] = {}
    self.__dict__['__hiddenflags'] = set()
    self.__dict__['__flags_by_module'] = {}
    self.__dict__['__flags_by_module_id'] = {}
    self.__dict__['__key_flags_by_module'] = {}
    self.__dict__['__flags_parsed'] = False
    self.__dict__['__unparse_flags_called'] = False
    self.__dict__['__set_unknown'] = None
    self.__dict__['__banned_flag_names'] = frozenset(dir(FlagValues))
    self.__dict__['__use_gnu_getopt'] = True
    self.__dict__['__use_gnu_getopt_explicitly_set'] = False
    self.__dict__['__is_retired_flag_func'] = None
  def set_gnu_getopt(self, gnu_getopt=True):
    self.__dict__['__use_gnu_getopt'] = gnu_getopt
    self.__dict__['__use_gnu_getopt_explicitly_set'] = True
  def is_gnu_getopt(self):
    return self.__dict__['__use_gnu_getopt']
  def _flags(self):
    return self.__dict__['__flags']
  def flags_by_module_dict(self):
    return self.__dict__['__flags_by_module']
  def flags_by_module_id_dict(self):
    return self.__dict__['__flags_by_module_id']
  def key_flags_by_module_dict(self):
    return self.__dict__['__key_flags_by_module']
  def register_flag_by_module(self, module_name, flag):
    flags_by_module = self.flags_by_module_dict()
    flags_by_module.setdefault(module_name, []).append(flag)
  def register_flag_by_module_id(self, module_id, flag):
    flags_by_module_id = self.flags_by_module_id_dict()
    flags_by_module_id.setdefault(module_id, []).append(flag)
  def register_key_flag_for_module(self, module_name, flag):
    key_flags_by_module = self.key_flags_by_module_dict()
    key_flags = key_flags_by_module.setdefault(module_name, [])
    if flag not in key_flags:
      key_flags.append(flag)
  def _flag_is_registered(self, flag_obj):
    flag_dict = self._flags()
    name = flag_obj.name
    if flag_dict.get(name, None) == flag_obj:
      return True
    short_name = flag_obj.short_name
    if (short_name is not None and flag_dict.get(short_name, None) == flag_obj):
      return True
    return False
  def _cleanup_unregistered_flag_from_module_dicts(self, flag_obj):
    if self._flag_is_registered(flag_obj):
      return
    for flags_by_module_dict in (self.flags_by_module_dict(),
                                 self.flags_by_module_id_dict(),
                                 self.key_flags_by_module_dict()):
      for flags_in_module in six.itervalues(flags_by_module_dict):
        while flag_obj in flags_in_module:
          flags_in_module.remove(flag_obj)
  def get_flags_for_module(self, module):
    if not isinstance(module, str):
      module = module.__name__
    if module == '__main__':
      module = sys.argv[0]
    return list(self.flags_by_module_dict().get(module, []))
  def get_key_flags_for_module(self, module):
    if not isinstance(module, str):
      module = module.__name__
    if module == '__main__':
      module = sys.argv[0]
    key_flags = self.get_flags_for_module(module)
    for flag in self.key_flags_by_module_dict().get(module, []):
      if flag not in key_flags:
        key_flags.append(flag)
    return key_flags
  def find_module_defining_flag(self, flagname, default=None):
    registered_flag = self._flags().get(flagname)
    if registered_flag is None:
      return default
    for module, flags in six.iteritems(self.flags_by_module_dict()):
      for flag in flags:
        if (flag.name == registered_flag.name and
            flag.short_name == registered_flag.short_name):
          return module
    return default
  def find_module_id_defining_flag(self, flagname, default=None):
    registered_flag = self._flags().get(flagname)
    if registered_flag is None:
      return default
    for module_id, flags in six.iteritems(self.flags_by_module_id_dict()):
      for flag in flags:
        if (flag.name == registered_flag.name and
            flag.short_name == registered_flag.short_name):
          return module_id
    return default
  def _register_unknown_flag_setter(self, setter):
    self.__dict__['__set_unknown'] = setter
  def _set_unknown_flag(self, name, value):
    setter = self.__dict__['__set_unknown']
    if setter:
      try:
        setter(name, value)
        return value
      except (TypeError, ValueError):
        raise _exceptions.IllegalFlagValueError(
            '"{1}" is not valid for --{0}'.format(name, value))
      except NameError:
        pass
    raise _exceptions.UnrecognizedFlagError(name, value)
  def append_flag_values(self, flag_values):
    for flag_name, flag in six.iteritems(flag_values._flags()):
      if flag_name == flag.name:
        try:
          self[flag_name] = flag
        except _exceptions.DuplicateFlagError:
          raise _exceptions.DuplicateFlagError.from_flag(
              flag_name, self, other_flag_values=flag_values)
  def remove_flag_values(self, flag_values):
    for flag_name in flag_values:
      self.__delattr__(flag_name)
  def __setitem__(self, name, flag):
    fl = self._flags()
    if not isinstance(flag, _flag.Flag):
      raise _exceptions.IllegalFlagValueError(flag)
    if str is bytes and isinstance(name, unicode):
      name = name.encode('utf-8')
    if not isinstance(name, type('')):
      raise _exceptions.Error('Flag name must be a string')
    if not name:
      raise _exceptions.Error('Flag name cannot be empty')
    if ' ' in name:
      raise _exceptions.Error('Flag name cannot contain a space')
    self._check_method_name_conflicts(name, flag)
    if name in fl and not flag.allow_override and not fl[name].allow_override:
      module, module_name = _helpers.get_calling_module_object_and_name()
      if (self.find_module_defining_flag(name) == module_name and
          id(module) != self.find_module_id_defining_flag(name)):
        return
      raise _exceptions.DuplicateFlagError.from_flag(name, self)
    short_name = flag.short_name
    flags_to_cleanup = set()
    if short_name is not None:
      if (short_name in fl and not flag.allow_override and
          not fl[short_name].allow_override):
        raise _exceptions.DuplicateFlagError.from_flag(short_name, self)
      if short_name in fl and fl[short_name] != flag:
        flags_to_cleanup.add(fl[short_name])
      fl[short_name] = flag
    if (name not in fl
        or fl[name].using_default_value or not flag.using_default_value):
      if name in fl and fl[name] != flag:
        flags_to_cleanup.add(fl[name])
      fl[name] = flag
    for f in flags_to_cleanup:
      self._cleanup_unregistered_flag_from_module_dicts(f)
  def __dir__(self):
    return sorted(self.__dict__['__flags'])
  def __getitem__(self, name):
    return self._flags()[name]
  def _hide_flag(self, name):
    self.__dict__['__hiddenflags'].add(name)
  def __getattr__(self, name):
    fl = self._flags()
    if name not in fl:
      raise AttributeError(name)
    if name in self.__dict__['__hiddenflags']:
      raise AttributeError(name)
    if self.__dict__['__flags_parsed'] or fl[name].present:
      return fl[name].value
    else:
      error_message = ('Trying to access flag --%s before flags were parsed.' %
                       name)
      if six.PY2:
        logging.error(error_message)
      raise _exceptions.UnparsedFlagAccessError(error_message)
  def __setattr__(self, name, value):
    self._set_attributes(**{name: value})
    return value
  def _set_attributes(self, **attributes):
    fl = self._flags()
    known_flags = set()
    for name, value in six.iteritems(attributes):
      if name in self.__dict__['__hiddenflags']:
        raise AttributeError(name)
      if name in fl:
        fl[name].value = value
        known_flags.add(name)
      else:
        self._set_unknown_flag(name, value)
    for name in known_flags:
      self._assert_validators(fl[name].validators)
      fl[name].using_default_value = False
  def validate_all_flags(self):
    all_validators = set()
    for flag in six.itervalues(self._flags()):
      all_validators.update(flag.validators)
    self._assert_validators(all_validators)
  def _assert_validators(self, validators):
    for validator in sorted(
        validators, key=lambda validator: validator.insertion_index):
      try:
        validator.verify(self)
      except _exceptions.ValidationError as e:
        message = validator.print_flags_with_values(self)
        raise _exceptions.IllegalFlagValueError('%s: %s' % (message, str(e)))
  def __delattr__(self, flag_name):
    fl = self._flags()
    if flag_name not in fl:
      raise AttributeError(flag_name)
    flag_obj = fl[flag_name]
    del fl[flag_name]
    self._cleanup_unregistered_flag_from_module_dicts(flag_obj)
  def set_default(self, name, value):
    fl = self._flags()
    if name not in fl:
      self._set_unknown_flag(name, value)
      return
    fl[name]._set_default(value)
    self._assert_validators(fl[name].validators)
  def __contains__(self, name):
    return name in self._flags()
  def __len__(self):
    return len(self.__dict__['__flags'])
  def __iter__(self):
    return iter(self._flags())
  def __call__(self, argv, known_only=False):
    if _helpers.is_bytes_or_string(argv):
      raise TypeError(
          'argv should be a tuple/list of strings, not bytes or string.')
    if not argv:
      raise ValueError(
          'argv cannot be an empty list, and must contain the program name as '
          'the first element.')
    program_name = argv[0]
    args = self.read_flags_from_files(argv[1:], force_gnu=False)
    unknown_flags, unparsed_args = self._parse_args(args, known_only)
    for name, value in unknown_flags:
      suggestions = _helpers.get_flag_suggestions(name, list(self))
      raise _exceptions.UnrecognizedFlagError(
          name, value, suggestions=suggestions)
    self.mark_as_parsed()
    self.validate_all_flags()
    return [program_name] + unparsed_args
  def __getstate__(self):
    raise TypeError("can't pickle FlagValues")
  def __copy__(self):
    raise TypeError('FlagValues does not support shallow copies. '
                    'Use absl.testing.flagsaver or copy.deepcopy instead.')
  def __deepcopy__(self, memo):
    result = object.__new__(type(self))
    result.__dict__.update(copy.deepcopy(self.__dict__, memo))
    return result
  def _set_is_retired_flag_func(self, is_retired_flag_func):
    self.__dict__['__is_retired_flag_func'] = is_retired_flag_func
  def _parse_args(self, args, known_only):
    unparsed_names_and_args = []
    undefok = set()
    retired_flag_func = self.__dict__['__is_retired_flag_func']
    flag_dict = self._flags()
    args = iter(args)
    for arg in args:
      value = None
      def get_value():
        try:
          return next(args) if value is None else value
        except StopIteration:
          raise _exceptions.Error('Missing value for flag ' + arg)
      if not arg.startswith('-'):
        unparsed_names_and_args.append((None, arg))
        if self.is_gnu_getopt():
          continue
        else:
          break
      if arg == '--':
        if known_only:
          unparsed_names_and_args.append((None, arg))
        break
      if arg.startswith('--'):
        arg_without_dashes = arg[2:]
      else:
        arg_without_dashes = arg[1:]
      if '=' in arg_without_dashes:
        name, value = arg_without_dashes.split('=', 1)
      else:
        name, value = arg_without_dashes, None
      if not name:
        unparsed_names_and_args.append((None, arg))
        if self.is_gnu_getopt():
          continue
        else:
          break
      if name == 'undefok':
        value = get_value()
        undefok.update(v.strip() for v in value.split(','))
        undefok.update('no' + v.strip() for v in value.split(','))
        continue
      flag = flag_dict.get(name)
      if flag:
        if flag.boolean and value is None:
          value = 'true'
        else:
          value = get_value()
      elif name.startswith('no') and len(name) > 2:
        noflag = flag_dict.get(name[2:])
        if noflag and noflag.boolean:
          if value is not None:
            raise ValueError(arg + ' does not take an argument')
          flag = noflag
          value = 'false'
      if retired_flag_func and not flag:
        is_retired, is_bool = retired_flag_func(name)
        if not is_retired and name.startswith('no'):
          is_retired, is_bool = retired_flag_func(name[2:])
          is_retired = is_retired and is_bool
        if is_retired:
          if not is_bool and value is None:
            get_value()
          logging.error(
              'Flag "%s" is retired and should no longer '
              'be specified. See go/totw/90.', name)
          continue
      if flag:
        flag.parse(value)
        flag.using_default_value = False
      else:
        unparsed_names_and_args.append((name, arg))
    unknown_flags = []
    unparsed_args = []
    for name, arg in unparsed_names_and_args:
      if name is None:
        unparsed_args.append(arg)
      elif name in undefok:
        continue
      else:
        if known_only:
          unparsed_args.append(arg)
        else:
          unknown_flags.append((name, arg))
    unparsed_args.extend(list(args))
    return unknown_flags, unparsed_args
  def is_parsed(self):
    return self.__dict__['__flags_parsed']
  def mark_as_parsed(self):
    self.__dict__['__flags_parsed'] = True
  def unparse_flags(self):
    for f in self._flags().values():
      f.unparse()
    logging.info('unparse_flags() called; flags access will now raise errors.')
    self.__dict__['__flags_parsed'] = False
    self.__dict__['__unparse_flags_called'] = True
  def flag_values_dict(self):
    return {name: flag.value for name, flag in six.iteritems(self._flags())}
  def __str__(self):
    return self.get_help()
  def get_help(self, prefix='', include_special_flags=True):
    flags_by_module = self.flags_by_module_dict()
    if flags_by_module:
      modules = sorted(flags_by_module)
      main_module = sys.argv[0]
      if main_module in modules:
        modules.remove(main_module)
        modules = [main_module] + modules
      return self._get_help_for_modules(modules, prefix, include_special_flags)
    else:
      output_lines = []
      values = six.itervalues(self._flags())
      if include_special_flags:
        values = itertools.chain(values,
                                 six.itervalues(
                                     _helpers.SPECIAL_FLAGS._flags()))
      self._render_flag_list(values, output_lines, prefix)
      return '\n'.join(output_lines)
  def _get_help_for_modules(self, modules, prefix, include_special_flags):
    output_lines = []
    for module in modules:
      self._render_our_module_flags(module, output_lines, prefix)
    if include_special_flags:
      self._render_module_flags(
          'absl.flags',
          six.itervalues(_helpers.SPECIAL_FLAGS._flags()),
          output_lines,
          prefix)
    return '\n'.join(output_lines)
  def _render_module_flags(self, module, flags, output_lines, prefix=''):
    if not isinstance(module, str):
      module = module.__name__
    output_lines.append('\n%s%s:' % (prefix, module))
    self._render_flag_list(flags, output_lines, prefix + '  ')
  def _render_our_module_flags(self, module, output_lines, prefix=''):
    flags = self.get_flags_for_module(module)
    if flags:
      self._render_module_flags(module, flags, output_lines, prefix)
  def _render_our_module_key_flags(self, module, output_lines, prefix=''):
    key_flags = self.get_key_flags_for_module(module)
    if key_flags:
      self._render_module_flags(module, key_flags, output_lines, prefix)
  def module_help(self, module):
    helplist = []
    self._render_our_module_key_flags(module, helplist)
    return '\n'.join(helplist)
  def main_module_help(self):
    return self.module_help(sys.argv[0])
  def _render_flag_list(self, flaglist, output_lines, prefix='  '):
    fl = self._flags()
    special_fl = _helpers.SPECIAL_FLAGS._flags()
    flaglist = [(flag.name, flag) for flag in flaglist]
    flaglist.sort()
    flagset = {}
    for (name, flag) in flaglist:
      if fl.get(name, None) != flag and special_fl.get(name, None) != flag:
        continue
      if flag in flagset:
        continue
      flagset[flag] = 1
      flaghelp = ''
      if flag.short_name:
        flaghelp += '-%s,' % flag.short_name
      if flag.boolean:
        flaghelp += '--[no]%s:' % flag.name
      else:
        flaghelp += '--%s:' % flag.name
      flaghelp += ' '
      if flag.help:
        flaghelp += flag.help
      flaghelp = _helpers.text_wrap(
          flaghelp, indent=prefix + '  ', firstline_indent=prefix)
      if flag.default_as_str:
        flaghelp += '\n'
        flaghelp += _helpers.text_wrap(
            '(default: %s)' % flag.default_as_str, indent=prefix + '  ')
      if flag.parser.syntactic_help:
        flaghelp += '\n'
        flaghelp += _helpers.text_wrap(
            '(%s)' % flag.parser.syntactic_help, indent=prefix + '  ')
      output_lines.append(flaghelp)
  def get_flag_value(self, name, default):
    value = self.__getattr__(name)
    if value is not None:
      return value
    else:
      return default
  def _is_flag_file_directive(self, flag_string):
    if isinstance(flag_string, type('')):
      if flag_string.startswith('--flagfile='):
        return 1
      elif flag_string == '--flagfile':
        return 1
      elif flag_string.startswith('-flagfile='):
        return 1
      elif flag_string == '-flagfile':
        return 1
      else:
        return 0
    return 0
  def _extract_filename(self, flagfile_str):
    if flagfile_str.startswith('--flagfile='):
      return os.path.expanduser((flagfile_str[(len('--flagfile=')):]).strip())
    elif flagfile_str.startswith('-flagfile='):
      return os.path.expanduser((flagfile_str[(len('-flagfile=')):]).strip())
    else:
      raise _exceptions.Error('Hit illegal --flagfile type: %s' % flagfile_str)
  def _get_flag_file_lines(self, filename, parsed_file_stack=None):
    if not filename:
      return []
    if parsed_file_stack is None:
      parsed_file_stack = []
    if filename in parsed_file_stack:
      sys.stderr.write('Warning: Hit circular flagfile dependency. Ignoring'
                       ' flagfile: %s\n' % (filename,))
      return []
    else:
      parsed_file_stack.append(filename)
    line_list = []
    flag_line_list = []
    try:
      file_obj = open(filename, 'r')
    except IOError as e_msg:
      raise _exceptions.CantOpenFlagFileError(
          'ERROR:: Unable to open flagfile: %s' % e_msg)
    with file_obj:
      line_list = file_obj.readlines()
    for line in line_list:
      if line.isspace():
        pass
        pass
      elif self._is_flag_file_directive(line):
        sub_filename = self._extract_filename(line)
        included_flags = self._get_flag_file_lines(
            sub_filename, parsed_file_stack=parsed_file_stack)
        flag_line_list.extend(included_flags)
      else:
        flag_line_list.append(line.strip())
    parsed_file_stack.pop()
    return flag_line_list
  def read_flags_from_files(self, argv, force_gnu=True):
    rest_of_args = argv
    new_argv = []
    while rest_of_args:
      current_arg = rest_of_args[0]
      rest_of_args = rest_of_args[1:]
      if self._is_flag_file_directive(current_arg):
        if current_arg == '--flagfile' or current_arg == '-flagfile':
          if not rest_of_args:
            raise _exceptions.IllegalFlagValueError(
                '--flagfile with no argument')
          flag_filename = os.path.expanduser(rest_of_args[0])
          rest_of_args = rest_of_args[1:]
        else:
          flag_filename = self._extract_filename(current_arg)
        new_argv.extend(self._get_flag_file_lines(flag_filename))
      else:
        new_argv.append(current_arg)
        if current_arg == '--':
          break
        if not current_arg.startswith('-'):
          if not force_gnu and not self.__dict__['__use_gnu_getopt']:
            break
        else:
          if ('=' not in current_arg and rest_of_args and
              not rest_of_args[0].startswith('-')):
            fl = self._flags()
            name = current_arg.lstrip('-')
            if name in fl and not fl[name].boolean:
              current_arg = rest_of_args[0]
              rest_of_args = rest_of_args[1:]
              new_argv.append(current_arg)
    if rest_of_args:
      new_argv.extend(rest_of_args)
    return new_argv
  def flags_into_string(self):
    module_flags = sorted(self.flags_by_module_dict().items())
    s = ''
    for unused_module_name, flags in module_flags:
      flags = sorted(flags, key=lambda f: f.name)
      for flag in flags:
        if flag.value is not None:
          s += flag.serialize() + '\n'
    return s
  def append_flags_into_file(self, filename):
    with open(filename, 'a') as out_file:
      out_file.write(self.flags_into_string())
  def write_help_in_xml_format(self, outfile=None):
    doc = minidom.Document()
    all_flag = doc.createElement('AllFlags')
    doc.appendChild(all_flag)
    all_flag.appendChild(
        _helpers.create_xml_dom_element(doc, 'program',
                                        os.path.basename(sys.argv[0])))
    usage_doc = sys.modules['__main__'].__doc__
    if not usage_doc:
      usage_doc = '\nUSAGE: %s [flags]\n' % sys.argv[0]
    else:
      usage_doc = usage_doc.replace('%s', sys.argv[0])
    all_flag.appendChild(
        _helpers.create_xml_dom_element(doc, 'usage', usage_doc))
    key_flags = self.get_key_flags_for_module(sys.argv[0])
    flags_by_module = self.flags_by_module_dict()
    all_module_names = list(flags_by_module.keys())
    all_module_names.sort()
    for module_name in all_module_names:
      flag_list = [(f.name, f) for f in flags_by_module[module_name]]
      flag_list.sort()
      for unused_flag_name, flag in flag_list:
        is_key = flag in key_flags
        all_flag.appendChild(
            flag._create_xml_dom_element(
                doc,
                module_name,
                is_key=is_key))
    outfile = outfile or sys.stdout
    if six.PY2:
      outfile.write(doc.toprettyxml(indent='  ', encoding='utf-8'))
    else:
      outfile.write(
          doc.toprettyxml(indent='  ', encoding='utf-8').decode('utf-8'))
    outfile.flush()
  def _check_method_name_conflicts(self, name, flag):
    if flag.allow_using_method_names:
      return
    short_name = flag.short_name
    flag_names = {name} if short_name is None else {name, short_name}
    for flag_name in flag_names:
      if flag_name in self.__dict__['__banned_flag_names']:
        raise _exceptions.FlagNameConflictsWithMethodError(
            'Cannot define a flag named "{name}". It conflicts with a method '
            'on class "{class_name}". To allow defining it, use '
            'allow_using_method_names and access the flag value with '
            "FLAGS['{name}'].value. FLAGS.{name} returns the method, "
            'not the flag value.'.format(
                name=flag_name, class_name=type(self).__name__))
FLAGS = FlagValues()
if typing:
  _T = typing.TypeVar('_T')
  _Base = typing.Generic[_T]
else:
  _Base = object
class FlagHolder(_Base):
  def __init__(self, flag_values, flag, ensure_non_none_value=False):
    self._flagvalues = flag_values
    self._name = flag.name
    self._ensure_non_none_value = ensure_non_none_value
  def __eq__(self, other):
    raise TypeError(
        "unsupported operand type(s) for ==: '{0}' and '{1}' "
        "(did you mean to use '{0}.value' instead?)".format(
            type(self).__name__, type(other).__name__))
  def __bool__(self):
    raise TypeError(
        "bool() not supported for instances of type '{0}' "
        "(did you mean to use '{0}.value' instead?)".format(
            type(self).__name__))
  __nonzero__ = __bool__
  @property
  def name(self):
    return self._name
  @property
  def value(self):
    val = getattr(self._flagvalues, self._name)
    if self._ensure_non_none_value and val is None:
      raise _exceptions.IllegalFlagValueError(
          'Unexpected None value for flag %s' % self._name)
    return val
  @property
  def default(self):
    return self._flagvalues[self._name].default
