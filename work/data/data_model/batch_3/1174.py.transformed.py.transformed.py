
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
class Validator(object):
  validators_count = 0
  def __init__(self, checker, message):
    self.checker = checker
    self.message = message
    Validator.validators_count += 1
    self.insertion_index = Validator.validators_count
  def verify(self, flag_values):
    param = self._get_input_to_checker_function(flag_values)
    if not self.checker(param):
      raise _exceptions.ValidationError(self.message)
  def get_flags_names(self):
    raise NotImplementedError('This method should be overloaded')
  def print_flags_with_values(self, flag_values):
    raise NotImplementedError('This method should be overloaded')
  def _get_input_to_checker_function(self, flag_values):
    raise NotImplementedError('This method should be overloaded')
class SingleFlagValidator(Validator):
  def __init__(self, flag_name, checker, message):
    super(SingleFlagValidator, self).__init__(checker, message)
    self.flag_name = flag_name
  def get_flags_names(self):
    return [self.flag_name]
  def print_flags_with_values(self, flag_values):
    return 'flag --%s=%s' % (self.flag_name, flag_values[self.flag_name].value)
  def _get_input_to_checker_function(self, flag_values):
    return flag_values[self.flag_name].value
class MultiFlagsValidator(Validator):
  def __init__(self, flag_names, checker, message):
    super(MultiFlagsValidator, self).__init__(checker, message)
    self.flag_names = flag_names
  def _get_input_to_checker_function(self, flag_values):
    return dict([key, flag_values[key].value] for key in self.flag_names)
  def print_flags_with_values(self, flag_values):
    prefix = 'flags '
    flags_with_values = []
    for key in self.flag_names:
      flags_with_values.append('%s=%s' % (key, flag_values[key].value))
    return prefix + ', '.join(flags_with_values)
  def get_flags_names(self):
    return self.flag_names
def register_validator(flag_name,
                       checker,
                       message='Flag validation failed',
                       flag_values=_flagvalues.FLAGS):
  v = SingleFlagValidator(flag_name, checker, message)
  _add_validator(flag_values, v)
def validator(flag_name, message='Flag validation failed',
              flag_values=_flagvalues.FLAGS):
  def decorate(function):
    register_validator(flag_name, function,
                       message=message,
                       flag_values=flag_values)
    return function
  return decorate
def register_multi_flags_validator(flag_names,
                                   multi_flags_checker,
                                   message='Flags validation failed',
                                   flag_values=_flagvalues.FLAGS):
  v = MultiFlagsValidator(
      flag_names, multi_flags_checker, message)
  _add_validator(flag_values, v)
def multi_flags_validator(flag_names,
                          message='Flag validation failed',
                          flag_values=_flagvalues.FLAGS):
  def decorate(function):
    register_multi_flags_validator(flag_names,
                                   function,
                                   message=message,
                                   flag_values=flag_values)
    return function
  return decorate
def mark_flag_as_required(flag_name, flag_values=_flagvalues.FLAGS):
  if flag_values[flag_name].default is not None:
    warnings.warn(
        'Flag --%s has a non-None default value; therefore, '
        'mark_flag_as_required will pass even if flag is not specified in the '
        'command line!' % flag_name)
  register_validator(
      flag_name,
      lambda value: value is not None,
      message='Flag --{} must have a value other than None.'.format(flag_name),
      flag_values=flag_values)
def mark_flags_as_required(flag_names, flag_values=_flagvalues.FLAGS):
  for flag_name in flag_names:
    mark_flag_as_required(flag_name, flag_values)
def mark_flags_as_mutual_exclusive(flag_names, required=False,
                                   flag_values=_flagvalues.FLAGS):
  for flag_name in flag_names:
    if flag_values[flag_name].default is not None:
      warnings.warn(
          'Flag --{} has a non-None default value. That does not make sense '
          'with mark_flags_as_mutual_exclusive, which checks whether the '
          'listed flags have a value other than None.'.format(flag_name))
  def validate_mutual_exclusion(flags_dict):
    flag_count = sum(1 for val in flags_dict.values() if val is not None)
    if flag_count == 1 or (not required and flag_count == 0):
      return True
    raise _exceptions.ValidationError(
        '{} one of ({}) must have a value other than None.'.format(
            'Exactly' if required else 'At most', ', '.join(flag_names)))
  register_multi_flags_validator(
      flag_names, validate_mutual_exclusion, flag_values=flag_values)
def mark_bool_flags_as_mutual_exclusive(flag_names, required=False,
                                        flag_values=_flagvalues.FLAGS):
  for flag_name in flag_names:
    if not flag_values[flag_name].boolean:
      raise _exceptions.ValidationError(
          'Flag --{} is not Boolean, which is required for flags used in '
          'mark_bool_flags_as_mutual_exclusive.'.format(flag_name))
  def validate_boolean_mutual_exclusion(flags_dict):
    flag_count = sum(bool(val) for val in flags_dict.values())
    if flag_count == 1 or (not required and flag_count == 0):
      return True
    raise _exceptions.ValidationError(
        '{} one of ({}) must be True.'.format(
            'Exactly' if required else 'At most', ', '.join(flag_names)))
  register_multi_flags_validator(
      flag_names, validate_boolean_mutual_exclusion, flag_values=flag_values)
def _add_validator(fv, validator_instance):
  for flag_name in validator_instance.get_flags_names():
    fv[flag_name].validators.append(validator_instance)
