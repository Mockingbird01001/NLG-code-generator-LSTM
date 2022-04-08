
import enum
class Rule(object):
  def __init__(self, module_prefix):
    self._prefix = module_prefix
  def matches(self, module_name):
    return (module_name.startswith(self._prefix + '.') or
            module_name == self._prefix)
class Action(enum.Enum):
  NONE = 0
  CONVERT = 1
  DO_NOT_CONVERT = 2
class DoNotConvert(Rule):
  def __str__(self):
    return 'DoNotConvert rule for {}'.format(self._prefix)
  def get_action(self, module):
    if self.matches(module.__name__):
      return Action.DO_NOT_CONVERT
    return Action.NONE
class Convert(Rule):
  def __str__(self):
    return 'Convert rule for {}'.format(self._prefix)
  def get_action(self, module):
    if self.matches(module.__name__):
      return Action.CONVERT
    return Action.NONE
