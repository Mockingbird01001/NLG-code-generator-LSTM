
import traceback
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
_LOCATION_TAG = "location"
_TYPE_TAG = "type"
class Registry(object):
  __slots__ = ["_name", "_registry"]
  def __init__(self, name):
    self._name = name
    self._registry = {}
  def register(self, candidate, name=None):
    if not name:
      name = candidate.__name__
    if name in self._registry:
      frame = self._registry[name][_LOCATION_TAG]
      raise KeyError(
          "Registering two %s with name '%s'! "
          "(Previous registration was in %s %s:%d)" %
          (self._name, name, frame.name, frame.filename, frame.lineno))
    logging.vlog(1, "Registering %s (%s) in %s.", name, candidate, self._name)
    stack = traceback.extract_stack(limit=3)
    stack_index = min(2, len(stack) - 1)
    if stack_index >= 0:
      location_tag = stack[stack_index]
    else:
      location_tag = ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN")
    self._registry[name] = {_TYPE_TAG: candidate, _LOCATION_TAG: location_tag}
  def list(self):
    return self._registry.keys()
  def lookup(self, name):
    name = compat.as_str(name)
    if name in self._registry:
      return self._registry[name][_TYPE_TAG]
    else:
      raise LookupError(
          "%s registry has no entry for: %s" % (self._name, name))
