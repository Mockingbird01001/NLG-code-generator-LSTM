
import importlib
import types
from tensorflow.python.platform import tf_logging as logging
class LazyLoader(types.ModuleType):
  def __init__(self, local_name, parent_module_globals, name, warning=None):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    self._warning = warning
    super(LazyLoader, self).__init__(name)
  def _load(self):
    module = importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module
    if self._warning:
      logging.warning(self._warning)
      self._warning = None
    self.__dict__.update(module.__dict__)
    return module
  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)
  def __dir__(self):
    module = self._load()
    return dir(module)
