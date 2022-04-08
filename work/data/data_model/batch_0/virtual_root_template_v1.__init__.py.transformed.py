
import sys as _sys
import importlib as _importlib
import types as _types
class _LazyLoader(_types.ModuleType):
  def __init__(self, local_name, parent_module_globals, name):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    super(_LazyLoader, self).__init__(name)
  def _load(self):
    module = _importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module
    self.__dict__.update(module.__dict__)
    return module
  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)
  def __dir__(self):
    module = self._load()
    return dir(module)
  def __reduce__(self):
    return __import__, (self.__name__,)
def _forward_module(old_name):
  parts = old_name.split(".")
  parts[0] = parts[0] + "_core"
  local_name = parts[-1]
  existing_name = ".".join(parts)
  _module = _LazyLoader(local_name, globals(), existing_name)
  return _sys.modules.setdefault(old_name, _module)
_top_level_modules = [
    "tensorflow._api",
    "tensorflow.python",
    "tensorflow.tools",
    "tensorflow.core",
    "tensorflow.compiler",
    "tensorflow.lite",
    "tensorflow.keras",
    "tensorflow.compat",
    "tensorflow.examples",
]
if "tensorflow_estimator" not in _sys.modules:
  _root_estimator = False
  _top_level_modules.append("tensorflow.estimator")
else:
  _root_estimator = True
for _m in _top_level_modules:
  _forward_module(_m)
from tensorflow_core import *
_major_api_version = 1
from tensorflow.python.util import deprecation_wrapper as _deprecation
if not isinstance(_sys.modules[__name__], _deprecation.DeprecationWrapper):
  _sys.modules[__name__] = _deprecation.DeprecationWrapper(
      _sys.modules[__name__], "")
try:
  del core
except NameError:
  pass
try:
  del python
except NameError:
  pass
try:
  del compiler
except NameError:
  pass
try:
  del tools
except NameError:
  pass
try:
  del examples
except NameError:
  pass
