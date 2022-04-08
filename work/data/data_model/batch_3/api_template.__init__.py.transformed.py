
"""
Top-level module of TensorFlow. By convention, we refer to this module as
`tf` instead of `tensorflow`, following the common practice of importing
TensorFlow via the command `import tensorflow as tf`.
The primary function of this module is to import all of the public TensorFlow
interfaces into a single place. The interfaces themselves are located in
sub-modules, as described below.
Note that the file `__init__.py` in the TensorFlow source code tree is actually
only a placeholder to enable test cases to run. The TensorFlow build replaces
this file with a file generated from [`api_template.__init__.py`](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/api_template.__init__.py)
"""
import distutils as _distutils
import inspect as _inspect
import logging as _logging
import os as _os
import site as _site
import sys as _sys
import typing as _typing
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
_os.environ['TF2_BEHAVIOR'] = '1'
from tensorflow.python import tf2 as _tf2
_tf2.enable()
_API_MODULE = _sys.modules[__name__].bitwise
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
_current_module = _sys.modules[__name__]
if not hasattr(_current_module, '__path__'):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)
try:
  from tensorboard.summary._tf import summary
  _current_module.__path__ = (
      [_module_util.get_parent_dir(summary)] + _current_module.__path__)
  setattr(_current_module, "summary", summary)
except ImportError:
  _logging.warning(
      "Limited tf.summary API due to missing TensorBoard installation.")
if (_os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == 'true' or
    _os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == '1'):
  import tensorflow_io_gcs_filesystem as _tensorflow_io_gcs_filesystem
_estimator_module = "tensorflow_estimator.python.estimator.api._v2.estimator"
estimator = _LazyLoader("estimator", globals(), _estimator_module)
_module_dir = _module_util.get_parent_dir_for_name(_estimator_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "estimator", estimator)
_keras_module = "keras.api._v2.keras"
keras = _LazyLoader("keras", globals(), _keras_module)
_module_dir = _module_util.get_parent_dir_for_name(_keras_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "keras", keras)
if _typing.TYPE_CHECKING:
  from tensorflow_estimator.python.estimator.api._v2 import estimator
_compat.enable_v2_behavior()
_major_api_version = 2
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi
_site_packages_dirs = []
if _site.ENABLE_USER_SITE and _site.USER_SITE is not None:
  _site_packages_dirs += [_site.USER_SITE]
_site_packages_dirs += [_p for _p in _sys.path if 'site-packages' in _p]
if 'getsitepackages' in dir(_site):
  _site_packages_dirs += _site.getsitepackages()
if 'sysconfig' in dir(_distutils):
  _site_packages_dirs += [_distutils.sysconfig.get_python_lib()]
_site_packages_dirs = list(set(_site_packages_dirs))
_current_file_location = _inspect.getfile(_inspect.currentframe())
def _running_from_pip_package():
  return any(
      _current_file_location.startswith(dir_) for dir_ in _site_packages_dirs)
if _running_from_pip_package():
  _tf_dir = _os.path.dirname(_current_file_location)
  _kernel_dir = _os.path.join(_tf_dir, 'core', 'kernels')
  if _os.path.exists(_kernel_dir):
    _ll.load_library(_kernel_dir)
  for _s in _site_packages_dirs:
    _plugin_dir = _os.path.join(_s, 'tensorflow-plugins')
    if _os.path.exists(_plugin_dir):
      _ll.load_library(_plugin_dir)
      _ll.load_pluggable_device_library(_plugin_dir)
if hasattr(_current_module, 'keras'):
  try:
    _keras_package = "keras.api._v2.keras."
    losses = _LazyLoader("losses", globals(), _keras_package + "losses")
    metrics = _LazyLoader("metrics", globals(), _keras_package + "metrics")
    optimizers = _LazyLoader(
        "optimizers", globals(), _keras_package + "optimizers")
    initializers = _LazyLoader(
        "initializers", globals(), _keras_package + "initializers")
    setattr(_current_module, "losses", losses)
    setattr(_current_module, "metrics", metrics)
    setattr(_current_module, "optimizers", optimizers)
    setattr(_current_module, "initializers", initializers)
  except ImportError:
    pass
if hasattr(_current_module, "keras"):
  try:
    keras._load()
  except ImportError:
    pass
try:
  del python
except NameError:
  pass
try:
  del core
except NameError:
  pass
try:
  del compiler
except NameError:
  pass
