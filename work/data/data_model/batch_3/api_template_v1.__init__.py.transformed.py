
import distutils as _distutils
import inspect as _inspect
import os as _os
import site as _site
import sys as _sys
import typing as _typing
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.platform import tf_logging as _logging
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
  _logging.warning("""
  TensorFlow's `tf-nightly` package will soon be updated to TensorFlow 2.0.
  Please upgrade your code to TensorFlow 2.0:
    * https://www.tensorflow.org/guide/migrate
  Or install the latest stable TensorFlow 1.X release:
    * `pip install -U "tensorflow==1.*"`
  Otherwise your code may be broken by the change.
  """)
_current_module = _sys.modules[__name__]
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, '__path__'):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)
if (_os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == 'true' or
    _os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == '1'):
  import tensorflow_io_gcs_filesystem as _tensorflow_io_gcs_filesystem
_estimator_module = "tensorflow_estimator.python.estimator.api._v1.estimator"
estimator = _LazyLoader("estimator", globals(), _estimator_module)
_module_dir = _module_util.get_parent_dir_for_name(_estimator_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "estimator", estimator)
_keras_module = "keras.api._v1.keras"
keras = _LazyLoader("keras", globals(), _keras_module)
_module_dir = _module_util.get_parent_dir_for_name(_keras_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "keras", keras)
if _typing.TYPE_CHECKING:
  from tensorflow_estimator.python.estimator.api._v1 import estimator
_CONTRIB_WARNING = """
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.
"""
contrib = LazyLoader('contrib', globals(), 'tensorflow.contrib',
                     _CONTRIB_WARNING)
del LazyLoader
if '__all__' in vars():
  vars()['__all__'].append('contrib')
setattr(_current_module, "flags", flags)
_major_api_version = 1
if hasattr(_current_module, "keras"):
  try:
    _layer_package = "keras.api._v1.keras.__internal__.legacy.layers"
    layers = _LazyLoader("layers", globals(), _layer_package)
    _module_dir = _module_util.get_parent_dir_for_name(_layer_package)
    if _module_dir:
      _current_module.__path__ = [_module_dir] + _current_module.__path__
    setattr(_current_module, "layers", layers)
    _legacy_rnn_package = "keras.api._v1.keras.__internal__.legacy.rnn_cell"
    _rnn_cell = _LazyLoader("legacy_rnn", globals(), _legacy_rnn_package)
    _module_dir = _module_util.get_parent_dir_for_name(_legacy_rnn_package)
    if _module_dir:
      _current_module.nn.__path__ = [_module_dir] + _current_module.nn.__path__
    _current_module.nn.rnn_cell = _rnn_cell
  except ImportError:
    pass
if hasattr(_current_module, "keras"):
  try:
    keras._load()
  except ImportError:
    pass
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi
_site_packages_dirs = []
_site_packages_dirs += [] if _site.USER_SITE is None else [_site.USER_SITE]
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
