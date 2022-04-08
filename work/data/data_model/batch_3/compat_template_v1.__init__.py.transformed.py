
import os as _os
import sys as _sys
import typing as _typing
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
_current_module = _sys.modules[__name__]
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
setattr(_current_module, "flags", flags)
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
