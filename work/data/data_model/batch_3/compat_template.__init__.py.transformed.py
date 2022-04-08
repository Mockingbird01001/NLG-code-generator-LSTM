
import logging as _logging
import os as _os
import sys as _sys
import typing as _typing
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
_current_module = _sys.modules[__name__]
try:
  from tensorboard.summary._tf import summary
  _current_module.__path__ = (
      [_module_util.get_parent_dir(summary)] + _current_module.__path__)
  setattr(_current_module, "summary", summary)
except ImportError:
  _logging.warning(
      "Limited tf.compat.v2.summary API due to missing TensorBoard "
      "installation.")
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
setattr(_current_module, "enable_v2_behavior", enable_v2_behavior)
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
