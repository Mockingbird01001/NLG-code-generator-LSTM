
from tensorflow_estimator.python.estimator import model_fn
_HAS_DYNAMIC_ATTRIBUTES = True
model_fn.__all__ = [s for s in dir(model_fn) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.model_fn import *
