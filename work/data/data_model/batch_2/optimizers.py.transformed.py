
from tensorflow_estimator.python.estimator.canned import optimizers
_HAS_DYNAMIC_ATTRIBUTES = True
optimizers.__all__ = [s for s in dir(optimizers) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.canned.optimizers import *
