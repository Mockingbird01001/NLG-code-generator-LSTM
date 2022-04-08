
from tensorflow_estimator.python.estimator import estimator
_HAS_DYNAMIC_ATTRIBUTES = True
estimator.__all__ = [s for s in dir(estimator) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.estimator import *
