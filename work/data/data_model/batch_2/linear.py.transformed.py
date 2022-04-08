
from tensorflow_estimator.python.estimator.canned import linear
_HAS_DYNAMIC_ATTRIBUTES = True
linear.__all__ = [s for s in dir(linear) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.canned.linear import *
