
from tensorflow_estimator.python.estimator.canned import dnn
_HAS_DYNAMIC_ATTRIBUTES = True
dnn.__all__ = [s for s in dir(dnn) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.canned.dnn import *
