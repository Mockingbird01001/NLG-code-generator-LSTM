
from tensorflow_estimator.python.estimator.canned import baseline
_HAS_DYNAMIC_ATTRIBUTES = True
baseline.__all__ = [s for s in dir(baseline) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.canned.baseline import *
