
from tensorflow_estimator.python.estimator.canned import metric_keys
_HAS_DYNAMIC_ATTRIBUTES = True
metric_keys.__all__ = [s for s in dir(metric_keys) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.canned.metric_keys import *
