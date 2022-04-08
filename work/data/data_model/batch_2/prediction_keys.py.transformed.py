
from tensorflow_estimator.python.estimator.canned import prediction_keys
_HAS_DYNAMIC_ATTRIBUTES = True
prediction_keys.__all__ = [
    s for s in dir(prediction_keys) if not s.startswith('__')
]
from tensorflow_estimator.python.estimator.canned.prediction_keys import *
