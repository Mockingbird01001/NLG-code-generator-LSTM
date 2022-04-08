
from tensorflow_estimator.python.estimator.canned import dnn_linear_combined
_HAS_DYNAMIC_ATTRIBUTES = True
dnn_linear_combined.__all__ = [
    s for s in dir(dnn_linear_combined) if not s.startswith('__')
]
from tensorflow_estimator.python.estimator.canned.dnn_linear_combined import *
