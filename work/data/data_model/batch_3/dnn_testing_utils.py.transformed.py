
from tensorflow_estimator.python.estimator.canned import dnn_testing_utils
_HAS_DYNAMIC_ATTRIBUTES = True
dnn_testing_utils.__all__ = [
    s for s in dir(dnn_testing_utils) if not s.startswith('__')
]
from tensorflow_estimator.python.estimator.canned.dnn_testing_utils import *
