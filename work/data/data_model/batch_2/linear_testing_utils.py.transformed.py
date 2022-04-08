
from tensorflow_estimator.python.estimator.canned import linear_testing_utils
_HAS_DYNAMIC_ATTRIBUTES = True
linear_testing_utils.__all__ = [
    s for s in dir(linear_testing_utils) if not s.startswith('__')
]
from tensorflow_estimator.python.estimator.canned.linear_testing_utils import *
