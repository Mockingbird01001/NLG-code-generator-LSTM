
from tensorflow_estimator.python.estimator.canned import parsing_utils
_HAS_DYNAMIC_ATTRIBUTES = True
parsing_utils.__all__ = [
    s for s in dir(parsing_utils) if not s.startswith('__')
]
from tensorflow_estimator.python.estimator.canned.parsing_utils import *
