
from tensorflow_estimator.python.estimator import estimator_lib
_HAS_DYNAMIC_ATTRIBUTES = True
estimator_lib.__all__ = [
    s for s in dir(estimator_lib) if not s.startswith('__')
]
from tensorflow_estimator.python.estimator.estimator_lib import *
