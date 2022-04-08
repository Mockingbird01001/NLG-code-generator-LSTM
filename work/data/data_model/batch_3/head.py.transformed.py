
from tensorflow_estimator.python.estimator.canned import head
_HAS_DYNAMIC_ATTRIBUTES = True
head.__all__ = [s for s in dir(head) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.canned.head import *
