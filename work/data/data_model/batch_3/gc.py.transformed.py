
from tensorflow_estimator.python.estimator import gc
_HAS_DYNAMIC_ATTRIBUTES = True
gc.__all__ = [s for s in dir(gc) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.gc import *
