
from tensorflow_estimator.python.estimator.inputs import numpy_io
_HAS_DYNAMIC_ATTRIBUTES = True
numpy_io.__all__ = [s for s in dir(numpy_io) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.inputs.numpy_io import *
