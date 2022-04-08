
from tensorflow_estimator.python.estimator.inputs import inputs
_HAS_DYNAMIC_ATTRIBUTES = True
inputs.__all__ = [s for s in dir(inputs) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.inputs.inputs import *
