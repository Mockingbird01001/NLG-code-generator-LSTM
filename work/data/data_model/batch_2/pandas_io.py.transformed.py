
from tensorflow_estimator.python.estimator.inputs import pandas_io
_HAS_DYNAMIC_ATTRIBUTES = True
pandas_io.__all__ = [s for s in dir(pandas_io) if not s.startswith('__')]
from tensorflow_estimator.python.estimator.inputs.pandas_io import *
