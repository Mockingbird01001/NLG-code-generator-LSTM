
from tensorflow_estimator.python.estimator.inputs.queues import feeding_functions
_HAS_DYNAMIC_ATTRIBUTES = True
feeding_functions.__all__ = [
    s for s in dir(feeding_functions) if not s.startswith('__')
]
from tensorflow_estimator.python.estimator.inputs.queues.feeding_functions import *
