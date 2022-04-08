
from tensorflow_estimator.python.estimator.inputs.queues import feeding_queue_runner
_HAS_DYNAMIC_ATTRIBUTES = True
feeding_queue_runner.__all__ = [
    s for s in dir(feeding_queue_runner) if not s.startswith('__')
]
from tensorflow_estimator.python.estimator.inputs.queues.feeding_queue_runner import *
