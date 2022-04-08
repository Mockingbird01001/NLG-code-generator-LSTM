
import contextlib as _contextlib
from tensorflow.python.util import tf_decorator
def contextmanager(target):
  context_manager = _contextlib.contextmanager(target)
  return tf_decorator.make_decorator(target, context_manager, 'contextmanager')
