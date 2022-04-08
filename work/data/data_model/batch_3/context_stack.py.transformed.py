
import contextlib
from tensorflow.python.framework.experimental import thread_local_stack
_default_ctx_stack = thread_local_stack.ThreadLocalStack()
def get_default():
  return _default_ctx_stack.peek()
@contextlib.contextmanager
def set_default(ctx):
  try:
    _default_ctx_stack.push(ctx)
    yield
  finally:
    _default_ctx_stack.pop()
