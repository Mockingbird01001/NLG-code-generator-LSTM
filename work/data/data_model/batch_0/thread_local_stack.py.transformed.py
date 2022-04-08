
import threading
class ThreadLocalStack(threading.local):
  def __init__(self):
    super(ThreadLocalStack, self).__init__()
    self._stack = []
  def peek(self):
    return self._stack[-1] if self._stack else None
  def push(self, ctx):
    return self._stack.append(ctx)
  def pop(self):
    self._stack.pop()
