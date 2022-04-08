
import contextlib
import threading
class LoadContext(threading.local):
  def __init__(self):
    super(LoadContext, self).__init__()
    self._entered_load_context = []
    self._load_options = None
  def set_load_options(self, load_options):
    self._load_options = load_options
    self._entered_load_context.append(True)
  def clear_load_options(self):
    self._load_options = None
    self._entered_load_context.pop()
  def load_options(self):
    return self._load_options
  def in_load_context(self):
    return self._entered_load_context
_load_context = LoadContext()
@contextlib.contextmanager
def load_context(load_options):
  _load_context.set_load_options(load_options)
  try:
    yield
  finally:
    _load_context.clear_load_options()
def get_load_options():
  return _load_context.load_options()
def in_load_context():
  return _load_context.in_load_context
