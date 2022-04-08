
import threading
_worker_context = threading.local()
def get_current_worker_context():
  try:
    return _worker_context.current
  except AttributeError:
    return None
