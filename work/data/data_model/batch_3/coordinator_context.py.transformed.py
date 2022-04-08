
import contextlib
import threading
from tensorflow.python.util.lazy_loader import LazyLoader
cluster_coordinator = LazyLoader(
    "cluster_coordinator", globals(),
    "tensorflow.python.distribute.coordinator.cluster_coordinator"
)
_dispatch_context = threading.local()
def get_current_dispatch_context():
  try:
    return _dispatch_context.current
  except AttributeError:
    return None
@contextlib.contextmanager
def with_dispatch_context(worker_obj):
  previous_context = getattr(_dispatch_context, "current", None)
  _dispatch_context.current = DispatchContext(worker_obj)
  yield
  _dispatch_context.current = previous_context
class DispatchContext(object):
  def __init__(self, worker_obj):
    self._worker = worker_obj
    self._worker_index = worker_obj.worker_index
  @property
  def worker(self):
    return self._worker
  @property
  def worker_index(self):
    return self._worker_index
  def maybe_rebuild_remote_values(self, remote_value):
    e = (
            self._worker, remote_value))
    if e:
      if not isinstance(e, cluster_coordinator.InputError):
        e = cluster_coordinator.InputError(e)
      raise e
  def maybe_get_remote_value(self, ret):
