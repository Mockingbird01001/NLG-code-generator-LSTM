
import threading
class GroupLock(object):
  """A lock to allow many members of a group to access a resource exclusively.
  This lock provides a way to allow access to a resource by multiple threads
  belonging to a logical group at the same time, while restricting access to
  threads from all other groups. You can think of this as an extension of a
  reader-writer lock, where you allow multiple writers at the same time. We
  made it generic to support multiple groups instead of just two - readers and
  writers.
  Simple usage example with two groups accessing the same resource:
  ```python
  lock = GroupLock(num_groups=2)
  with lock.group(0):
  with lock.group(1):
  ```
  Using as a context manager with `.group(group_id)` is the easiest way. You
  can also use the `acquire` and `release` method directly.
  """
  __slots__ = ["_ready", "_num_groups", "_group_member_counts"]
  def __init__(self, num_groups=2):
    if num_groups < 1:
      raise ValueError(
          "Argument `num_groups` must be a positive integer. "
          f"Received: num_groups={num_groups}")
    self._ready = threading.Condition(threading.Lock())
    self._num_groups = num_groups
    self._group_member_counts = [0] * self._num_groups
  def group(self, group_id):
    self._validate_group_id(group_id)
    return self._Context(self, group_id)
  def acquire(self, group_id):
    self._validate_group_id(group_id)
    self._ready.acquire()
    while self._another_group_active(group_id):
      self._ready.wait()
    self._group_member_counts[group_id] += 1
    self._ready.release()
  def release(self, group_id):
    self._validate_group_id(group_id)
    self._ready.acquire()
    self._group_member_counts[group_id] -= 1
    if self._group_member_counts[group_id] == 0:
      self._ready.notify_all()
    self._ready.release()
  def _another_group_active(self, group_id):
    return any(
        c > 0 for g, c in enumerate(self._group_member_counts) if g != group_id)
  def _validate_group_id(self, group_id):
    if group_id < 0 or group_id >= self._num_groups:
      raise ValueError(
          "Argument `group_id` should verify `0 <= group_id < num_groups` "
          f"(with `num_groups={self._num_groups}`). "
          f"Received: group_id={group_id}")
  class _Context(object):
    __slots__ = ["_lock", "_group_id"]
    def __init__(self, lock, group_id):
      self._lock = lock
      self._group_id = group_id
    def __enter__(self):
      self._lock.acquire(self._group_id)
    def __exit__(self, type_arg, value_arg, traceback_arg):
      del type_arg, value_arg, traceback_arg
      self._lock.release(self._group_id)
