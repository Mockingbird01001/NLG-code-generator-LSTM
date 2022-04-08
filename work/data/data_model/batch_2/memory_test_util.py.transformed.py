
import collections
import gc
import time
import six
from tensorflow.python.eager import context
try:
except ImportError:
  memory_profiler = None
def _instance_count_by_class():
  counter = collections.Counter()
  for obj in gc.get_objects():
    try:
      counter[obj.__class__.__name__] += 1
      pass
  return counter
def assert_no_leak(f, num_iters=100000, increase_threshold_absolute_mb=10):
  with context.eager_mode():
    f()
    time.sleep(4)
    gc.collect()
    initial = memory_profiler.memory_usage(-1)[0]
    instance_count_by_class_before = _instance_count_by_class()
    for _ in six.moves.range(num_iters):
      f()
    gc.collect()
    increase = memory_profiler.memory_usage(-1)[0] - initial
    assert increase < increase_threshold_absolute_mb, (
        "Increase is too high. Initial memory usage: %f MB. Increase: %f MB. "
        "Maximum allowed increase: %f MB. "
        "Instance count diff before/after: %s") % (
            initial, increase, increase_threshold_absolute_mb,
            _instance_count_by_class() - instance_count_by_class_before)
def memory_profiler_is_available():
  return memory_profiler is not None
