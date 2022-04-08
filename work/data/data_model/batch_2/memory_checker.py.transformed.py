
from tensorflow.python.framework.python_memory_checker import _PythonMemoryChecker
from tensorflow.python.profiler import trace
from tensorflow.python.util import tf_inspect
try:
except ImportError:
  CppMemoryChecker = None
def _get_test_name_best_effort():
  for stack in tf_inspect.stack():
    function_name = stack[3]
    if function_name.startswith('test'):
      try:
        class_name = stack[0].f_locals['self'].__class__.__name__
        return class_name + '.' + function_name
        pass
  return None
class MemoryChecker(object):
  """Memory leak detection class.
  This is a utility class to detect Python and C++ memory leaks. It's intended
  for both testing and debugging. Basic usage:
  >>> with MemoryChecker() as memory_checker:
  >>>   tensors = []
  >>>   for _ in range(10):
  >>>     tensors.append(tf.constant(1))
  >>>
  >>>     memory_checker.record_snapshot()
  >>>
  >>> memory_checker.report()
  >>>
  >>> memory_checker.assert_no_leak_if_all_possibly_except_one()
  `record_snapshot()` must be called once every iteration at the same location.
  This is because the detection algorithm relies on the assumption that if there
  is a leak, it's happening similarly on every snapshot.
  """
  @trace.trace_wrapper
  def __enter__(self):
    self._python_memory_checker = _PythonMemoryChecker()
    if CppMemoryChecker:
      self._cpp_memory_checker = CppMemoryChecker(_get_test_name_best_effort())
    return self
  @trace.trace_wrapper
  def __exit__(self, exc_type, exc_value, traceback):
    if CppMemoryChecker:
      self._cpp_memory_checker.stop()
  def record_snapshot(self):
    """Take a memory snapshot for later analysis.
    `record_snapshot()` must be called once every iteration at the same
    location. This is because the detection algorithm relies on the assumption
    that if there is a leak, it's happening similarly on every snapshot.
    The recommended number of `record_snapshot()` call depends on the testing
    code complexity and the allcoation pattern.
    """
    self._python_memory_checker.record_snapshot()
    if CppMemoryChecker:
      self._cpp_memory_checker.record_snapshot()
  @trace.trace_wrapper
  def report(self):
    self._python_memory_checker.report()
    if CppMemoryChecker:
      self._cpp_memory_checker.report()
  @trace.trace_wrapper
  def assert_no_leak_if_all_possibly_except_one(self):
    """Raises an exception if a leak is detected.
    This algorithm classifies a series of allocations as a leak if it's the same
    type(Python) orit happens at the same stack trace(C++) at every snapshot,
    but possibly except one snapshot.
    """
    self._python_memory_checker.assert_no_leak_if_all_possibly_except_one()
    if CppMemoryChecker:
      self._cpp_memory_checker.assert_no_leak_if_all_possibly_except_one()
  @trace.trace_wrapper
  def assert_no_new_python_objects(self, threshold=None):
    self._python_memory_checker.assert_no_new_objects(threshold=threshold)
