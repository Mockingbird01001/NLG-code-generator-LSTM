
import collections
import enum
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.python.util import tf_inspect
Example = collections.namedtuple('Example', ['arg', 'out', 'failure', 'bugs'])
class RunMode(enum.Enum):
  RAW = 0
  FUNCTION = 1
  SAVED = 2
  XLA = 3
dashboard_data = {}
class ConsistencyTestBase(tf.test.TestCase):
  def recordProperty(self, property_name, property_value):
    base = super(ConsistencyTestBase, self)
    if hasattr(base, 'recordProperty'):
      getattr(base, 'recordProperty')(property_name, property_value)
  def _deep_equal(self, left, right):
    if isinstance(left, tf.TensorArray):
      return self._deep_equal(left.stack(), right)
    if isinstance(right, tf.TensorArray):
      return self._deep_equal(left, right.stack())
    if isinstance(left, tf.Tensor):
      return self._deep_equal(left.numpy(), right)
    if isinstance(right, tf.Tensor):
      return self._deep_equal(left, right.numpy())
    if isinstance(left, tf.SparseTensor) and isinstance(right, tf.SparseTensor):
      return (self._deep_equal(left.indices, right.indices)
              and self._deep_equal(left.values, right.values)
              and self._deep_equal(left.shape, right.shape))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
      return np.array_equal(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
      return all(self._deep_equal(l, r) for l, r in zip(left, right))
    return left == right
  def _run_and_check(self, f, mode, examples):
    for arg, out, failure, bugs in examples:
      del bugs
      err_msg = '.*'
      if isinstance(failure, dict):
        if mode in failure.keys():
          err_msg = failure[mode]
        failure = failure.keys()
      if mode in failure:
        with self.assertRaisesWithPredicateMatch(BaseException, err_msg):
          self._deep_equal(f(*arg), out)
      else:
        self.assertTrue(self._deep_equal(f(*arg), out))
  def _generic_test(self,
                    f_raw,
                    examples,
                    input_signature=None,
                    skip_modes=None):
    """Test a function `f_raw` against all tests `examples`.
    Args:
      f_raw: a callable.
      examples: A list of `Example` named tuples.
      input_signature: Input signature to tf.function.
      skip_modes: A list of `RunMode` enums to entirely skip testing in the
        specified `RunMode`s. This is necessary when things fail in a certain
        `RunMode` even before executing the function (e.g. during saving or
        loading in `RunMode.SAVED` mode).
    """
    f_tf = None
    if not skip_modes:
      skip_modes = []
    if tf_inspect.isfunction(f_raw):
      self.recordProperty('f', tf_inspect.getsource(f_raw))
    else:
      self.recordProperty('f', tf_inspect.getdoc(f_raw))
    for arg, out, failure, bugs in examples:
      del out
      self.recordProperty('Input "{}"'.format(arg), {
          'not-working': failure,
          'bugs': bugs
      })
    if RunMode.RAW not in skip_modes:
      self._run_and_check(f_raw, RunMode.RAW, examples)
    if RunMode.FUNCTION not in skip_modes:
      f_tf = tf.function(f_raw, input_signature=input_signature)
      self._run_and_check(f_tf, RunMode.FUNCTION, examples)
    if RunMode.XLA not in skip_modes:
      f_xla = tf.function(
          f_raw, input_signature=input_signature, experimental_compile=True)
      self._run_and_check(f_xla, RunMode.XLA, examples)
    if RunMode.SAVED not in skip_modes:
      module = tf.Module()
      if f_tf:
        module.f = f_tf
      else:
        module.f = tf.function(f_raw, input_signature=input_signature)
      saved_model_dir = tempfile.gettempdir()
      tf.saved_model.save(module, saved_model_dir)
      module_loaded = tf.saved_model.load(saved_model_dir)
      self._run_and_check(module_loaded.f, RunMode.SAVED, examples)
if __name__ == '__main__':
  tf.test.main()
