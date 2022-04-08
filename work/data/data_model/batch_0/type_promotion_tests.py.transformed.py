
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.tools.consistency_integration_test.consistency_test_base import ConsistencyTestBase
from tensorflow.tools.consistency_integration_test.consistency_test_base import Example
from tensorflow.tools.consistency_integration_test.consistency_test_base import RunMode
class TypePromotionTests(ConsistencyTestBase):
  def testFloatingPointPrecision(self):
    """Tests inconsistent floating point precision between eager vs. graph.
    Bugs:   b/187097409
    Status: Inconsistent floating point precision
    Issue:  Output returned from a function is different between when the
            function is decorated with tf.function or not. Running the
            tf.function in XLA mode also is inconsistent with running the
            function in RAW mode (i.e. running tf.function eagerly).
    Notes:
    * This behavior is consistent with the tensor wrapping rules (i.e.
      `tf.constant`s are taken as `tf.float32` by default) but requires further
      discussion for achieving better consistency.
    * For getting consistent results back, the suggestion is to explicitly
      construct tensors for inputs. See the test case below that passes
      `tf.constant(3.2)` as `arg`.
    """
    def f(x):
      return x * x
    self._generic_test(
        f, [
            Example(arg=(3.2,), out=10.240000000000002, failure=[], bugs=[]),
        ],
        skip_modes=[RunMode.FUNCTION, RunMode.XLA, RunMode.SAVED])
    self._generic_test(
        f, [
            Example(arg=(3.2,), out=10.239999771118164, failure=[], bugs=[]),
        ],
        skip_modes=[RunMode.RAW])
    self._generic_test(
        f, [
            Example(
                arg=(tf.constant(3.2),),
                out=10.24000072479248,
                failure=[],
                bugs=[]),
        ],
        skip_modes=[])
if __name__ == '__main__':
  test.main()
