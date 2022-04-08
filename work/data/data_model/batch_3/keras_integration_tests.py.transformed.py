
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.tools.consistency_integration_test.consistency_test_base import ConsistencyTestBase
class KerasIntegrationTest(ConsistencyTestBase, parameterized.TestCase):
  @parameterized.named_parameters([('_RunFunctionEagerly', True),
                                   ('_RunFunctionNonEagerly', False)])
  def testVariableCreationKerasLayers(self, run_eagerly):
    """Tests tf.function variable creation in Keras layers.
    Bugs:   b/184210116
    Status: Known issue
            (However, moving forward, we should support re-creating
            `tf.Variables` inside tf.function for each trace. This test case
            should pass eventually.)
    Issue:  `tf.Variable` creation in Keras layers causes 'non-first call
            variable creation' error in a tf.function.
    Error message:
      "tf.function-decorated function tried to create variables on non-first
      call."
    Improve error message? Needed. (b/187847612)
    Notes:
    * Inconsistent behavior between eager and non-eager mode execution of the
      tf.function.
    * In non-eager mode (graph mode), double tracing (i.e. first one during
      function tracing and second one in execution) causes variable creation in
      non-first call error.
    * This is an expected behavior as Keras's Dense layer creates variables.
    * go/tf-mutable-refs is a work-in-progress, longer term project designed to
      address this issue.
    Args:
      run_eagerly: Boolean deciding whether to run tf.function decorated
        functions eagerly or not.
    """
    self.skipTest('b/184210116')
    try:
      original_setting = tf.config.functions_run_eagerly()
      tf.config.run_functions_eagerly(run_eagerly)
      @tf.function
      def f(x):
        layer = tf.keras.layers.Dense(2)(x)
        return layer
      if run_eagerly:
        self.assertAllEqual(
            tf.constant([[0.7891873, -0.5761101], [1.7832438, -1.6489036]],
                        dtype=tf.float32), f(tf.constant([[1., 2.], [3., 4.]])))
      else:
        f(tf.constant([[1., 2.], [3., 4.]]))
    finally:
      tf.config.run_functions_eagerly(original_setting)
  def testVariableCreationKerasLayersRecommended(self):
    layer = None
    @tf.function
    def f(x):
      nonlocal layer
      if layer is None:
        layer = tf.keras.layers.Dense(2)
      return layer(x)
    self.assertAllEqual(
        f(tf.constant([[1., 2.], [3., 4.]])),
        tf.constant([[0.7891873, -0.5761101], [1.7832438, -1.6489036]]))
  @parameterized.named_parameters([('_RunFunctionEagerly', True),
                                   ('_RunFunctionNonEagerly', False)])
  def testRetracingKerasOptimAsPythonObj(self, run_eagerly):
    """Tests tf.function variable creation in Keras optimizers.
    Bugs:   b/184210116
    Status: Working as intended
            (However, moving forward, we should support re-creating
            `tf.Variables` inside tf.function for each trace. This test case
            should pass eventually.)
    Issue:  Passing in different Keras optimizers (Python objects) to
            tf.function is not allowed as they create
            `tf.Variable`s and will result in 'non-first call variable creation'
            error.
    Error message:
      "tf.function-decorated function tried to create variables on non-first
      call."
    Notes:
    * Inconsistent behavior between eager and non-eager mode execution of the
      tf.function.
    * go/tf-mutable-refs is a work-in-progress, longer term project designed to
      address this issue.
      `train_one_step()` call) when two is generally expected. Why?
      Answer: First `train_one_step` call traces twice because, after the first
      trace, tf.function detects `tf.Variable` creation and immediately traces a
      second time to see whether new variables are being created. (This has
      been a common source of confusion.)
    Args:
      run_eagerly: Boolean deciding whether to run tf.function decorated
        functions eagerly or not.
    """
    self.skipTest('b/184210116')
    try:
      original_setting = tf.config.functions_run_eagerly()
      tf.config.run_functions_eagerly(run_eagerly)
      trace = []
      @tf.function
      def train_one_step(a, x, y, optim):
        nonlocal trace
        with tf.GradientTape() as tape:
          l = tf.reduce_sum(tf.square(a * x - y))
        w = [a]
        g = tape.gradient(l, w)
        optim.apply_gradients(zip(g, w))
        return a
      optim0 = tf.keras.optimizers.Adam()
      optim1 = tf.keras.optimizers.Adam()
      a = tf.Variable(2.)
      x = tf.Variable([-1., -1., -1.])
      y = tf.Variable([2., 2., 2.])
      tf.config.run_functions_eagerly(run_eagerly)
      if run_eagerly:
        train_one_step(a, x, y, optim1)
      else:
        train_one_step(a, x, y, optim1)
    finally:
      tf.config.run_functions_eagerly(original_setting)
  def testCachedTensorKerasLayers(self):
    """Tests tf.function I/O behavior with cached tensors in Keras layers.
    Bugs:   b/149094965
    Status: Working as intended
    Issue:  When there exists a trace that has cached tensors, retracing the
            function (upon receiving new input signature) will result in an
            error as the cached tensor is from the previous trace.
    Error message:
      "The tensor 'Tensor("Placeholder:0", shape=(None, 1), dtype=float32)'
      cannot be accessed here: it is defined in another function or code block."
    Notes:
    * `self._cached_value` is already a cached tensor when the program tries to
      retrace upon calling `model.fit()`.
    * This test is equivalent to `testCachedTensor` test case but just with
      Keras layers.
    * Calling custom Keras layer initially with
      `pred_out = layer(tf.constant(1.0))` as input should cache
      `self._cached_value` as tensor, leading to an error upon calling
      `model.fit()` with a different input signature. However, commenting out
      the first step does not have any effect. Why? Left a TODO.
    """
    self.skipTest('b/149094965')
    class Context(object):
      def __init__(self):
        self._cached_value = None
      def f(self, x):
        result = x + 1
        if self._cached_value is not None:
          result += self._cached_value
        self._cached_value = x
        return result
    class CustomLayer(tf.keras.layers.Layer):
      def __init__(self, context, **kwargs):
        self.context = context
        super(CustomLayer, self).__init__(**kwargs)
      def call(self, x, training=None):
        return self.context.f(x)
    ctx = Context()
    layer = CustomLayer(ctx)
    model = tf.keras.models.Sequential([layer])
    model.compile('sgd', 'mean_squared_error')
    model.fit(tf.constant([1., 2., 3.]), tf.constant([1., 2., 3.]))
if __name__ == '__main__':
  test.main()
