
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
class GradientDescentOptimizerTest(test.TestCase, parameterized.TestCase):
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasic(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      var0 = variables.Variable([1.0, 2.0], dtype=dtype)
      var1 = variables.Variable([3.0, 4.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
      sgd = gradient_descent.SGD(3.0)
      sgd_op = sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(sgd_op)
      self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                         self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                         self.evaluate(var1))
  def _test_basic_sgd_with_learning_rate_decay(self, sgd, dtype):
    var0 = variables.Variable([1.0, 2.0], dtype=dtype)
    var1 = variables.Variable([3.0, 4.0], dtype=dtype)
    grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
    grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
    if not context.executing_eagerly():
      sgd_op = sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.evaluate(variables.global_variables_initializer())
    if not context.executing_eagerly():
      self.evaluate(sgd_op)
    else:
      sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                       self.evaluate(var0))
    self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                       self.evaluate(var1))
    if not context.executing_eagerly():
      self.evaluate(sgd_op)
    else:
      sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.assertAllCloseAccordingToType(
        [1.0 - 3.0 * 0.1 - 2.0 * 0.1, 2.0 - 3.0 * 0.1 - 2.0 * 0.1],
        self.evaluate(var0))
    self.assertAllCloseAccordingToType(
        [3.0 - 3.0 * 0.01 - 2.0 * 0.01, 4.0 - 3.0 * 0.01 - 2.0 * 0.01],
        self.evaluate(var1))
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasicWithLearningRateDecay(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      learning_rate = 3.0
      decay = 0.5
      sgd = gradient_descent.SGD(learning_rate=learning_rate, decay=decay)
      self._test_basic_sgd_with_learning_rate_decay(sgd, dtype)
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasicWithLearningRateInverseTimeDecay(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      learning_rate = learning_rate_schedule.InverseTimeDecay(
          3.0, decay_steps=1.0, decay_rate=0.5)
      sgd = gradient_descent.SGD(learning_rate=learning_rate)
      self._test_basic_sgd_with_learning_rate_decay(sgd, dtype)
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasicWithLearningRateInverseTimeDecaySerializeAndDeserialize(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      learning_rate = learning_rate_schedule.InverseTimeDecay(
          3.0, decay_steps=1.0, decay_rate=0.5)
      sgd = gradient_descent.SGD(learning_rate=learning_rate)
      sgd = gradient_descent.SGD.from_config(sgd.get_config())
      self._test_basic_sgd_with_learning_rate_decay(sgd, dtype)
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasicCallableParams(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      var0 = variables.Variable([1.0, 2.0], dtype=dtype)
      var1 = variables.Variable([3.0, 4.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
      lr = lambda: 3.0
      sgd = gradient_descent.SGD(lr)
      sgd_op = sgd.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(sgd_op)
      self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                         self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                         self.evaluate(var1))
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testMinimizeResourceVariable(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      var0 = variables.Variable([[1.0, 2.0]], dtype=dtype)
      var1 = variables.Variable([3.0], dtype=dtype)
      x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
      sgd = gradient_descent.SGD(1.0)
      sgd_op = sgd.minimize(loss, [var0, var1])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(sgd_op)
      self.assertAllCloseAccordingToType([[1.0 - 4.0, 2.0 - 5.0]],
                                         self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0 - 1.0], self.evaluate(var1))
  def testMinimizeSparseResourceVariable(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        var0 = variables.Variable([[1.0, 2.0]], dtype=dtype)
        var1 = variables.Variable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        def loss():
          return pred * pred
        sgd_op = gradient_descent.SGD(1.0).minimize(loss, [var0, var1])
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(sgd_op)
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))
  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      var0 = variables.Variable([1.0, 2.0], dtype=dtype)
      var1 = variables.Variable([3.0, 4.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
      lrate = constant_op.constant(3.0)
      sgd_op = gradient_descent.SGD(lrate).apply_gradients(
          zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(sgd_op)
      self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                         self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                         self.evaluate(var1))
  def testGradWrtRef(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        opt = gradient_descent.SGD(3.0)
        values = [1.0, 3.0]
        vars_ = [variables.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt._compute_gradients(loss, vars_)
        self.evaluate(variables.global_variables_initializer())
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.0], self.evaluate(grad))
  def testSparseBasic(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant([0.01], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))
        sgd_op = gradient_descent.SGD(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(sgd_op)
        self.assertAllCloseAccordingToType([[1.0 - 3.0 * 0.1], [2.0]],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0 - 3.0 * 0.01]],
                                           self.evaluate(var1))
  def testSparseBasicWithLearningRateDecay(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant([0.01], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))
        sgd_op = gradient_descent.SGD(
            3.0, decay=0.5).apply_gradients(
                zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(sgd_op)
        self.assertAllCloseAccordingToType([[1.0 - 3.0 * 0.1], [2.0]],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0 - 3.0 * 0.01]],
                                           self.evaluate(var1))
        self.evaluate(sgd_op)
        self.assertAllCloseAccordingToType(
            [[1.0 - 3.0 * 0.1 - 2.0 * 0.1], [2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            [[3.0], [4.0 - 3.0 * 0.01 - 2.0 * 0.01]], self.evaluate(var1))
  @combinations.generate(combinations.combine(mode=["eager"]))
  def testCapturingInFunctionWhileExecutingEagerly(self):
    optimizer = gradient_descent.SGD(1.0)
    var_holder = {}
    def step():
      if not var_holder:
        var_holder["var"] = variables.Variable(1.0)
      else:
        var_holder["var"].assign(1.0)
      with backprop.GradientTape() as tape:
        loss = var_holder["var"]**2
      grad = tape.gradient(loss, var_holder["var"])
      optimizer.apply_gradients([(grad, var_holder["var"])])
      return var_holder["var"].read_value()
    compiled_step = def_function.function(step)
    self.assertEqual(float(step()), -1.0)
    self.assertEqual(float(compiled_step()), -1.0)
    self.assertEqual(float(step()), -1.0)
  def testConstructSGDWithLR(self):
    opt = gradient_descent.SGD(lr=1.0)
    opt_2 = gradient_descent.SGD(learning_rate=0.1, lr=1.0)
    opt_3 = gradient_descent.SGD(learning_rate=0.1)
    self.assertIsInstance(opt.lr, variables.Variable)
    self.assertIsInstance(opt_2.lr, variables.Variable)
    self.assertIsInstance(opt_3.lr, variables.Variable)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_2.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_3.lr), (0.1))
class MomentumOptimizerTest(test.TestCase, parameterized.TestCase):
  def _update_nesterov_momentum_numpy(self, var, accum, g, lr, momentum):
    accum = accum * momentum - g * lr
    var += (accum * momentum - g * lr)
    return var, accum
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasic(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      var0 = variables.Variable([1.0, 2.0], dtype=dtype, name="var0")
      var1 = variables.Variable([3.0, 4.0], dtype=dtype, name="var1")
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
      learning_rate = 2.0
      momentum = 0.9
      mom_opt = gradient_descent.SGD(
          learning_rate=learning_rate, momentum=momentum)
      mom_update = mom_opt.apply_gradients(
          zip([grads0, grads1], [var0, var1]))
      slot0 = mom_opt.get_slot(var0, "momentum")
      self.assertEqual(slot0.shape, var0.shape)
      slot1 = mom_opt.get_slot(var1, "momentum")
      self.assertEqual(slot1.shape, var1.shape)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(mom_update)
      self.assertAllCloseAccordingToType(
          np.array([-0.2, -0.2]), self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([-0.02, -0.02]), self.evaluate(slot1))
      self.assertAllCloseAccordingToType(
          np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]),
          self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
          self.evaluate(var1))
      self.evaluate(mom_update)
      if context.executing_eagerly():
        mom_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
          self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                    (0.9 * (-0.02) - 2.0 * 0.01)]), self.evaluate(slot1))
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
              2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
          ]), self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([
              2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
              3.98 - ((0.9 * 0.01 + 0.01) * 2.0)
          ]), self.evaluate(var1))
  def testNesterovMomentum(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.float32, dtypes.float64]:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype, name="var0")
        var1 = variables.Variable([3.0, 4.0], dtype=dtype, name="var1")
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom_op = gradient_descent.SGD(
            learning_rate=2.0, momentum=0.9, nesterov=True)
        opt_op = mom_op.minimize(loss, [var0, var1])
        self.evaluate(variables.global_variables_initializer())
        for _ in range(1, 5):
          self.evaluate(opt_op)
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(
              var1_np, accum1_np, 3, 2.0, 0.9)
          self.assertAllClose(var0_np, self.evaluate(var0))
          self.assertAllClose(var1_np, self.evaluate(var1))
  def testSparseNesterovMomentum(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session() as sess:
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        grads = []
        for t in range(1, 5):
          grads.append(var0_np * 10)
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(
              var1_np, accum1_np, 3, 2.0, 0.9)
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np, dtype=dtype, name="var0")
        var1 = variables.Variable(var1_np, dtype=dtype, name="var1")
        mom_op = gradient_descent.SGD(
            learning_rate=2.0, momentum=0.9, nesterov=True)
        x_feed = array_ops.placeholder(dtype)
        y_feed = indexed_slices.IndexedSlices(x_feed,
                                              constant_op.constant([0, 1]),
                                              constant_op.constant([2]))
        grads_and_vars = [(y_feed, var0),
                          (constant_op.constant([3.0, 3.0], dtype=dtype), var1)]
        opt_update = mom_op.apply_gradients(grads_and_vars)
        self.evaluate(variables.global_variables_initializer())
        for t in range(1, 5):
          sess.run(opt_update, feed_dict={x_feed: grads[t - 1]})
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(
              var1_np, accum1_np, 3, 2.0, 0.9)
          self.assertAllClose(var0_np, self.evaluate(var0))
          self.assertAllClose(var1_np, self.evaluate(var1))
  def testMinimizeSparseResourceVariable(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        var0 = variables.Variable([[1.0, 2.0]], dtype=dtype)
        def loss():
          x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
          return pred * pred
        opt = gradient_descent.SGD(learning_rate=1.0, momentum=0.9)
        sgd_op = opt.minimize(loss, [var0])
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(sgd_op)
        self.assertAllCloseAccordingToType([[-111, -138]], self.evaluate(var0))
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testMinimizeWith2DIndicesForEmbeddingLookup(self):
    var0 = variables.Variable(array_ops.ones([2, 2]))
    def loss():
      return math_ops.reduce_sum(embedding_ops.embedding_lookup(var0, [[1]]))
    opt = gradient_descent.SGD(learning_rate=1.0, momentum=0.9)
    sgd_op = opt.minimize(loss, [var0])
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(sgd_op)
    self.assertAllCloseAccordingToType([[1, 1], [0, 0]], self.evaluate(var0))
  def testTensorLearningRateAndMomentum(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        mom_opt = gradient_descent.SGD(
            learning_rate=constant_op.constant(2.0),
            momentum=constant_op.constant(0.9))
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEqual(slot0.shape, var0.shape)
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEqual(slot1.shape, var1.shape)
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        self.evaluate(mom_update)
        self.assertAllCloseAccordingToType(
            np.array([-0.2, -0.2]), self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([-0.02, -0.02]), self.evaluate(slot1))
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]),
            self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
            self.evaluate(var1))
        self.evaluate(mom_update)
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
            self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                      (0.9 * (-0.02) - 2.0 * 0.01)]), self.evaluate(slot1))
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                3.98 - ((0.9 * 0.01 + 0.01) * 2.0)
            ]), self.evaluate(var1))
  def testSparse(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        var0 = variables.Variable(array_ops.zeros([4, 2], dtype=dtype))
        var1 = variables.Variable(constant_op.constant(1.0, dtype, [4, 2]))
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant([[.1, .1]], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([4, 2]))
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant([[.01, .01], [.01, .01]], dtype=dtype),
            constant_op.constant([2, 3]), constant_op.constant([4, 2]))
        mom_opt = gradient_descent.SGD(learning_rate=2.0, momentum=0.9)
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEqual(slot0.shape, var0.shape)
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEqual(slot1.shape, var1.shape)
        self.assertAllClose([0, 0], self.evaluate(var0)[0])
        self.assertAllClose([0, 0], self.evaluate(var0)[1])
        self.assertAllClose([1, 1], self.evaluate(var1)[2])
        self.evaluate(mom_update)
        self.assertAllCloseAccordingToType(
            np.array([0, 0]),
            self.evaluate(slot0)[0])
        self.assertAllCloseAccordingToType(
            np.array([-2.0 * .1, -2.0 * .1]),
            self.evaluate(slot0)[1])
        self.assertAllCloseAccordingToType(
            np.array([-2.0 * .01, -2.0 * .01]),
            self.evaluate(slot1)[2])
        self.assertAllCloseAccordingToType(
            np.array([0, 0]),
            self.evaluate(var0)[0])
        self.assertAllCloseAccordingToType(
            np.array([-(0.1 * 2.0), -(0.1 * 2.0)]),
            self.evaluate(var0)[1])
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.01 * 2.0), 1.0 - (0.01 * 2.0)]),
            self.evaluate(var1)[2])
        self.evaluate(mom_update)
        self.assertAllClose(np.array([0, 0]), self.evaluate(slot0)[0])
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
            self.evaluate(slot0)[1])
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                      (0.9 * (-0.02) - 2.0 * 0.01)]),
            self.evaluate(slot1)[2])
        self.assertAllClose(np.array([0, 0]), self.evaluate(var0)[0])
        self.assertAllCloseAccordingToType(
            np.array([
                -(0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                -(0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]),
            self.evaluate(var0)[1])
        self.assertAllCloseAccordingToType(
            np.array([
                0.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                0.98 - ((0.9 * 0.01 + 0.01) * 2.0)
            ]),
            self.evaluate(var1)[2])
  def testSharing(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        mom_opt = gradient_descent.SGD(learning_rate=2.0, momentum=0.9)
        mom_update1 = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        mom_update2 = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEqual(slot0.shape, var0.shape)
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEqual(slot1.shape, var1.shape)
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        self.evaluate(mom_update1)
        self.assertAllCloseAccordingToType(
            np.array([-0.2, -0.2]), self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([-0.02, -0.02]), self.evaluate(slot1))
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]),
            self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
            self.evaluate(var1))
        self.evaluate(mom_update2)
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
            self.evaluate(slot0))
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                      (0.9 * (-0.02) - 2.0 * 0.01)]), self.evaluate(slot1))
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                3.98 - ((0.9 * 0.01 + 0.01) * 2.0)
            ]), self.evaluate(var1))
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testConfig(self):
    opt = gradient_descent.SGD(learning_rate=1.0, momentum=0.9, nesterov=True)
    config = opt.get_config()
    opt2 = gradient_descent.SGD.from_config(config)
    lr = opt.lr
    lr2 = opt2.lr
    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(lr), self.evaluate(lr2))
    self.assertAllClose(
        self.evaluate(opt._get_hyper("momentum")),
        self.evaluate(opt2._get_hyper("momentum")))
    self.assertAllClose(
        self.evaluate(opt._get_hyper("decay")),
        self.evaluate(opt2._get_hyper("decay")))
    var0 = variables.Variable([[1.0], [2.0]], dtype=dtypes.float32)
    loss = lambda: 3 * var0
    opt.minimize(loss, [var0])
    self.evaluate(variables.global_variables_initializer())
    config = opt.get_config()
    opt3 = gradient_descent.SGD.from_config(config)
    lr3 = opt3.lr
    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(lr), self.evaluate(lr3))
    self.assertAllClose(
        self.evaluate(opt._get_hyper("momentum")),
        self.evaluate(opt3._get_hyper("momentum")))
    self.assertAllClose(
        self.evaluate(opt._get_hyper("decay")),
        self.evaluate(opt3._get_hyper("decay")))
    self.assertTrue(opt3.nesterov)
  def testNesterovWithoutMomentum(self):
    with self.assertRaisesRegex(ValueError, "must be between"):
      gradient_descent.SGD(learning_rate=1.0, momentum=2.0)
  def testConstructMomentumWithLR(self):
    opt = gradient_descent.SGD(lr=1.0, momentum=0.9)
    opt_2 = gradient_descent.SGD(learning_rate=0.1, momentum=0.9, lr=1.0)
    opt_3 = gradient_descent.SGD(learning_rate=0.1, momentum=0.9)
    self.assertIsInstance(opt.lr, variables.Variable)
    self.assertIsInstance(opt_2.lr, variables.Variable)
    self.assertIsInstance(opt_3.lr, variables.Variable)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_2.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_3.lr), (0.1))
  @combinations.generate(combinations.combine(mode=["eager"]))
  def testMinimizeLossTensor(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      var0 = variables.Variable([[1.0, 2.0]], dtype=dtype)
      var1 = variables.Variable([3.0], dtype=dtype)
      x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
      tape = backprop.GradientTape()
      with tape:
        loss = math_ops.matmul(var0, x) + var1
      sgd = gradient_descent.SGD(1.0)
      with self.assertRaisesRegex(ValueError, "`tape` is required"):
        sgd.minimize(loss, [var0, var1])
      sgd.minimize(loss, [var0, var1], tape=tape)
      self.assertAllCloseAccordingToType([[1.0 - 4.0, 2.0 - 5.0]],
                                         self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0 - 1.0], self.evaluate(var1))
if __name__ == "__main__":
  test.main()
