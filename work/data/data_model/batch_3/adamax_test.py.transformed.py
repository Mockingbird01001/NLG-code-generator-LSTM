
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras.optimizer_v2 import adamax
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
def adamax_update_numpy(param,
                        g_t,
                        t,
                        m,
                        v,
                        alpha=0.001,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-8):
  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = np.maximum(beta2 * v, np.abs(g_t))
  param_t = param - (alpha / (1 - beta1**(t + 1))) * (m_t / (v_t + epsilon))
  return param_t, m_t, v_t
def adamax_sparse_update_numpy(param,
                               indices,
                               g_t,
                               t,
                               m,
                               v,
                               alpha=0.001,
                               beta1=0.9,
                               beta2=0.999,
                               epsilon=1e-8):
  m_t, v_t, param_t = np.copy(m), np.copy(v), np.copy(param)
  m_t_slice = beta1 * m[indices] + (1 - beta1) * g_t
  v_t_slice = np.maximum(beta2 * v[indices], np.abs(g_t))
  param_t_slice = param[indices] - (
      (alpha / (1 - beta1**(t + 1))) * (m_t_slice / (v_t_slice + epsilon)))
  m_t[indices] = m_t_slice
  v_t[indices] = v_t_slice
  param_t[indices] = param_t_slice
  return param_t, m_t, v_t
def get_beta_accumulators(opt, dtype):
  local_step = math_ops.cast(opt.iterations + 1, dtype)
  beta_1_t = math_ops.cast(opt._get_hyper("beta_1"), dtype)
  beta_1_power = math_ops.pow(beta_1_t, local_step)
  return beta_1_power
class AdamaxOptimizerTest(test.TestCase, parameterized.TestCase):
  def testResourceSparse(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        m0, v0, m1, v1 = zero_slots(), zero_slots(), zero_slots(), zero_slots()
        var0_np = np.array([1.0, 2.0, 3.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([4.0, 5.0, 6.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([3]))
        grads1_np_indices = np.array([2, 1], dtype=np.int32)
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([3]))
        opt = adamax.Adamax()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([1.0, 2.0, 3.0], var0)
        self.assertAllClose([4.0, 5.0, 6.0], var1)
        beta1_power = get_beta_accumulators(opt, dtype)
        for t in range(3):
          self.assertAllCloseAccordingToType(0.9**(t + 1), beta1_power)
          update.run()
          var0_np, m0, v0 = adamax_sparse_update_numpy(
              var0_np, grads0_np_indices, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adamax_sparse_update_numpy(
              var1_np, grads1_np_indices, grads1_np, t, m1, v1)
          self.assertAllCloseAccordingToType(var0_np, var0)
          self.assertAllCloseAccordingToType(var1_np, var1)
  def testSparseDevicePlacement(self):
    for index_dtype in [dtypes.int32, dtypes.int64]:
      with ops.Graph().as_default(), self.cached_session(
          force_gpu=test.is_gpu_available()):
        var = variables.Variable([[1.0], [2.0]])
        indices = constant_op.constant([0, 1], dtype=index_dtype)
        optimizer = adamax.Adamax(3.0)
        minimize_op = optimizer.minimize(g_sum, var_list=[var])
        self.evaluate(variables.global_variables_initializer())
        minimize_op.run()
  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        repeated_index_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = indexed_slices.IndexedSlices(
            constant_op.constant(
                [0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]),
            constant_op.constant([2, 1]))
        grad_aggregated = indexed_slices.IndexedSlices(
            constant_op.constant(
                [0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        repeated_update = adamax.Adamax().apply_gradients(
            [(grad_repeated_index, repeated_index_update_var)])
        aggregated_update = adamax.Adamax().apply_gradients(
            [(grad_aggregated, aggregated_update_var)])
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose(aggregated_update_var,
                            repeated_index_update_var.eval())
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(aggregated_update_var,
                              repeated_index_update_var.eval())
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasic(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph(), use_gpu=True):
        m0 = np.array([0.0, 0.0])
        v0 = np.array([0.0, 0.0])
        m1 = np.array([0.0, 0.0])
        v1 = np.array([0.0, 0.0])
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np, name="var0_%d" % i)
        var1 = variables.Variable(var1_np, name="var1_%d" % i)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adamax.Adamax()
        if not context.executing_eagerly():
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        for t in range(3):
          beta_1_power = get_beta_accumulators(opt, dtype)
          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta_1_power))
          if not context.executing_eagerly():
            self.evaluate(update)
          else:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          var0_np, m0, v0 = adamax_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adamax_update_numpy(var1_np, grads1_np, t, m1, v1)
          self.assertAllCloseAccordingToType(
              var0_np, self.evaluate(var0), rtol=1e-2)
          self.assertAllCloseAccordingToType(
              var1_np, self.evaluate(var1), rtol=1e-2)
  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasicWithLearningRateDecay(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph(), use_gpu=True):
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np, name="var0_%d" % i)
        var1 = variables.Variable(var1_np, name="var1_%d" % i)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        learning_rate = 0.001
        decay = 0.002
        opt = adamax.Adamax(learning_rate=learning_rate, decay=decay)
        if not context.executing_eagerly():
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        for t in range(3):
          beta_1_power = get_beta_accumulators(opt, dtype)
          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta_1_power))
          if not context.executing_eagerly():
            self.evaluate(update)
          else:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          lr = learning_rate / (1 + decay * t)
          var0_np, m0, v0 = adamax_update_numpy(
              var0_np, grads0_np, t, m0, v0, alpha=lr)
          var1_np, m1, v1 = adamax_update_numpy(
              var1_np, grads1_np, t, m1, v1, alpha=lr)
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0),
                                             rtol=1e-2)
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1),
                                             rtol=1e-2)
  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adamax.Adamax(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([1.0, 2.0], var0)
        self.assertAllClose([3.0, 4.0], var1)
        beta1_power = get_beta_accumulators(opt, dtype)
        for t in range(3):
          self.assertAllCloseAccordingToType(0.9**(t + 1), beta1_power)
          update.run()
          var0_np, m0, v0 = adamax_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adamax_update_numpy(var1_np, grads1_np, t, m1, v1)
          self.assertAllCloseAccordingToType(var0_np, var0)
          self.assertAllCloseAccordingToType(var1_np, var1)
  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adamax.Adamax()
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        beta1_power = get_beta_accumulators(opt, dtype)
        self.assertAllClose([1.0, 2.0], var0)
        self.assertAllClose([3.0, 4.0], var1)
        for t in range(3):
          self.assertAllCloseAccordingToType(0.9**(t + 1), beta1_power)
          if t % 2 == 0:
            update1.run()
          else:
            update2.run()
          var0_np, m0, v0 = adamax_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adamax_update_numpy(var1_np, grads1_np, t, m1, v1)
          self.assertAllCloseAccordingToType(var0_np, var0)
          self.assertAllCloseAccordingToType(var1_np, var1)
  @combinations.generate(combinations.combine(mode=["eager"]))
  def testSlotsUniqueEager(self):
    v1 = variables.Variable(1.)
    v2 = variables.Variable(1.)
    opt = adamax.Adamax(1.)
    opt.minimize(lambda: v1 + v2, var_list=[v1, v2])
    self.assertLen({id(v) for v in opt.variables()}, 5)
  def testConstructAdamaxWithLR(self):
    opt = adamax.Adamax(lr=1.0)
    opt_2 = adamax.Adamax(learning_rate=0.1, lr=1.0)
    opt_3 = adamax.Adamax(learning_rate=0.1)
    self.assertIsInstance(opt.lr, variables.Variable)
    self.assertIsInstance(opt_2.lr, variables.Variable)
    self.assertIsInstance(opt_3.lr, variables.Variable)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_2.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_3.lr), (0.1))
if __name__ == "__main__":
  test.main()
