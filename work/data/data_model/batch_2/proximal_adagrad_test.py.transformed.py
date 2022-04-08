
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import proximal_adagrad
class ProximalAdagradOptimizerTest(test.TestCase):
  def doTestProximalAdagradwithoutRegularization(self, use_resource=False):
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([0.0, 0.0])
      var1 = variables.Variable([0.0, 0.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_adagrad.ProximalAdagradOptimizer(
          3.0,
          initial_accumulator_value=0.1,
          l1_regularization_strength=0.0,
          l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([0.0, 0.0], v0_val)
      self.assertAllClose([0.0, 0.0], v1_val)
      for _ in range(3):
        update.run()
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-2.60260963, -4.29698515]), v0_val)
      self.assertAllClose(np.array([-0.28432083, -0.56694895]), v1_val)
      opt_vars = opt.variables()
      self.assertStartsWith(opt_vars[0].name, var0._shared_name)
      self.assertStartsWith(opt_vars[1].name, var1._shared_name)
      self.assertEqual(2, len(opt_vars))
  def testProximalAdagradwithoutRegularization(self):
    self.doTestProximalAdagradwithoutRegularization(use_resource=False)
  def testResourceProximalAdagradwithoutRegularization(self):
    self.doTestProximalAdagradwithoutRegularization(use_resource=True)
  def testProximalAdagradwithoutRegularization2(self):
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_adagrad.ProximalAdagradOptimizer(
          3.0,
          initial_accumulator_value=0.1,
          l1_regularization_strength=0.0,
          l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)
      for _ in range(3):
        update.run()
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-1.60261, -2.296985]), v0_val)
      self.assertAllClose(np.array([3.715679, 2.433051]), v1_val)
  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        loss = pred * pred
        sgd_op = proximal_adagrad.ProximalAdagradOptimizer(1.0).minimize(loss)
        self.evaluate(variables.global_variables_initializer())
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        sgd_op.run()
        self.assertAllCloseAccordingToType([[0, 1]],
                                           self.evaluate(var0),
                                           atol=0.01)
  def testProximalAdagradWithL1(self):
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_adagrad.ProximalAdagradOptimizer(
          3.0,
          initial_accumulator_value=0.1,
          l1_regularization_strength=0.001,
          l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)
      for _ in range(10):
        update.run()
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-6.663634, -9.190331]), v0_val)
      self.assertAllClose(np.array([2.959304, 1.029232]), v1_val)
  def testProximalAdagradWithL1_L2(self):
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_adagrad.ProximalAdagradOptimizer(
          3.0,
          initial_accumulator_value=0.1,
          l1_regularization_strength=0.001,
          l2_regularization_strength=2.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)
      for _ in range(10):
        update.run()
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-0.0495, -0.0995]), v0_val)
      self.assertAllClose(np.array([-0.0045, -0.0095]), v1_val)
  def applyOptimizer(self, opt, steps=5, is_sparse=False):
    if is_sparse:
      var0 = variables.Variable([[1.0], [2.0]])
      var1 = variables.Variable([[3.0], [4.0]])
      grads0 = indexed_slices.IndexedSlices(
          constant_op.constant(
              [0.1], shape=[1, 1]),
          constant_op.constant([0]),
          constant_op.constant([2, 1]))
      grads1 = indexed_slices.IndexedSlices(
          constant_op.constant(
              [0.02], shape=[1, 1]),
          constant_op.constant([1]),
          constant_op.constant([2, 1]))
    else:
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([3.0, 4.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.evaluate(variables.global_variables_initializer())
    sess = ops.get_default_session()
    v0_val, v1_val = self.evaluate([var0, var1])
    if is_sparse:
      self.assertAllClose([[1.0], [2.0]], v0_val)
      self.assertAllClose([[3.0], [4.0]], v1_val)
    else:
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)
    for _ in range(steps):
      update.run()
    v0_val, v1_val = self.evaluate([var0, var1])
    return v0_val, v1_val
  def testEquivAdagradwithoutRegularization(self):
    with ops.Graph().as_default(), self.cached_session():
      val0, val1 = self.applyOptimizer(
          proximal_adagrad.ProximalAdagradOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0))
    with ops.Graph().as_default(), self.cached_session():
      val2, val3 = self.applyOptimizer(
          adagrad.AdagradOptimizer(
              3.0, initial_accumulator_value=0.1))
    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)
  def testEquivSparseAdagradwithoutRegularization(self):
    with ops.Graph().as_default(), self.cached_session():
      val0, val1 = self.applyOptimizer(
          proximal_adagrad.ProximalAdagradOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0),
          is_sparse=True)
    with ops.Graph().as_default(), self.cached_session():
      val2, val3 = self.applyOptimizer(
          adagrad.AdagradOptimizer(
              3.0, initial_accumulator_value=0.1),
          is_sparse=True)
    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)
if __name__ == "__main__":
  test.main()
