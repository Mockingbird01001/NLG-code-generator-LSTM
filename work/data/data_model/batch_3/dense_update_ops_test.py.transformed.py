
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
class AssignOpTest(test.TestCase):
  def _initAssignFetch(self, x, y, use_gpu):
    super(AssignOpTest, self).setUp()
    with test_util.device(use_gpu=use_gpu):
      p = variables.Variable(x)
      assign = state_ops.assign(p, y)
      self.evaluate(p.initializer)
      new_value = self.evaluate(assign)
      return self.evaluate(p), new_value
  def _initAssignAddFetch(self, x, y, use_gpu):
    with test_util.device(use_gpu=use_gpu):
      p = variables.Variable(x)
      add = state_ops.assign_add(p, y)
      self.evaluate(p.initializer)
      new_value = self.evaluate(add)
      return self.evaluate(p), new_value
  def _initAssignSubFetch(self, x, y, use_gpu):
    with test_util.device(use_gpu=use_gpu):
      p = variables.Variable(x)
      sub = state_ops.assign_sub(p, y)
      self.evaluate(p.initializer)
      new_value = self.evaluate(sub)
      return self.evaluate(p), new_value
  def _testTypes(self, vals):
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
      x = np.zeros(vals.shape).astype(dtype)
      y = vals.astype(dtype)
      var_value, op_value = self._initAssignFetch(x, y, use_gpu=False)
      self.assertAllEqual(y, var_value)
      self.assertAllEqual(y, op_value)
      var_value, op_value = self._initAssignAddFetch(x, y, use_gpu=False)
      self.assertAllEqual(x + y, var_value)
      self.assertAllEqual(x + y, op_value)
      var_value, op_value = self._initAssignSubFetch(x, y, use_gpu=False)
      self.assertAllEqual(x - y, var_value)
      self.assertAllEqual(x - y, op_value)
      if test.is_built_with_gpu_support() and dtype in [np.float32, np.float64]:
        var_value, op_value = self._initAssignFetch(x, y, use_gpu=True)
        self.assertAllEqual(y, var_value)
        self.assertAllEqual(y, op_value)
        var_value, op_value = self._initAssignAddFetch(x, y, use_gpu=True)
        self.assertAllEqual(x + y, var_value)
        self.assertAllEqual(x + y, op_value)
        var_value, op_value = self._initAssignSubFetch(x, y, use_gpu=True)
        self.assertAllEqual(x - y, var_value)
        self.assertAllEqual(x - y, op_value)
  def testBasic(self):
    self._testTypes(np.arange(0, 20).reshape([4, 5]))
  @test_util.run_v1_only("b/120545219")
  def testAssignNonStrictShapeChecking(self):
    with self.cached_session():
      data = array_ops.fill([1024, 1024], 0)
      p = variables.VariableV1([1])
      a = state_ops.assign(p, data, validate_shape=False)
      a.op.run()
      self.assertAllEqual(p, self.evaluate(data))
      data2 = array_ops.fill([10, 10], 1)
      a2 = state_ops.assign(p, data2, validate_shape=False)
      a2.op.run()
      self.assertAllEqual(p, self.evaluate(data2))
  @test_util.run_v1_only("b/120545219")
  def testInitRequiredAssignAdd(self):
    with self.cached_session():
      p = variables.VariableV1(array_ops.fill([1024, 1024], 1), dtypes.int32)
      a = state_ops.assign_add(p, array_ops.fill([1024, 1024], 0))
      with self.assertRaisesOpError("use uninitialized"):
        a.op.run()
  @test_util.run_v1_only("b/120545219")
  def testInitRequiredAssignSub(self):
    with self.cached_session():
      p = variables.VariableV1(array_ops.fill([1024, 1024], 1), dtypes.int32)
      a = state_ops.assign_sub(p, array_ops.fill([1024, 1024], 0))
      with self.assertRaisesOpError("use uninitialized"):
        a.op.run()
if __name__ == "__main__":
  test.main()
