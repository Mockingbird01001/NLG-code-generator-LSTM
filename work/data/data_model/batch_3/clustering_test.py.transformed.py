
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
CPU_DEVICE = "/job:localhost/replica:0/task:0/cpu:0"
class ClusteringTest(xla_test.XLATestCase):
  def testAdd(self):
    val1 = np.array([4, 3, 2, 1], dtype=np.float32)
    val2 = np.array([5, 6, 7, 8], dtype=np.float32)
    expected = val1 + val2
    with self.session():
      with self.test_scope():
        input1 = constant_op.constant(val1, name="const1")
        input2 = constant_op.constant(val2, name="const2")
        output = math_ops.add(input1, input2)
      result = self.evaluate(output)
    self.assertAllClose(result, expected, rtol=1e-3)
  def testAddFromCpuMultiple(self):
    val1 = np.array([4, 3, 2, 1]).astype(np.float32)
    val2 = np.array([5, 6, 7, 8]).astype(np.float32)
    expected = val1 + val2
    with self.session():
      with ops.device(CPU_DEVICE):
        input1 = constant_op.constant(val1, name="const1")
        input2 = constant_op.constant(val2, name="const2")
      with self.test_scope():
        output = math_ops.add(input1, input2)
      for _ in range(10):
        result = self.evaluate(output)
        self.assertAllClose(result, expected, rtol=1e-3)
  def testDeadlock(self):
    with self.session() as sess:
      with ops.device(CPU_DEVICE):
        x = array_ops.placeholder(dtypes.float32, [2])
      with self.test_scope():
        y = x * 2
      with ops.device(CPU_DEVICE):
        z = y * y
      with self.test_scope():
        w = y + z
      result = sess.run(w, {x: [1.5, 0.5]})
    self.assertAllClose(result, [12., 2.], rtol=1e-3)
  def testHostMemory(self):
    with self.session() as sess:
      x = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        y = x + 1
      with ops.device(CPU_DEVICE):
        z = y * 2
      with self.test_scope():
        w = array_ops.reshape(z, y)
      result = sess.run(w, {x: [1, 0]})
      expected = np.array([[4], [2]], dtype=np.int32)
      self.assertAllClose(expected, result, rtol=1e-3)
if __name__ == "__main__":
  googletest.main()
