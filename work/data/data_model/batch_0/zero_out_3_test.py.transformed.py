
import tensorflow as tf
from tensorflow.examples.adding_an_op import zero_out_op_3
class ZeroOut3Test(tf.test.TestCase):
  def test(self):
    result = zero_out_op_3.zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])
  def testAttr(self):
    result = zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=3)
    self.assertAllEqual(result, [0, 0, 0, 2, 0])
  def testNegative(self):
    with self.assertRaisesOpError("Need preserve_index >= 0, got -1"):
      self.evaluate(zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=-1))
  def testLarge(self):
    with self.assertRaisesOpError("preserve_index out of range"):
      self.evaluate(zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=17))
if __name__ == "__main__":
  tf.test.main()
