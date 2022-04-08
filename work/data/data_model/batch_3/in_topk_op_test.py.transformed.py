
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
class InTopKTest(test.TestCase):
  def _validateInTopK(self, predictions, target, k, expected):
    np_ans = np.array(expected, np.bool_)
    with self.cached_session():
      precision = nn_ops.in_top_k(predictions, target, k)
      out = self.evaluate(precision)
      self.assertAllClose(np_ans, out)
      self.assertShapeEqual(np_ans, precision)
  def testInTop1(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [3, 2]
    self._validateInTopK(predictions, target, 1, [True, False])
  def testInTop2(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [2, 2]
    self._validateInTopK(predictions, target, 2, [False, True])
  def testInTop2Tie(self):
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 3]
    self._validateInTopK(predictions, target, 2, [True, True])
  def testInTop2_int64Target(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = np.asarray([0, 2]).astype(np.int64)
    self._validateInTopK(predictions, target, 2, [False, True])
  def testInTopNan(self):
    predictions = [[0.1, float("nan"), 0.2, 0.4], [0.1, 0.2, 0.3, float("inf")]]
    target = [1, 3]
    self._validateInTopK(predictions, target, 2, [False, False])
  def testBadTarget(self):
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    self._validateInTopK(predictions, target, 2, [True, False])
  def testEmpty(self):
    predictions = np.empty([0, 5])
    target = np.empty([0], np.int32)
    self._validateInTopK(predictions, target, 2, [])
  def testTensorK(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [0, 2]
    k = constant_op.constant(3)
    self._validateInTopK(predictions, target, k, [False, True])
if __name__ == "__main__":
  test.main()
