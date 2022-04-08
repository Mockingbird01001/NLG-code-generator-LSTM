
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
def ConstantOf(x):
  x = np.asarray(x)
  if x.dtype.char not in "SU":
    x = np.asarray(x, dtype=np.int64)
  return constant_op.constant(x)
class EditDistanceTest(test.TestCase):
  def _testEditDistanceST(self,
                          hypothesis_st,
                          truth_st,
                          normalize,
                          expected_output,
                          expected_shape,
                          expected_err_re=None):
    edit_distance = array_ops.edit_distance(
        hypothesis=hypothesis_st, truth=truth_st, normalize=normalize)
    if expected_err_re is None:
      self.assertEqual(edit_distance.get_shape(), expected_shape)
      output = self.evaluate(edit_distance)
      self.assertAllClose(output, expected_output)
    else:
      with self.assertRaisesOpError(expected_err_re):
        self.evaluate(edit_distance)
  def _testEditDistance(self,
                        hypothesis,
                        truth,
                        normalize,
                        expected_output,
                        expected_err_re=None):
    expected_shape = [
        max(h, t) for h, t in tuple(zip(hypothesis[2], truth[2]))[:-1]
    ]
    with ops.Graph().as_default() as g, self.session(g):
      self._testEditDistanceST(
          hypothesis_st=sparse_tensor.SparseTensorValue(
              *[ConstantOf(x) for x in hypothesis]),
          truth_st=sparse_tensor.SparseTensorValue(
              *[ConstantOf(x) for x in truth]),
          normalize=normalize,
          expected_output=expected_output,
          expected_shape=expected_shape,
          expected_err_re=expected_err_re)
    with ops.Graph().as_default() as g, self.session(g):
      self._testEditDistanceST(
          hypothesis_st=sparse_tensor.SparseTensor(
              *[ConstantOf(x) for x in hypothesis]),
          truth_st=sparse_tensor.SparseTensor(*[ConstantOf(x) for x in truth]),
          normalize=normalize,
          expected_output=expected_output,
          expected_shape=expected_shape,
          expected_err_re=expected_err_re)
  def testEditDistanceNormalized(self):
    hypothesis_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
    hypothesis_values = [0, 1, 1, -1]
    hypothesis_shape = [2, 2]
    truth_indices = [[0, 0], [1, 0], [1, 1]]
    truth_values = [0, 1, 1]
    truth_shape = [2, 2]
    expected_output = [1.0, 0.5]
    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)
  def testEditDistanceUnnormalized(self):
    hypothesis_indices = [[0, 0], [1, 0], [1, 1]]
    hypothesis_values = [10, 10, 11]
    hypothesis_shape = [2, 2]
    truth_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
    truth_values = [1, 2, 1, -1]
    truth_shape = [2, 3]
    expected_output = [2.0, 2.0]
    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=False,
        expected_output=expected_output)
  def testEditDistanceProperDistance(self):
    hypothesis_indices = ([[0, i] for i, _ in enumerate("algorithm")] +
                          [[1, i] for i, _ in enumerate("altruistic")])
    hypothesis_values = [x for x in "algorithm"] + [x for x in "altruistic"]
    hypothesis_shape = [2, 11]
    truth_indices = ([[0, i] for i, _ in enumerate("altruistic")] +
                     [[1, i] for i, _ in enumerate("algorithm")])
    truth_values = [x for x in "altruistic"] + [x for x in "algorithm"]
    truth_shape = [2, 11]
    expected_unnormalized = [6.0, 6.0]
    expected_normalized = [6.0 / len("altruistic"), 6.0 / len("algorithm")]
    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=False,
        expected_output=expected_unnormalized)
    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_normalized)
  def testEditDistance3D(self):
    hypothesis_indices = [[0, 0, 0], [1, 0, 0]]
    hypothesis_values = [0, 1]
    hypothesis_shape = [2, 1, 1]
    truth_indices = [[0, 1, 0], [1, 0, 0], [1, 1, 0]]
    truth_values = [0, 1, 1]
    truth_shape = [2, 2, 1]
    expected_output = [
        [0.0, 1.0]
    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)
  def testEditDistanceZeroLengthHypothesis(self):
    hypothesis_indices = np.empty((0, 2), dtype=np.int64)
    hypothesis_values = []
    hypothesis_shape = [1, 0]
    truth_indices = [[0, 0]]
    truth_values = [0]
    truth_shape = [1, 1]
    expected_output = [1.0]
    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)
  def testEditDistanceZeroLengthTruth(self):
    hypothesis_indices = [[0, 0]]
    hypothesis_values = [0]
    hypothesis_shape = [1, 1]
    truth_indices = np.empty((0, 2), dtype=np.int64)
    truth_values = []
    truth_shape = [1, 0]
    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)
  def testEditDistanceZeroLengthHypothesisAndTruth(self):
    hypothesis_indices = np.empty((0, 2), dtype=np.int64)
    hypothesis_values = []
    hypothesis_shape = [1, 0]
    truth_indices = np.empty((0, 2), dtype=np.int64)
    truth_values = []
    truth_shape = [1, 0]
    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)
if __name__ == "__main__":
  test.main()
