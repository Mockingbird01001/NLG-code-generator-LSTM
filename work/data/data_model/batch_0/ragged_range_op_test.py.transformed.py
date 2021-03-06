
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.platform import googletest
@test_util.run_all_in_graph_and_eager_modes
class RaggedRangeOpTest(test_util.TensorFlowTestCase):
  def testDocStringExamples(self):
    rt1 = ragged_math_ops.range([3, 5, 2])
    self.assertAllEqual(rt1, [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]])
    rt2 = ragged_math_ops.range([0, 5, 8], [3, 3, 12])
    self.assertAllEqual(rt2, [[0, 1, 2], [], [8, 9, 10, 11]])
    rt3 = ragged_math_ops.range([0, 5, 8], [3, 3, 12], 2)
    self.assertAllEqual(rt3, [[0, 2], [], [8, 10]])
  def testBasicRanges(self):
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5]),
        [list(range(0)), list(range(3)),
         list(range(5))])
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [2, 3, 10]),
        [list(range(0, 2)),
         list(range(3, 3)),
         list(range(5, 10))])
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [4, 4, 15], [2, 3, 4]),
        [list(range(0, 4, 2)),
         list(range(3, 4, 3)),
         list(range(5, 15, 4))])
  def testFloatRanges(self):
    expected = [[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6], [3.0],
                [5.0, 7.2, 9.4, 11.6, 13.8]]
    actual = ragged_math_ops.range([0.0, 3.0, 5.0], [3.9, 4.0, 15.0],
                                   [0.4, 1.5, 2.2])
    self.assertAllClose(actual, expected)
  def testNegativeDeltas(self):
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], limits=0, deltas=-1),
        [list(range(0, 0, -1)),
         list(range(3, 0, -1)),
         list(range(5, 0, -1))])
    self.assertAllEqual(
        ragged_math_ops.range([0, -3, 5], limits=0, deltas=[-1, 1, -2]),
        [list(range(0, 0, -1)),
         list(range(-3, 0, 1)),
         list(range(5, 0, -2))])
  def testBroadcast(self):
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [4, 4, 15], 3),
        [list(range(0, 4, 3)),
         list(range(3, 4, 3)),
         list(range(5, 15, 3))])
    self.assertAllEqual(
        ragged_math_ops.range(0, 5, 1), [list(range(0, 5, 1))])
  def testEmptyRanges(self):
    rt1 = ragged_math_ops.range([0, 5, 3], [0, 3, 5])
    rt2 = ragged_math_ops.range([0, 5, 5], [0, 3, 5], -1)
    self.assertAllEqual(rt1, [[], [], [3, 4]])
    self.assertAllEqual(rt2, [[], [5, 4], []])
  def testShapeFnErrors(self):
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, [[0]], 5)
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, 0, [[5]])
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, 0, 5, [[0]])
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, [0], [1, 2])
  def testKernelErrors(self):
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'Requires delta != 0'):
      self.evaluate(ragged_math_ops.range(0, 0, 0))
  def testShape(self):
    self.assertAllEqual(
        ragged_math_ops.range(0, 0, 1).shape.as_list(), [1, None])
    self.assertAllEqual(
        ragged_math_ops.range([1, 2, 3]).shape.as_list(), [3, None])
    self.assertAllEqual(
        ragged_math_ops.range([1, 2, 3], [4, 5, 6]).shape.as_list(), [3, None])
if __name__ == '__main__':
  googletest.main()
