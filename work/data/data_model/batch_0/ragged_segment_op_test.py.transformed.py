
import math
from absl.testing import parameterized
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
def prod(values):
  val = 1
  for v in values:
    val *= v
  return val
def mean(values):
  return 1.0 * sum(values) / len(values)
def sqrt_n(values):
  return 1.0 * sum(values) / math.sqrt(len(values))
@test_util.run_all_in_graph_and_eager_modes
class RaggedSegmentOpsTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  def expected_value(self, data, segment_ids, num_segments, combiner):
    self.assertLen(data, len(segment_ids))
    ncols = max(len(row) for row in data)
    grouped = [[[] for _ in range(ncols)] for row in range(num_segments)]
    for row in range(len(data)):
      for col in range(len(data[row])):
        grouped[segment_ids[row]][col].append(data[row][col])
    return [[combiner(values)
             for values in grouped_row
             if values]
            for grouped_row in grouped]
  @parameterized.parameters(
      (ragged_math_ops.segment_sum, sum, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_sum, sum, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_sum, sum, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_sum, sum, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_prod, prod, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_min, min, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_min, min, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_min, min, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_min, min, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_max, max, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_max, max, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_max, max, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_max, max, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_mean, mean, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 0, 10, 10, 10]),
  )
  def testRaggedSegment_Int(self, segment_op, combiner, segment_ids):
    rt_as_list = [[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]]
    rt = ragged_factory_ops.constant(rt_as_list)
    num_segments = max(segment_ids) + 1
    expected = self.expected_value(rt_as_list, segment_ids, num_segments,
                                   combiner)
    segmented = segment_op(rt, segment_ids, num_segments)
    self.assertAllEqual(segmented, expected)
  @parameterized.parameters(
      (ragged_math_ops.segment_sum, sum, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_sum, sum, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_sum, sum, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_sum, sum, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_prod, prod, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_min, min, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_min, min, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_min, min, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_min, min, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_max, max, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_max, max, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_max, max, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_max, max, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_mean, mean, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_sqrt_n, sqrt_n, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_sqrt_n, sqrt_n, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_sqrt_n, sqrt_n, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_sqrt_n, sqrt_n, [0, 0, 0, 10, 10, 10]),
  )
  def testRaggedSegment_Float(self, segment_op, combiner, segment_ids):
    rt_as_list = [[0., 1., 2., 3.], [4.], [], [5., 6.], [7.], [8., 9.]]
    rt = ragged_factory_ops.constant(rt_as_list)
    num_segments = max(segment_ids) + 1
    expected = self.expected_value(rt_as_list, segment_ids, num_segments,
                                   combiner)
    segmented = segment_op(rt, segment_ids, num_segments)
    self.assertAllClose(segmented, expected)
  def testRaggedRankTwo(self):
    rt = ragged_factory_ops.constant([
    segment_ids1 = [0, 2, 2, 2]
    segmented1 = ragged_math_ops.segment_sum(rt, segment_ids1, 3)
    self.assertAllEqual(segmented1, expected1)
    segment_ids2 = [1, 2, 1, 1]
    segmented2 = ragged_math_ops.segment_sum(rt, segment_ids2, 3)
    expected2 = [[],
                 [[111+411, 112+412, 113, 114], [121+321, 322], [331]],
    self.assertAllEqual(segmented2, expected2)
  def testRaggedSegmentIds(self):
    rt = ragged_factory_ops.constant([
    segment_ids = ragged_factory_ops.constant([[1, 2], [], [1, 1, 2], [2]])
    segmented = ragged_math_ops.segment_sum(rt, segment_ids, 3)
    expected = [[],
                [111+321, 112+322, 113, 114],
    self.assertAllEqual(segmented, expected)
  def testShapeMismatchError1(self):
    dt = constant_op.constant([1, 2, 3, 4, 5, 6])
    segment_ids = ragged_factory_ops.constant([[1, 2], []])
    self.assertRaisesRegex(
        ValueError, 'segment_ids.shape must be a prefix of data.shape, '
        'but segment_ids is ragged and data is not.',
        ragged_math_ops.segment_sum, dt, segment_ids, 3)
  def testShapeMismatchError2(self):
    rt = ragged_factory_ops.constant([
    segment_ids = ragged_factory_ops.constant([[1, 2], [1], [1, 1, 2], [2]])
    self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'segment_ids.shape must be a prefix of data.shape.*',
        ragged_math_ops.segment_sum, rt, segment_ids, 3)
    segment_ids2 = ragged_tensor.RaggedTensor.from_row_splits(
        array_ops.placeholder_with_default(segment_ids.values, None),
        array_ops.placeholder_with_default(segment_ids.row_splits, None))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'segment_ids.shape must be a prefix of data.shape.*'):
      self.evaluate(ragged_math_ops.segment_sum(rt, segment_ids2, 3))
if __name__ == '__main__':
  googletest.main()
