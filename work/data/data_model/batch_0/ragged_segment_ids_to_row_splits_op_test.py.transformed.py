
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.platform import googletest
@test_util.run_all_in_graph_and_eager_modes
class RaggedSplitsToSegmentIdsOpTest(test_util.TensorFlowTestCase):
  def testDocStringExample(self):
    segment_ids = [0, 0, 0, 2, 2, 3, 4, 4, 4]
    expected = [0, 3, 3, 5, 6, 9]
    splits = segment_id_ops.segment_ids_to_row_splits(segment_ids)
    self.assertAllEqual(splits, expected)
  def testEmptySegmentIds(self):
    segment_ids = segment_id_ops.segment_ids_to_row_splits([])
    self.assertAllEqual(segment_ids, [0])
  def testErrors(self):
    self.assertRaisesRegex(
        TypeError,
        r'Argument `tensor` \(name\: segment_ids\) must be of type integer.*',
        segment_id_ops.segment_ids_to_row_splits, constant_op.constant([0.5]))
    self.assertRaisesRegex(ValueError, r'Shape \(\) must have rank 1',
                           segment_id_ops.segment_ids_to_row_splits, 0)
    self.assertRaisesRegex(ValueError, r'Shape \(1, 1\) must have rank 1',
                           segment_id_ops.segment_ids_to_row_splits, [[0]])
  def testNumSegments(self):
    segment_ids = [0, 0, 0, 2, 2, 3, 4, 4, 4]
    num_segments = 7
    expected = [0, 3, 3, 5, 6, 9, 9, 9]
    splits = segment_id_ops.segment_ids_to_row_splits(segment_ids, num_segments)
    self.assertAllEqual(splits, expected)
  def testUnsortedSegmentIds(self):
    segment_ids = [0, 4, 3, 2, 4, 4, 2, 0, 0]
    splits1 = segment_id_ops.segment_ids_to_row_splits(segment_ids)
    expected1 = [0, 3, 3, 5, 6, 9]
    splits2 = segment_id_ops.segment_ids_to_row_splits(segment_ids, 7)
    expected2 = [0, 3, 3, 5, 6, 9, 9, 9]
    self.assertAllEqual(splits1, expected1)
    self.assertAllEqual(splits2, expected2)
if __name__ == '__main__':
  googletest.main()
