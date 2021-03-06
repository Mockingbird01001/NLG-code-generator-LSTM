
from absl.testing import parameterized
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest
@test_util.run_all_in_graph_and_eager_modes
class RaggedPlaceholderOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  @parameterized.parameters([
      (dtypes.int32, 0, [5], None,
       'Tensor("Placeholder:0", shape=(5,), dtype=int32)'),
      (dtypes.int32, 1, [], 'ph', 'tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None,), dtype=int32), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.string, 1, [5], 'ph', 'tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None, 5), dtype=string), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.float32, 2, [], 'ph', 'tf.RaggedTensor(values=tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None,), dtype=float32), '
       'row_splits=Tensor("ph/row_splits_1:0", shape=(None,), dtype=int64)), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.int32, 2, [3, 5], 'ph', 'tf.RaggedTensor(values=tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None, 3, 5), dtype=int32), '
       'row_splits=Tensor("ph/row_splits_1:0", shape=(None,), dtype=int64)), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
  ])
  def testRaggedPlaceholder(self, dtype, ragged_rank, value_shape, name,
                            expected):
    if not context.executing_eagerly():
      placeholder = ragged_factory_ops.placeholder(
          dtype, ragged_rank, value_shape, name)
      result = str(placeholder).replace('?', 'None')
      self.assertEqual(result, expected)
  def testRaggedPlaceholderRaisesExceptionInEagerMode(self):
    if context.executing_eagerly():
      with self.assertRaises(RuntimeError):
        ragged_factory_ops.placeholder(dtypes.int32, 1, [])
  def testRaggedPlaceholderDoesNotIncludeValidationOps(self):
    if context.executing_eagerly():
      return
    graph = ops.Graph()
    with graph.as_default():
      ragged_factory_ops.placeholder(
          dtypes.float32, ragged_rank=1, value_shape=[])
      self.assertEqual([op.type for op in graph.get_operations()],
                       ['Placeholder', 'Placeholder'])
if __name__ == '__main__':
  googletest.main()
