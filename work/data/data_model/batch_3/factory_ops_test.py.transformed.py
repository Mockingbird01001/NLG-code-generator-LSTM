
from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test
def sparse_int64():
  return sparse_tensor.SparseTensor(
      indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 0], [5, 1], [6, 2], [7, 3]],
      values=constant_op.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtypes.int64),
      dense_shape=[8, 4])
def sparse_str():
  return sparse_tensor.SparseTensor(
      indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 0], [5, 1], [6, 2], [7, 3]],
      values=constant_op.constant(['1', '2', '3', '4', '5', '6', '7', '8']),
      dense_shape=[8, 4])
class FactoryOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.parameters(
      (sparse_int64,),
      (sparse_str,),
  )
  def testSparseWithDistributedDataset(self, sparse_factory):
    @def_function.function
    def distributed_dataset_producer(t):
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      sparse_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(sparse_ds)
      ds = iter(dist_dataset)
      return strategy.experimental_local_results(next(ds))[0]
    t = sparse_factory()
    result = distributed_dataset_producer(t)
    self.assertAllEqual(
        self.evaluate(sparse_ops.sparse_tensor_to_dense(t)[0]),
        self.evaluate(sparse_ops.sparse_tensor_to_dense(result)[0]))
if __name__ == '__main__':
  test.main()
