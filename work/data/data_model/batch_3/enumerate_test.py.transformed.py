
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test
class EnumerateTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(test_base.default_test_combinations())
  def testEnumerate(self):
    components = (["a", "b"], [1, 2], [37.0, 38])
    start = constant_op.constant(20, dtype=dtypes.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).enumerate(
        start)
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset)[0])
    dataset_output_shapes = dataset_ops.get_legacy_output_shapes(dataset)
    self.assertEqual((), dataset_output_shapes[0])
    self.assertEqual([tensor_shape.TensorShape([])] * 3,
                     [shape for shape in dataset_output_shapes[1]])
    self.assertDatasetProduces(dataset, [(20, (b"a", 1, 37.0)),
                                         (21, (b"b", 2, 38.0))])
if __name__ == "__main__":
  test.main()
