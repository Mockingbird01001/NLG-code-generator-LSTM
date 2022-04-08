
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test
class NonSerializableTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(test_base.default_test_combinations())
  def testNonSerializable(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(testing.assert_next(["FiniteSkip"]))
    dataset = dataset.apply(testing.non_serializable())
    dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.cache()
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])
  @combinations.generate(test_base.default_test_combinations())
  def testNonSerializableAsDirectInput(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(testing.non_serializable())
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])
if __name__ == "__main__":
  test.main()
