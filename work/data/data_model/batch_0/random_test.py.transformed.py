
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test
class RandomTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(global_seed=[None, 10], local_seed=[None, 20])))
  def testDeterminism(self, global_seed, local_seed):
    expect_determinism = (global_seed is not None) or (local_seed is not None)
    random_seed.set_random_seed(global_seed)
    ds = dataset_ops.Dataset.random(seed=local_seed).take(10)
    output_1 = self.getDatasetOutput(ds)
    ds = self.graphRoundTrip(ds)
    output_2 = self.getDatasetOutput(ds)
    if expect_determinism:
      self.assertEqual(output_1, output_2)
    else:
      self.assertNotEqual(output_1, output_2)
  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.random(
        seed=42, name="random").take(1).map(lambda _: 42)
    self.assertDatasetProduces(dataset, [42])
if __name__ == "__main__":
  test.main()
