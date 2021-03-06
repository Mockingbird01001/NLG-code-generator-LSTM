
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
class TakeTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[-1, 0, 4, 10, 25])))
  def testBasic(self, count):
    components = (np.arange(10),)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).take(count)
    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    num_output = min(count, 10) if count != -1 else 10
    self.assertDatasetProduces(
        dataset, [tuple(components[0][i:i + 1]) for i in range(num_output)])
  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).take(1, name="take")
    self.assertDatasetProduces(dataset, [42])
class TakeDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):
  def _build_take_dataset(self, count):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).take(count)
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(count=[5], num_outputs=[5]) +
          combinations.combine(count=[20, 10, -1], num_outputs=[10]) +
          combinations.combine(count=[0], num_outputs=[0])))
  def test(self, verify_fn, count, num_outputs):
    verify_fn(self, lambda: self._build_take_dataset(count), num_outputs)
class TakeRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 3, 4])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.range(10).take(3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-2, 0, 1])))
  def testEmptyDataset(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([]).take(5)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[-1, 0, 4, 10, 25])))
  def testBasic(self, count):
    dataset = dataset_ops.Dataset.range(10).take(count)
    num_output = min(count, 10) if count != -1 else 10
    for i in range(num_output):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), i)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=num_output))
if __name__ == "__main__":
  test.main()
