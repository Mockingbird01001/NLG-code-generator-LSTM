
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
class PrefetchWithSlackTest(test_base.DatasetTestBase, parameterized.TestCase):
  def setUp(self):
    super(PrefetchWithSlackTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)
  @combinations.generate(test_base.default_test_combinations())
  def testPrefetchWithSlackOption(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])
    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)
  @combinations.generate(test_base.default_test_combinations())
  def testPrefetchWithSlackOptionWithoutIterator(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(10))
  @combinations.generate(test_base.default_test_combinations())
  def testWithPassthroughDataset(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    dataset = dataset.map(lambda x: x + 1)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(1, 11))
  @combinations.generate(test_base.default_test_combinations())
  def testNoErrorWithoutPrefetch(self):
    dataset = dataset_ops.Dataset.range(10)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(10))
  @combinations.generate(test_base.default_test_combinations())
  def testNoErrorWithInvalidDataset(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensors)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(10))
if __name__ == "__main__":
  test.main()
