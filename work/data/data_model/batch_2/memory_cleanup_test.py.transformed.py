
import gc
import time
from absl.testing import parameterized
import six
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
try:
except ImportError:
  memory_profiler = None
class MemoryCleanupTest(test_base.DatasetTestBase, parameterized.TestCase):
  def setUp(self):
    super(MemoryCleanupTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)
  def assertMemoryNotIncreasing(self, f, num_iters, max_increase_mb):
    f()
    time.sleep(4)
    initial = memory_profiler.memory_usage(-1)[0]
    for _ in six.moves.range(num_iters):
      f()
    increase = memory_profiler.memory_usage(-1)[0] - initial
    logging.info("Memory increase observed: %f MB" % increase)
    assert increase < max_increase_mb, (
        "Increase is too high. Initial memory usage: %f MB. Increase: %f MB. "
        "Maximum allowed increase: %f") % (initial, increase, max_increase_mb)
  def assertNoMemoryLeak(self, dataset_fn):
    def run():
      get_next = self.getNext(dataset_fn())
      for _ in range(100):
        self.evaluate(get_next())
    for _ in range(10):
      run()
    gc.collect()
    tensors = [
        o for o in gc.get_objects() if isinstance(o, internal.NativeObject)
    ]
    self.assertEmpty(tensors, "%d Tensors are still alive." % len(tensors))
  @combinations.generate(test_base.eager_only_combinations())
  def testEagerMemoryUsageWithReset(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])
    def f():
      self.evaluate(multi_device_iterator.get_next())
      multi_device_iterator._eager_reset()
    self.assertMemoryNotIncreasing(f, num_iters=50, max_increase_mb=250)
  @combinations.generate(test_base.eager_only_combinations())
  def testEagerMemoryUsageWithRecreation(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")
    dataset = dataset_ops.Dataset.range(10)
    def f():
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, [self._devices[1], self._devices[2]])
      self.evaluate(multi_device_iterator.get_next())
      del multi_device_iterator
    self.assertMemoryNotIncreasing(f, num_iters=50, max_increase_mb=250)
  @combinations.generate(test_base.eager_only_combinations())
  def testFilter(self):
    def get_dataset():
      def fn(_):
        return True
      return dataset_ops.Dataset.range(0, 100).filter(fn)
    self.assertNoMemoryLeak(get_dataset)
  @combinations.generate(combinations.combine(tf_api_version=1, mode="eager"))
  def testFilterLegacy(self):
    def get_dataset():
      def fn(_):
        return True
      return dataset_ops.Dataset.range(0, 100).filter_with_legacy_function(fn)
    self.assertNoMemoryLeak(get_dataset)
  @combinations.generate(test_base.eager_only_combinations())
  def testFlatMap(self):
    def get_dataset():
      def fn(x):
        return dataset_ops.Dataset.from_tensors(x * x)
      return dataset_ops.Dataset.range(0, 100).flat_map(fn)
    self.assertNoMemoryLeak(get_dataset)
  @combinations.generate(test_base.eager_only_combinations())
  def testFromGenerator(self):
    def get_dataset():
      def fn():
        return six.moves.range(100)
      return dataset_ops.Dataset.from_generator(fn, output_types=dtypes.float32)
    self.assertNoMemoryLeak(get_dataset)
  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_parallel_calls=[None, 10])))
  def testMap(self, num_parallel_calls):
    def get_dataset():
      def fn(x):
        return x * x
      return dataset_ops.Dataset.range(0, 100).map(
          fn, num_parallel_calls=num_parallel_calls)
    self.assertNoMemoryLeak(get_dataset)
  @combinations.generate(
      combinations.combine(
          tf_api_version=1, mode="eager", num_parallel_calls=[None, 10]))
  def testMapLegacy(self, num_parallel_calls):
    def get_dataset():
      def fn(x):
        return x * x
      return dataset_ops.Dataset.range(0, 100).map_with_legacy_function(
          fn, num_parallel_calls=num_parallel_calls)
    self.assertNoMemoryLeak(get_dataset)
  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_parallel_calls=[None, 10])))
  def testInterleave(self, num_parallel_calls):
    def get_dataset():
      def fn(x):
        return dataset_ops.Dataset.from_tensors(x * x)
      return dataset_ops.Dataset.range(0, 100).interleave(
          fn, num_parallel_calls=num_parallel_calls, cycle_length=10)
    self.assertNoMemoryLeak(get_dataset)
if __name__ == "__main__":
  test.main()
