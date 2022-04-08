
import functools
import os
from os import path
import shutil
import tempfile
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as trackable_utils
class FileCacheTest(test_base.DatasetTestBase, parameterized.TestCase):
  def setUp(self):
    super(FileCacheTest, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.cache_prefix = path.join(self.tmp_dir, "cache")
  def tearDown(self):
    if self.tmp_dir:
      shutil.rmtree(self.tmp_dir, ignore_errors=True)
    super(FileCacheTest, self).tearDown()
  @combinations.generate(test_base.default_test_combinations())
  def testCacheDatasetPassthrough(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))
    def dataset_fn(count=5, filename=None):
      repeat_dataset = (
          dataset_ops.Dataset.from_tensor_slices(components).repeat(count))
      if filename:
        return repeat_dataset.cache(filename)
      else:
        return repeat_dataset
    self.assertEqual(
        tuple([c.shape[1:] for c in components]),
        dataset_ops.get_legacy_output_shapes(dataset_fn()))
    get_next = self.getNext(dataset_fn())
    elements = []
    for _ in range(20):
      elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    get_next = self.getNext(dataset_fn(filename=self.cache_prefix))
    cached_elements = []
    for _ in range(20):
      cached_elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertAllEqual(elements, cached_elements)
    get_next = self.getNext(dataset_fn(count=0, filename=self.cache_prefix))
    replayed_elements = []
    for _ in range(20):
      replayed_elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertEqual(cached_elements, replayed_elements)
    get_next = self.getNext(
        dataset_fn(count=0, filename=self.cache_prefix + "nonsense"))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentWriters(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))
    cache_dataset1 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))
    cache_dataset2 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))
    get_next1 = self.getNext(cache_dataset1)
    get_next2 = self.getNext(cache_dataset2)
    with self.assertRaises(errors.AlreadyExistsError):
      self.evaluate(get_next2())
  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentReaders(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))
    cache_dataset1 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))
    cache_dataset2 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))
    get_next1 = self.getNext(cache_dataset1)
    get_next2 = self.getNext(cache_dataset2)
    elements = []
    for _ in range(4):
      elements.append(self.evaluate(get_next1()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())
    get_next1 = self.getNext(cache_dataset1, requires_initialization=True)
    get_next2 = self.getNext(cache_dataset2, requires_initialization=True)
    elements_itr1 = []
    elements_itr2 = []
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next2())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())
    self.assertAllEqual(elements, elements_itr1)
    self.assertAllEqual(elements, elements_itr2)
  @combinations.generate(test_base.default_test_combinations())
  def testReadingPastEndOfSequence(self):
    dataset = dataset_ops.Dataset.range(10).cache(self.cache_prefix)
    dataset = dataset.map(lambda a: a).batch(4).repeat(2)
    expected_output = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]] * 2
    self.assertDatasetProduces(dataset, expected_output)
  @combinations.generate(test_base.default_test_combinations())
  def testCacheZipped(self):
    def make_dataset(i):
      cache_path = self.cache_prefix + "_" + str(i)
      return dataset_ops.Dataset.range(100).shuffle(100).cache(cache_path)
    datasets = [make_dataset(i) for i in range(3)]
    dataset = dataset_ops.Dataset.zip(tuple(datasets))
    first_order = self.getDatasetOutput(dataset)
    second_order = self.getDatasetOutput(dataset)
    self.assertEqual(first_order, second_order)
  @combinations.generate(test_base.default_test_combinations())
  def testCleaningUpCacheFiles(self):
    def do_test(i):
      dataset = dataset_ops.Dataset.range(10).cache(self.cache_prefix)
      get_next = self.getNext(dataset)
      for _ in range(i):
        try:
          self.evaluate(get_next())
        except errors.OutOfRangeError:
          break
    if not context.executing_eagerly():
      self.skipTest(
          "Test requires eager mode for iterators to be deconstructed")
    for i in [0, 3, 10, 12, 15]:
      do_test(i)
class MemoryCacheTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(test_base.default_test_combinations())
  def testCacheDatasetPassthrough(self):
    with ops.device("cpu:0"):
      repeat_count = variables.Variable(constant_op.constant(10, dtypes.int64))
      dataset = dataset_ops.Dataset.range(3).flat_map(
          lambda x: dataset_ops.Dataset.from_tensors(x).repeat(repeat_count))
      cached_dataset = dataset.cache().repeat(2)
      uncached_dataset = dataset.repeat(2)
      self.evaluate(repeat_count.initializer)
      cached_next = self.getNext(cached_dataset, requires_initialization=True)
      uncached_next = self.getNext(
          uncached_dataset, requires_initialization=True)
      for i in range(3):
        for _ in range(10):
          self.assertEqual(self.evaluate(cached_next()), i)
          self.assertEqual(self.evaluate(uncached_next()), i)
      self.evaluate(repeat_count.assign(0))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(uncached_next())
      for i in range(3):
        for _ in range(10):
          self.assertEqual(self.evaluate(cached_next()), i)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(cached_next())
  @combinations.generate(test_base.default_test_combinations())
  def testEmptyCacheReading(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))
    repeat_dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).repeat(0))
    cache_dataset = repeat_dataset.cache()
    self.assertDatasetProduces(cache_dataset, expected_output=[])
  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentReaders(self):
    dataset_fn = lambda: dataset_ops.Dataset.range(5).cache()
    d1 = dataset_fn().map(lambda x: x + 1)
    d2 = dataset_fn().map(lambda x: x + 6)
    get_next1 = self.getNext(d1)
    self.assertEqual(1, self.evaluate(get_next1()))
    self.assertEqual(2, self.evaluate(get_next1()))
    self.assertEqual(3, self.evaluate(get_next1()))
    get_next2 = self.getNext(d2)
    self.assertEqual(6, self.evaluate(get_next2()))
    self.assertEqual(7, self.evaluate(get_next2()))
    self.assertEqual([8, 5],
                     [self.evaluate(get_next2()),
                      self.evaluate(get_next1())])
    self.assertEqual(9, self.evaluate(get_next2()))
    self.assertEqual(10, self.evaluate(get_next2()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next2())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())
  @combinations.generate(test_base.default_test_combinations())
  def testCacheTakeRepeat(self):
    dataset = dataset_ops.Dataset.range(10).cache().take(5).repeat(2)
    expected_output = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    self.assertDatasetProduces(dataset, expected_output=expected_output)
  @combinations.generate(test_base.default_test_combinations())
  def testCacheRepeatEpochs(self):
    counter = variables.Variable(0)
    self.evaluate(counter.initializer)
    def increment_fn(x):
      counter.assign_add(1)
      return x
    dataset = dataset_ops.Dataset.range(10).map(increment_fn).cache().repeat(2)
    get_next = self.getNext(dataset, requires_initialization=True)
    for i in range(10):
      self.assertEqual(i, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(get_next()))
    for i in range(10):
      self.assertEqual(10, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheIterationEpochs(self):
    counter = variables.Variable(0)
    self.evaluate(counter.initializer)
    def increment_fn(x):
      counter.assign_add(1)
      return x
    dataset = dataset_ops.Dataset.range(10).map(increment_fn).cache()
    i = 0
    for elem in dataset:
      self.assertEqual(i, self.evaluate(elem))
      i += 1
      self.assertEqual(i, self.evaluate(counter))
    i = 0
    for elem in dataset:
      self.assertEqual(10, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(elem))
      i += 1
  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheV2ResourceCapture(self):
    def make_dataset():
      ids = dataset_ops.Dataset.range(10)
      ids = ids.cache()
      def interleave_fn(dataset, _):
        return dataset
      dataset = dataset_ops.Dataset.range(1)
      dataset = dataset.interleave(functools.partial(interleave_fn, ids))
      return dataset
    results = []
    for elem in make_dataset():
      results.append(elem.numpy())
    self.assertAllEqual(results, range(10))
  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheV2ConcurrentIterators(self):
    dataset = dataset_ops.Dataset.range(10).cache()
    it1 = iter(dataset)
    it2 = iter(dataset)
    for i in range(10):
      self.assertEqual(next(it1), i)
      self.assertEqual(next(it2), i)
  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheKnownCardinality(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.shuffle(10, reshuffle_each_iteration=True).cache()
    it = iter(dataset)
    results = []
    for _ in range(10):
      results.append(next(it))
    it = iter(dataset)
    for i in range(10):
      self.assertEqual(next(it), results[i])
  @combinations.generate(test_base.eager_only_combinations())
  def testCheckpointFinishedCache(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.cache()
    iterator = iter(ds)
    for i in range(num_elements):
      self.assertEqual(next(iterator).numpy(), i)
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=1)
    manager.save()
    manager.restore_or_initialize()
    with self.assertRaises(StopIteration):
      next(iterator)
  @combinations.generate(test_base.eager_only_combinations())
  def testCheckpointLargeCache(self):
    dataset = dataset_ops.Dataset.from_tensors(
        array_ops.ones((25, 1000, 1000), dtype=dtypes.float32))
    dataset = dataset.repeat(25)
    dataset = dataset.cache()
    iterator = iter(dataset)
    for _ in range(23):
      next(iterator)
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=1)
    manager.save()
  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).cache(name="cache")
    self.assertDatasetProduces(dataset, [42])
class CacheCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                          parameterized.TestCase):
  def setUp(self):
    super(CacheCheckpointTest, self).setUp()
    self.range_size = 10
    self.num_repeats = 3
    self.num_outputs = self.range_size * self.num_repeats
    self.cache_file_prefix = "test"
  def make_dataset_fn(self, is_memory):
    if is_memory:
      filename = ""
    else:
      filename = os.path.join(self.get_temp_dir(), self.cache_file_prefix)
    def ds_fn():
      return dataset_ops.Dataset.range(self.range_size).cache(filename).repeat(
          self.num_repeats)
    return ds_fn
  def expected_outputs(self):
    return list(range(self.range_size)) * self.num_repeats
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointBeforeOneEpoch(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))
    outputs.extend(
        self.gen_outputs(
            ds_fn, [],
            self.num_outputs - 5,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, self.expected_outputs())
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointBeforeOneEpochThenRunFewSteps(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(
        ds_fn, [5], 8, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, range(8))
    outputs = outputs[:5]
    outputs.extend(
        self.gen_outputs(
            ds_fn, [],
            self.num_outputs - 5,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, self.expected_outputs())
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointAfterOneEpoch(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(ds_fn, [], 15, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(5)))
    outputs.extend(
        self.gen_outputs(
            ds_fn, [],
            self.num_outputs - 15,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, self.expected_outputs())
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointAfterOneEpochThenRunFewSteps(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(
        ds_fn, [15], 18, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(8)))
    outputs = list(range(10)) + list(range(5)) + self.gen_outputs(
        ds_fn, [],
        self.num_outputs - 15,
        ckpt_saved=True,
        verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointBeforeOneEpochButRunCompleteEpoch(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(
        ds_fn, [5], 13, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(3)))
    outputs = list(range(5)) + self.gen_outputs(
        ds_fn, [],
        self.num_outputs - 5,
        ckpt_saved=True,
        verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointUnusedWriterIterator(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(ds_fn, [], 0, verify_exhausted=False)
    self.assertSequenceEqual(outputs, [])
    outputs = self.gen_outputs(
        ds_fn, [], self.num_outputs, ckpt_saved=True, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointUnusedMidwayWriterIterator(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))
    outputs.extend(
        self.gen_outputs(ds_fn, [], 0, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(5))
    outputs.extend(
        self.gen_outputs(
            ds_fn, [],
            self.num_outputs - 5,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, list(range(10)) * 3)
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testUnusedCheckpointError(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))
    if is_memory:
      outputs = self.gen_outputs(
          ds_fn, [], self.num_outputs, verify_exhausted=False)
      self.assertSequenceEqual(outputs, self.expected_outputs())
    else:
      with self.assertRaises(errors.AlreadyExistsError):
        outputs = self.gen_outputs(
            ds_fn, [], self.num_outputs, verify_exhausted=False)
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testIgnoreCheckpointIfCacheWritten(self, is_memory):
    ds_fn = self.make_dataset_fn(is_memory)
    outputs = self.gen_outputs(ds_fn, [], 15, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(5)))
    outputs = self.gen_outputs(
        ds_fn, [], self.num_outputs, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)
class CacheRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 3, 4])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3]).cache()
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index))
  @combinations.generate(test_base.default_test_combinations())
  def testCacheRangeDataset(self):
    dataset = dataset_ops.Dataset.range(10).cache()
    expected_elements = list(range(10))
    self.verifyRandomAccess(dataset, expected_elements)
  @combinations.generate(test_base.default_test_combinations())
  def testCacheOneDimensionalElements(self):
    tensor = [1, 2, 3]
    dataset = dataset_ops.Dataset.from_tensor_slices(tensor).cache()
    self.verifyRandomAccess(dataset, tensor)
  @combinations.generate(test_base.default_test_combinations())
  def testCacheTwoDimensionalElements(self):
    tensor = [[1, 2], [3, 4]]
    dataset = dataset_ops.Dataset.from_tensor_slices(tensor).cache()
    self.verifyRandomAccess(dataset, tensor)
  @combinations.generate(test_base.default_test_combinations())
  def testCacheThreeComponents(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        ([1, 2], [3, 4], [5, 6])).cache()
    expected = [(1, 3, 5), (2, 4, 6)]
    self.verifyRandomAccess(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testCacheInputDatasetNotRandomlyAccessible(self):
    dataset = dataset_ops.Dataset.range(10)
    initial_state = constant_op.constant(0, dtypes.int64)
    scan_func = lambda state, i: (state + i, state + i)
    dataset = dataset.scan(
        initial_state=initial_state, scan_func=scan_func).cache()
    expected = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    self.verifyRandomAccess(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testCacheInputDatasetUnknownCardinality(self):
    dataset = dataset_ops.Dataset.range(20).filter(
        lambda x: math_ops.equal(x % 2, 0)).cache()
    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    self.verifyRandomAccess(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testCacheInputDatasetInfiniteCardinality(self):
    dataset = dataset_ops.Dataset.range(20).filter(
        lambda x: math_ops.equal(x % 2, 0)).repeat(-1).cache()
    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 2]
    self.verifyRandomAccessInfiniteCardinality(dataset, expected)
if __name__ == "__main__":
  test.main()
