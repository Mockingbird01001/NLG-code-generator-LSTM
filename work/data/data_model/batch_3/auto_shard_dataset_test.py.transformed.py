
import os
from absl.testing import parameterized
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.experimental.ops import unique
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
def chunk(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]
class AutoShardDatasetTest(tf_record_test_base.TFRecordTestBase,
                           parameterized.TestCase):
  def setUp(self):
    super(AutoShardDatasetTest, self).setUp()
    self._num_files = 10
    self._num_records = 10
    self._filenames = self._createFiles()
  def getAllDatasetElements(self, dataset):
    actual = []
    next_fn = self.getNext(dataset)
    while True:
      try:
        actual.append(self.evaluate(next_fn()))
      except errors.OutOfRangeError:
        break
    return actual
  def assertDatasetProducesWithShuffle(self, dataset, expected, batch,
                                       num_examples, shuffle):
    if shuffle:
      actual = []
      next_fn = self.getNext(dataset)
      for _ in range(num_examples):
        elem = self.evaluate(next_fn())
        if isinstance(elem, tuple):
          actual.extend(elem)
        else:
          actual.extend(elem.tolist())
      self.assertCountEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_fn())
    else:
      self.assertDatasetProduces(dataset, list(chunk(expected, batch)))
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testFlatMapReaderPipeline(self, shuffle):
    dataset = dataset_ops.Dataset.list_files(
        self._filenames, shuffle=shuffle)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    expected = [
        for f in (3, 8)
        for r in range(0, 10)
    ]
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 4, shuffle)
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(batch_size=[1, 3, 10])))
  def testDatasetOfReaderDatasetsPipeline(self, batch_size):
    def batch(iterator, n):
      l = len(iterator)
      for i in range(0, l, n):
        yield iterator[i:min(i + n, l)]
    datasets = []
    for files in batch(self._filenames, batch_size):
      datasets.append(
          dataset_ops.Dataset.list_files(files, shuffle=False).map(
              core_readers.TFRecordDataset))
    dataset = dataset_ops.Dataset.from_tensor_slices(datasets)
    dataset = dataset.flat_map(lambda x: x)
    dataset = dataset.prefetch(1)
    dataset = dataset.prefetch(1)
    dataset = dataset.interleave(
        lambda x: x, cycle_length=1, num_parallel_calls=1)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)
    expected = [
        for f in (0, 5)
        for r in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testZipReaderPipeline(self):
    dataset1 = dataset_ops.Dataset.list_files(
        self._filenames, shuffle=False)
    dataset1 = dataset1.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset2 = dataset_ops.Dataset.list_files(
        self._filenames, shuffle=False)
    dataset2 = dataset2.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    expected = [
        for r in range(0, 10)
        for f in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testConcatenateReaderPipeline(self, shuffle):
    dataset1 = dataset_ops.Dataset.list_files(
        self._filenames, shuffle=shuffle)
    dataset1 = dataset1.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset1 = dataset1.batch(5)
    dataset2 = dataset_ops.Dataset.list_files(
        self._filenames, shuffle=shuffle)
    dataset2 = dataset2.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset2 = dataset2.batch(5)
    dataset = dataset1.concatenate(dataset2)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    expected = [
        for r in range(0, 10)
        for f in (3, 8)
    ]
    expected += expected
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 8, shuffle)
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testPipelineWithMap(self, shuffle):
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset.map(lambda x: string_ops.substr_v2(x, 2, 1000))
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    expected = [
        for r in range(0, 10)
        for f in (3, 8)
    ]
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 4, shuffle)
  @combinations.generate(test_base.default_test_combinations())
  def testDirectFilenameTFRecordReaderPipeline(self):
    dataset = core_readers.TFRecordDataset(self._filenames)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)
    expected = [
        for f in (0, 5)
        for r in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testValidPipelineWithRangeDataset(self, shuffle):
    dataset = dataset_ops.Dataset.range(self._num_files)
        [self.get_temp_dir(),
         string_ops.string_format("/tf_record.{}.txt", [n])]))
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset.map(lambda x: string_ops.substr_v2(x, 2, 1000))
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    expected = [
        for r in range(0, 10)
        for f in (3, 8)
    ]
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 4, shuffle)
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(params=[(1, 0, 10, 10), (2, 1, 20, 5),
                                       (10, 1, 1, 10)])))
  def testStandardReaderPipeline(self, params):
    num_epochs, index, batch_size, parallel_reads = params
    dataset = readers.make_tf_record_dataset(
        file_pattern=self._filenames,
        num_epochs=num_epochs,
        batch_size=batch_size,
        parser_fn=None,
        num_parallel_reads=parallel_reads,
        drop_final_batch=True,
        shuffle=False)
    dataset = distribute._AutoShardDataset(dataset, 2, index)
    outputs = self.getNext(dataset)
    self._verify_records(
        outputs,
        batch_size=batch_size,
        file_index=[i for i in range(index, self._num_records, 2)],
        num_epochs=num_epochs,
        interleave_cycle_length=parallel_reads,
        drop_final_batch=True,
        use_parser_fn=None)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(outputs())
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testSampleResNetPipeline(self, shuffle):
    dataset = dataset_ops.Dataset.list_files(
        self._filenames, shuffle=shuffle)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    expected = [
        for r in range(0, 10)
        for f in (3, 8)
    ]
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 4, shuffle)
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              options_lib.AutoShardPolicy.DATA,
              options_lib.AutoShardPolicy.AUTO
          ])))
  def testShardByDataBeforePrefetch(self, sharding_policy):
    dataset = dataset_ops.Dataset.range(4)
    dataset = dataset.apply(testing.assert_next(["Shard", "Prefetch"]))
    dataset = dataset.prefetch(1)
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = sharding_policy
    dataset = dataset.with_options(options)
    dataset = distribute._AutoShardDataset(dataset, 2, 0)
    self.assertDatasetProduces(dataset, [0, 2])
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.times(combinations.combine(
              sharding_policy=[options_lib.AutoShardPolicy.DATA,
                               options_lib.AutoShardPolicy.FILE]),
                             combinations.combine(shuffle=[True, False]))))
  def testReplicateAndShardProduceDisjointData(self, shuffle, sharding_policy):
    dataset = dataset_ops.Dataset.list_files(self._filenames,
                                             shuffle=shuffle)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    graph_def = dataset._as_serialized_graph(
        strip_device_assignment=True,
        external_state_policy=options_lib.ExternalStatePolicy.WARN)
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = sharding_policy
    ds1 = distribute._RemoteDataset(graph_def, "/device:CPU:0",
                                    dataset.element_spec)
    ds2 = distribute._RemoteDataset(graph_def, "/device:CPU:0",
                                    dataset.element_spec)
    ds1 = ds1.with_options(options)
    ds2 = ds2.with_options(options)
    ds1 = distribute._AutoShardDataset(ds1, 2, 0)
    ds2 = distribute._AutoShardDataset(ds2, 2, 1)
    elems1 = set(self.getAllDatasetElements(ds1))
    elems2 = set(self.getAllDatasetElements(ds2))
    self.assertEmpty(elems1.intersection(elems2))
  @combinations.generate(test_base.default_test_combinations())
  def testWorkersGreaterThanNumFilesWithDataSharding(self):
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = (
        options_lib.AutoShardPolicy.DATA)
    dataset = core_readers._TFRecordDataset(self._filenames)
    dataset = dataset.with_options(options)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)
    expected = [
        for f in range(0, 10)
        for r in (0, 5)
    ]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testAutoshardPolicyOff(self):
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = (
        options_lib.AutoShardPolicy.OFF)
    dataset = core_readers._TFRecordDataset(self._filenames)
    dataset = dataset.with_options(options)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)
    expected = [
        for f in range(0, 10)
        for r in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testFileShardingWithoutReaderDatasetOp(self):
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = (
        options_lib.AutoShardPolicy.FILE)
    dataset = dataset_ops.Dataset.range(1024)
    dataset = dataset.with_options(options)
    with self.assertRaises(errors.NotFoundError):
      dataset = distribute._AutoShardDataset(dataset, 10, 0)
      self.evaluate(self.getNext(dataset)())
  @combinations.generate(test_base.default_test_combinations())
  def testWorkersGreaterThanNumFiles(self):
    dataset = dataset_ops.Dataset.list_files(self._filenames)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 500, 499)
    self.assertDatasetProduces(dataset, [])
  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordReaderWithDirectFileNames(self):
    dataset = core_readers._TFRecordDataset(self._filenames)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)
    expected = [
        for f in range(0, 10)
        for r in (0, 5)
    ]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordReaderWithDirectFileNamesAndShapes(self):
    dataset = core_readers._TFRecordDataset(self._filenames)
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 2, 0)
    expected = [
        for f in range(0, 10)
        for r in range(0, 5)
    ]
    self.assertDatasetProduces(dataset, list(chunk(expected, 5)))
  @combinations.generate(test_base.default_test_combinations())
  def testShardOutOfRange(self):
    dataset = dataset_ops.Dataset.range(5)
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = distribute._AutoShardDataset(dataset, 10, 0)
      self.evaluate(self.getNext(dataset)())
  @combinations.generate(test_base.default_test_combinations())
  def testShardOutOfRangeEmptyDataset(self):
    dataset = dataset_ops.Dataset.range(0)
    with self.assertRaises(errors.OutOfRangeError):
      dataset = distribute._AutoShardDataset(dataset, 10, 0)
      self.evaluate(self.getNext(dataset)())
  @combinations.generate(test_base.default_test_combinations())
  def testNoReaderPipelines(self):
    dataset = dataset_ops.Dataset.range(1024)
    dataset = distribute._AutoShardDataset(dataset, 2, 0)
    self.assertDatasetProduces(dataset, [i for i in range(1024) if i % 2 == 0])
  @combinations.generate(test_base.default_test_combinations())
  def testUnknownOpInPipelineStillShardsAtTheEnd(self):
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.apply(unique.unique())
    dataset = distribute._AutoShardDataset(dataset, 5, 0)
    expected = [
        for f in range(0, 10)
        for r in (0, 5)
    ]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testInvalidWorkerIndex(self):
    dataset = dataset_ops.Dataset.list_files(self._filenames)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = distribute._AutoShardDataset(dataset, 2, 2)
      self.evaluate(self.getNext(dataset)())
  @combinations.generate(test_base.default_test_combinations())
  def testAssertCardinality(self):
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)
    dataset = dataset.apply(cardinality.assert_cardinality(42))
    dataset = distribute._AutoShardDataset(dataset, 5, 0)
    expected = [
        for f in (0, 5)
        for r in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, list(chunk(expected, 5)))
  @combinations.generate(test_base.default_test_combinations())
  def testMakeBatchedFeaturesDataset(self):
    files = 2
    records_per_file = 5
    def make_record(file_index):
      example = example_pb2.Example(
          features=feature_pb2.Features(
              feature={
                  "file":
                      feature_pb2.Feature(
                          int64_list=feature_pb2.Int64List(value=[file_index])),
              }))
      return example.SerializeToString()
    filenames = []
    for file_index in range(files):
      filename = os.path.join(self.get_temp_dir(),
                              "tf_record.%d.txt" % file_index)
      filenames.append(filename)
      writer = python_io.TFRecordWriter(filename)
      for _ in range(records_per_file):
        writer.write(make_record(file_index))
      writer.close()
    dataset = readers.make_batched_features_dataset(
        file_pattern=filenames,
        batch_size=records_per_file,
        features={
            "file": parsing_ops.FixedLenFeature([], dtypes.int64),
        },
        reader=core_readers.TFRecordDataset,
        num_epochs=1)
    dataset = distribute._AutoShardDataset(dataset, 2, 0)
    dataset = dataset.unbatch()
    output = self.getDatasetOutput(dataset)
    files = [elem["file"] for elem in output]
    self.assertEqual(files, [0] * records_per_file)
  @combinations.generate(test_base.default_test_combinations())
  def testHintShardingValidPattern(self):
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = (
        options_lib.AutoShardPolicy.HINT)
    dataset = dataset_ops.Dataset.range(100).shard(distribute.SHARD_HINT, 0)
    dataset = dataset.with_options(options)
    dataset = distribute._AutoShardDataset(dataset, 10, 0)
    self.assertDatasetProduces(dataset, list(range(0, 100, 10)))
  @combinations.generate(test_base.default_test_combinations())
  def testHintShardingInvalidPattern(self):
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = (
        options_lib.AutoShardPolicy.HINT)
    dataset = dataset_ops.Dataset.range(100).shard(1, 0)
    dataset = dataset.with_options(options)
    dataset = distribute._AutoShardDataset(dataset, 10, 0)
    self.assertDatasetProduces(dataset, list(range(100)))
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              auto_shard_policy=list(options_lib.AutoShardPolicy))))
  def testEnumerateAutoShardPolicies(self, auto_shard_policy):
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = auto_shard_policy
    dataset = dataset.with_options(options)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    self.getDatasetOutput(dataset, requires_initialization=True)
class AutoShardWithRebatchDatasetTest(tf_record_test_base.TFRecordTestBase,
                                      parameterized.TestCase):
  def _setUpFiles(self, num_files, num_records_per_file):
    self._num_files = num_files
    self._num_records = num_records_per_file
    self._filenames = self._createFiles()
  @combinations.generate(test_base.default_test_combinations())
  def testFileShardingWithLegacyRebatch(self):
    self._setUpFiles(num_files=5, num_records_per_file=10)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.apply(
        testing.assert_next(["Shard", "FlatMap", "Batch", "Rebatch"]))
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)
    dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    expected = [[self._record(3, i)] for i in range(10)]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(test_base.default_test_combinations())
  def testFileShardingWithRebatch(self):
    self._setUpFiles(num_files=3, num_records_per_file=5)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.apply(
        testing.assert_next(["Shard", "FlatMap", "Batch", "Rebatch"]))
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)
    dataset = distribute._RebatchDataset(dataset, batch_sizes=[2, 1, 2])
    dataset = distribute._AutoShardDataset(dataset, 3, 1)
    expected = [[self._record(1, 0), self._record(1, 1)], [self._record(1, 2)],
                [self._record(1, 3), self._record(1, 4)]]
    self.assertDatasetProduces(dataset, expected)
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.times(
              combinations.combine(sharding_policy=[
                  options_lib.AutoShardPolicy.DATA,
                  options_lib.AutoShardPolicy.AUTO
              ]), combinations.combine(with_prefetch=[True, False]))))
  def testUseLegacyRebatchWithDataSharding(self, sharding_policy,
                                           with_prefetch):
    dataset = dataset_ops.Dataset.range(8)
    dataset = dataset.batch(4)
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = sharding_policy
    dataset = dataset.with_options(options)
    worker_a_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 1, 1])
    if with_prefetch:
      worker_a_dataset = worker_a_dataset.prefetch(1)
    worker_a_dataset = distribute._AutoShardDataset(
        worker_a_dataset, 3, 0, num_replicas=3)
    expected = [[0, 1], [4, 5]]
    self.assertDatasetProduces(worker_a_dataset, expected)
    worker_b_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[1, 1, 2])
    if with_prefetch:
      worker_b_dataset = worker_b_dataset.prefetch(1)
    worker_b_dataset = distribute._AutoShardDataset(
        worker_b_dataset, 3, 1, num_replicas=3)
    expected = [[2, 3], [6, 7]]
    self.assertDatasetProduces(worker_b_dataset, expected)
    worker_c_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[1, 2, 1])
    if with_prefetch:
      worker_c_dataset = worker_c_dataset.prefetch(1)
    worker_c_dataset = distribute._AutoShardDataset(
        worker_c_dataset, 3, 2, num_replicas=3)
    expected = [[], []]
    self.assertDatasetProduces(worker_c_dataset, expected)
class AutoShardDatasetCheckpointTest(tf_record_test_base.TFRecordTestBase,
                                     checkpoint_test_base.CheckpointTestBase,
                                     parameterized.TestCase):
  def setUp(self):
    super(AutoShardDatasetCheckpointTest, self).setUp()
    self._num_files = 10
    self._num_records = 10
    self._filenames = self._createFiles()
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    def build_dataset():
      dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
      dataset = dataset.apply(
          interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
      dataset = distribute._AutoShardDataset(dataset, 5, 3)
      return dataset
    verify_fn(self, build_dataset, num_outputs=20)
if __name__ == "__main__":
  test.main()
