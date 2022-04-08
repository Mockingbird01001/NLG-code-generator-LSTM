
import copy
import os
from absl.testing import parameterized
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.saving import save as keras_save
from tensorflow.python.module import module
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import save as tf_save
source_combination = combinations.combine(source=["textfile", "keyvaluetensor"])
source_and_load_combination = combinations.combine(
    source=["textfile", "keyvaluetensor"], load=["tf_load", "keras_load"])
class DistributedTableTest(test.TestCase, parameterized.TestCase):
  @classmethod
  def setUpClass(cls):
    super(DistributedTableTest, cls).setUpClass()
    cls.cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=2, num_ps=3, rpc_layer="grpc")
    cls.cluster_resolver = cls.cluster.cluster_resolver
  @classmethod
  def tearDownClass(cls):
    super(DistributedTableTest, cls).tearDownClass()
    cls.cluster.stop()
  def make_initializer(self, init_source, vals):
    if init_source == "textfile":
      file = os.path.join(self.get_temp_dir(), "text_file_initializer")
      with open(file, "w") as f:
        f.write("\n".join(str(v) for v in vals) + "\n")
      return lookup_ops.TextFileInitializer(
          filename=file,
          key_dtype=dtypes.int64,
          key_index=lookup_ops.TextFileIndex.LINE_NUMBER,
          value_dtype=dtypes.int64,
          value_index=lookup_ops.TextFileIndex.WHOLE_LINE)
    elif init_source == "keyvaluetensor":
      keys_tensor = constant_op.constant(
          list(range(len(vals))), dtype=dtypes.int64)
      vals_tensor = constant_op.constant(vals)
      return lookup_ops.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    else:
      raise ValueError("Unrecognized init_source: " + init_source)
  def createStaticHashTable(self,
                            init_source=None,
                            vals=None,
                            default_value=None,
                            initializer=None):
    if not initializer:
      initializer = self.make_initializer(init_source, vals)
    return lookup_ops.StaticHashTable(
        initializer=initializer, default_value=default_value)
  def makeDatasetFromTensorWithoutUsingResource(self, input_context, tensor):
    global_batch_size = 24
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset_ops.DatasetV2.from_tensors(tensor).repeat().batch(
        batch_size, drop_remainder=True)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)
    return dataset
  @combinations.generate(source_combination)
  def testCreateDistributedTableInScope(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator_lib.ClusterCoordinator(strategy=strategy)
    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)
    self.assertIsInstance(lookuptable, ps_values.DistributedTable)
    self.assertEqual(self.evaluate(lookuptable.size()), 3)
    output = lookuptable.lookup(
        constant_op.constant([0, 1, -1], dtype=dtypes.int64))
    self.assertAllEqual([0, 1, -2], output)
    self.assertEqual(lookuptable.size(), 3)
  @combinations.generate(source_combination)
  def testCopyDistributedTable(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator_lib.ClusterCoordinator(strategy=strategy)
    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)
    new_table = copy.copy(lookuptable)
    self.assertDictEqual(lookuptable.__dict__, new_table.__dict__)
  @combinations.generate(source_combination)
  def testCreateLookupInDatasetFnUnderScope(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)
    with strategy.scope():
      def dataset_fn(input_context):
        some_out_of_range_tensor = constant_op.constant(10, dtype=dtypes.int64)
        lookuptable = self.createStaticHashTable(
            init_source=source, vals=[0, 1, 2], default_value=-2)
        self.assertNotIsInstance(lookuptable, ps_values.DistributedTable)
        generation_tensor = lookuptable.lookup(some_out_of_range_tensor)
        dataset = self.makeDatasetFromTensorWithoutUsingResource(
            input_context, generation_tensor)
        return dataset
      @def_function.function
      def per_worker_dataset_fn():
        return strategy.distribute_datasets_from_function(dataset_fn)
      per_worker_dataset = coordinator.create_per_worker_dataset(
          per_worker_dataset_fn)
      per_worker_iterator = iter(per_worker_dataset)
      @def_function.function
      def worker_fn(iterator):
        return math_ops.reduce_sum(next(iterator))
      result = []
      for _ in range(10):
        result.append(
            coordinator.schedule(worker_fn, args=(per_worker_iterator,)))
      for r in result:
        returned_input = r.fetch()
        self.assertAllClose(-48, returned_input)
  @combinations.generate(source_combination)
  def testAccessingResourceHandleInDatasetFnWithoutMap(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)
    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)
    def dataset_fn(input_context):
      some_out_of_range_tensor = constant_op.constant(10, dtype=dtypes.int64)
      self.assertIsInstance(lookuptable, ps_values.DistributedTable)
      generation_tensor = lookuptable.lookup(some_out_of_range_tensor)
      dataset = self.makeDatasetFromTensorWithoutUsingResource(
          input_context, generation_tensor)
      return dataset
    @def_function.function
    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(dataset_fn)
    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)
    @def_function.function
    def worker_fn(iterator):
      return math_ops.reduce_sum(next(iterator))
    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))
    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-48, returned_input)
  @combinations.generate(source_combination)
  def testAccessingResourceHandleInDatasetFnWithMapFnDefinedInside(
      self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)
    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)
    def dataset_fn(input_context):
      generation_tensor = constant_op.constant([0, 1, 3], dtype=dtypes.int64)
      dataset = self.makeDatasetFromTensorWithoutUsingResource(
          input_context, generation_tensor)
      dataset = dataset.map(lookuptable.lookup)
      return dataset
    @def_function.function
    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(dataset_fn)
    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)
    @def_function.function
    def worker_fn(iterator):
      return math_ops.reduce_sum(next(iterator))
    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))
    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)
  @combinations.generate(source_combination)
  def testAccessingResourceHandleInDatasetFnWithMapFnDefinedOutside(
      self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)
    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)
    def map_fn(vals):
      return lookuptable.lookup(vals)
    def dataset_fn(input_context):
      generation_tensor = constant_op.constant([0, 1, 3], dtype=dtypes.int64)
      dataset = self.makeDatasetFromTensorWithoutUsingResource(
          input_context, generation_tensor)
      dataset = dataset.map(map_fn)
      return dataset
    @def_function.function
    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(dataset_fn)
    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)
    @def_function.function
    def worker_fn(iterator):
      return math_ops.reduce_sum(next(iterator))
    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))
    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)
  class Model(module.Module):
    def __init__(self, init_source, filepath):
      vals = [0, 1, 2]
      if init_source == "textfile":
        with open(filepath, "w") as f:
          f.write("\n".join(str(v) for v in vals) + "\n")
        self.initializer = lookup_ops.TextFileInitializer(
            filepath, dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER,
            dtypes.int64, lookup_ops.TextFileIndex.WHOLE_LINE)
      else:
        keys_tensor = constant_op.constant(
            list(range(len(vals))), dtype=dtypes.int64)
        vals_tensor = constant_op.constant(vals)
        self.initializer = lookup_ops.KeyValueTensorInitializer(
            keys_tensor, vals_tensor)
      self.table = lookup_ops.StaticHashTable(
          self.initializer, default_value=-2)
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int64)])
    def use_table(self, x):
      return self.table.lookup(x)
  def verifyWorkerLocalInstance(self, coordinator, model):
    for worker in coordinator._cluster.workers:
      with coordinator_context.with_dispatch_context(worker):
        captures = model.use_table.get_concrete_function().captured_inputs
        resource_capture = [t for t in captures if t.dtype == dtypes.resource]
        self.assertNotEmpty(resource_capture)
        for capture in resource_capture:
          self.assertEqual(
              capture.device,
              device_util.canonicalize("/CPU:0", default=worker.device_name))
  @combinations.generate(source_combination)
  def testInModelAndCapture(self, source):
    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    model = self.Model(source, file_path)
    func_captures = model.use_table.get_concrete_function(
    ).graph.external_captures
    self.assertLen(func_captures, 2)
    self.assertTrue(
        any(model.table.resource_handle is t for t in func_captures))
    deferred_captures = model.use_table.get_concrete_function(
    ).graph.deferred_external_captures
    self.assertEmpty(deferred_captures)
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy)
    with strategy.scope():
      distributed_model = self.Model("value", file_path)
    func_captures = distributed_model.use_table.get_concrete_function(
    ).graph.external_captures
    self.assertLen(func_captures, 1)
    self.assertFalse(
        any(model.table.resource_handle is t for t in func_captures))
    deferred_captures = distributed_model.use_table.get_concrete_function(
    ).graph.deferred_external_captures
    self.assertNotEmpty(deferred_captures)
    self.verifyWorkerLocalInstance(coordinator, distributed_model)
  @combinations.generate(source_and_load_combination)
  def testDistributeTableSaveAndServe(self, load, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with strategy.scope():
      model = self.Model(source, file_path)
    model_dir = self.get_temp_dir()
    tf_save.save(model, model_dir)
    if load == "tf_load":
      load_fn = tf_load.load
    else:
      load_fn = keras_save.load_model
    loaded_without_strategy = load_fn(model_dir)
    loaded_func_captures_without_strategy = (
        loaded_without_strategy.use_table.get_concrete_function().graph
        .external_captures)
    loaded_func_deferred_captures_without_strategy = (
        loaded_without_strategy.use_table.get_concrete_function().graph
        .deferred_external_captures)
    self.assertLen(loaded_func_captures_without_strategy, 2)
    self.assertEmpty(loaded_func_deferred_captures_without_strategy)
    self.assertAllEqual(
        loaded_without_strategy.use_table(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64)), [0, 1, -2])
  @combinations.generate(source_and_load_combination)
  def testDistributeTableSaveAndLoadUnderStrategy(self, load, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy)
    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with strategy.scope():
      model = self.Model(source, file_path)
    model_dir = self.get_temp_dir()
    tf_save.save(model, model_dir)
    if load == "tf_load":
      load_fn = tf_load.load
    else:
      load_fn = keras_save.load_model
    with strategy.scope():
      loaded = load_fn(model_dir)
    loaded_func_captures = (
        loaded.use_table.get_concrete_function().graph.external_captures)
    loaded_func_deferred_captures = (
        loaded.use_table.get_concrete_function().graph
        .deferred_external_captures)
    self.assertLen(loaded_func_captures, 1)
    self.assertNotEmpty(loaded_func_deferred_captures)
    self.assertIsInstance(loaded.table, ps_values.DistributedTable)
    self.assertLen([
        t for t in loaded.use_table.get_concrete_function().captured_inputs
        if t.dtype == dtypes.resource
    ], 1)
    self.verifyWorkerLocalInstance(coordinator, loaded)
if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
