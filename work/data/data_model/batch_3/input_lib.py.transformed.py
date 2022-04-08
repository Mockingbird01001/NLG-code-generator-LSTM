
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util.deprecation import deprecated
class DistributedDatasetV1(input_lib.DistributedDataset):
  def __init__(self,
               dataset,
               input_workers,
               strategy,
               num_replicas_in_sync=None,
               input_context=None,
               options=None):
    self._input_workers = input_workers
    super(DistributedDatasetV1, self).__init__(
        input_workers,
        strategy,
        dataset,
        num_replicas_in_sync=num_replicas_in_sync,
        input_context=input_context,
        options=options)
  def make_one_shot_iterator(self):
    return self._make_one_shot_iterator()
  def _make_one_shot_iterator(self):
    if not context.executing_eagerly():
      raise ValueError("Cannot create a one shot iterator. Please use "
                       "`make_initializable_iterator()` instead.")
    return self._get_iterator()
  def make_initializable_iterator(self):
    """Get an initializable iterator for DistributedDatasetV1.
    Note: This API is deprecated. Please use
    `tf.compat.v1.data.make_initializable_iterator(dataset)` to create an
    initializable iterator.
    Returns:
      A DistributedIteratorV1 instance.
    """
    return self._make_initializable_iterator()
    if context.executing_eagerly():
      raise ValueError("Cannot create initializable iterator in Eager mode. "
                       "Please use `iter()` instead.")
    return self._get_iterator()
  def _get_iterator(self):
    worker_iterators = _create_iterators_per_worker(self._cloned_datasets,
                                                    self._input_workers,
                                                    self._options)
    iterator = DistributedIteratorV1(self._input_workers, worker_iterators,
                                     self._strategy, cardinality,
                                     self._enable_get_next_as_optional)
    if context.executing_eagerly():
      context.async_wait()
    return iterator
  def __iter__(self):
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      return self._get_iterator()
    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")
class DistributedDatasetsFromFunctionV1(
    input_lib.DistributedDatasetsFromFunction):
  def _make_initializable_iterator(self, shared_name=None):
    if context.executing_eagerly():
      raise ValueError("Cannot create initializable iterator in Eager mode. "
                       "Please use `iter()` instead.")
    return self._get_iterator()
  def _make_one_shot_iterator(self):
    if not context.executing_eagerly():
      raise ValueError("Cannot create a one shot iterator. Please use "
                       "`make_initializable_iterator()` instead.")
    return self._get_iterator()
  def _get_iterator(self):
    iterators = _create_iterators_per_worker(self._datasets,
                                             self._input_workers, self._options)
    iterator = DistributedIteratorV1(self._input_workers, iterators,
                                     self._strategy, cardinality,
                                     self._enable_get_next_as_optional)
    if context.executing_eagerly():
      context.async_wait()
    return iterator
  def __iter__(self):
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      return self._get_iterator()
    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")
class DistributedIteratorV1(input_lib.DistributedIteratorBase):
  @property
  def _initializer(self):
    init_ops = []
    for it in self._iterators:
      init_ops.extend(it.initialize())
    return control_flow_ops.group(init_ops)
  @deprecated(None, "Use the iterator's `initializer` property instead.")
  def initialize(self):
    return self._initializer
  @property
  def initializer(self):
    return self.initialize()
  @property
  def output_classes(self):
    return self._iterators[0].output_classes
  @property
  def output_shapes(self):
    return self._iterators[0].output_shapes
  @property
  def output_types(self):
    return self._iterators[0].output_types
  def get_iterator(self, worker):
    for i, w in enumerate(self._input_workers.worker_devices):
      if worker == w:
        return self._iterators[i]
    return None
  @property
  def element_spec(self):
    return self._element_spec
class DatasetIterator(DistributedIteratorV1):
  def __init__(self,
               dataset,
               input_workers,
               strategy,
               num_replicas_in_sync=None,
               input_context=None):
    """Make an iterator for the dataset on given devices.
    If `num_replicas_in_sync` is not None, we split each batch of the dataset
    into `num_replicas_in_sync` smaller batches, to be distributed among that
    worker's replicas, so that the batch size for a global step (across all
    workers and replicas) is as expected.
    Args:
      dataset: `tf.data.Dataset` that will be used as the input source.
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      num_replicas_in_sync: Optional integer. If this is not None, the value is
        used to decide how to rebatch datasets into smaller batches so that the
        total batch size for each step (across all workers and replicas) adds up
        to `dataset`'s batch size.
      input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.
    """
    dist_dataset = DistributedDatasetV1(
        dataset,
        input_workers,
        strategy,
        num_replicas_in_sync=num_replicas_in_sync,
        input_context=input_context)
    worker_iterators = _create_iterators_per_worker(
        dist_dataset._cloned_datasets, input_workers)
    super(DatasetIterator,
          self).__init__(input_workers, worker_iterators, strategy,
                         dist_dataset.cardinality,
                         dist_dataset._enable_get_next_as_optional)
    self._element_spec = dist_dataset.element_spec
class InputFunctionIterator(DistributedIteratorV1):
  def __init__(self, input_fn, input_workers, input_contexts, strategy):
    """Make an iterator for input provided via an input function.
    Currently implements PER_WORKER mode, in which the `input_fn` is called
    once on each worker.
    TODO(priyag): Add other replication modes.
    Args:
      input_fn: Input function that returns a `tf.data.Dataset` object.
      input_workers: an `InputWorkers` object.
      input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `input_fn`. Length and order should match worker order in
        `worker_device_pairs`.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
    """
    assert isinstance(input_workers, input_lib.InputWorkers)
    if input_workers.num_workers != len(input_contexts):
      raise ValueError("Number of input workers (%d) is not same as number of "
                       "input_contexts (%d)" %
                       (input_workers.num_workers, len(input_contexts)))
    iterators = []
    for i, ctx in enumerate(input_contexts):
      worker = input_workers.worker_devices[i]
      with ops.device(worker):
        result = input_fn(ctx)
        devices = input_workers.compute_devices_for_worker(i)
        if isinstance(result, dataset_ops.DatasetV2):
          iterator = _SingleWorkerDatasetIterator(result, worker, devices)
        elif callable(result):
          iterator = _SingleWorkerCallableIterator(result, worker, devices)
        else:
          raise ValueError(
              "input_fn must return a tf.data.Dataset or a callable.")
        iterators.append(iterator)
    super(InputFunctionIterator, self).__init__(
        input_workers,
        iterators,
        strategy,
        cardinality=cardinality_lib.UNKNOWN,
        enable_get_next_as_optional=False)
    self._enable_get_next_as_optional = False
  def _make_iterator(self):
    with ops.device(self._worker):
      if self._options is not None:
        self._iterator = multi_device_iterator_ops.MultiDeviceIterator(
            self._dataset,
            self._devices,
            max_buffer_size=self._options.experimental_per_replica_buffer_size,
            prefetch_buffer_size=self._options
            .experimental_per_replica_buffer_size)
      else:
        self._iterator = multi_device_iterator_ops.MultiDeviceIterator(
            self._dataset,
            self._devices,
        )
  def initialize(self):
    if ops.executing_eagerly_outside_functions():
      return []
    else:
      return [self._iterator.initializer]
  @property
  def output_classes(self):
    return dataset_ops.get_legacy_output_classes(self._iterator)
  @property
  def output_shapes(self):
    return dataset_ops.get_legacy_output_shapes(self._iterator)
  @property
  def output_types(self):
    return dataset_ops.get_legacy_output_types(self._iterator)
class _SingleWorkerCallableIterator(object):
  def __init__(self, fn, worker, devices):
    self._fn = fn
    self._worker = worker
    self._devices = devices
  def get_next(self, device, name=None):
    del device, name
    with ops.device(self._worker):
      return self._fn()
  def get_next_as_list(self, name=None):
    del name
    with ops.device(self._worker):
      data_list = [self._fn() for _ in self._devices]
      return data_list
  def get_next_as_optional_list(self):
    with ops.device(self._worker):
      data_list = [
          optional_ops.Optional.from_value(self._fn()) for _ in self._devices
      ]
      return data_list
  def initialize(self):
    return []
def _create_iterators_per_worker(worker_datasets, input_workers, options=None):
  assert isinstance(input_workers, input_lib.InputWorkers)
  assert len(worker_datasets) == len(input_workers.worker_devices)
  iterators = []
  for i, worker in enumerate(input_workers.worker_devices):
    with ops.device(worker):
      worker_devices = input_workers.compute_devices_for_worker(i)
      iterator = _SingleWorkerDatasetIterator(
          worker,
          worker_devices,
          options)
      iterators.append(iterator)
  return iterators
