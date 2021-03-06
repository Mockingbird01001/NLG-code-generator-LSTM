
import copy
import threading
import time
import weakref
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.training.tracking import base
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export("distribute.MultiWorkerMirroredStrategy", v1=[])
class CollectiveAllReduceStrategy(distribute_lib.Strategy):
  """A distribution strategy for synchronous training on multiple workers.
  This strategy implements synchronous distributed training across multiple
  workers, each with potentially multiple GPUs. Similar to
  `tf.distribute.MirroredStrategy`, it replicates all variables and computations
  to each local device. The difference is that it uses a distributed collective
  implementation (e.g. all-reduce), so that multiple workers can work together.
  You need to launch your program on each worker and configure
  `cluster_resolver` correctly. For example, if you are using
  `tf.distribute.cluster_resolver.TFConfigClusterResolver`, each worker needs to
  have its corresponding `task_type` and `task_id` set in the `TF_CONFIG`
  environment variable. An example TF_CONFIG on worker-0 of a two worker cluster
  is:
  ```
  TF_CONFIG = '{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 0} }'
  ```
  Your program runs on each worker as-is. Note that collectives require each
  worker to participate. All `tf.distribute` and non `tf.distribute` API may use
  collectives internally, e.g. checkpointing and saving since reading a
  `tf.Variable` with `tf.VariableSynchronization.ON_READ` all-reduces the value.
  Therefore it's recommended to run exactly the same program on each worker.
  Dispatching based on `task_type` or `task_id` of the worker is error-prone.
  `cluster_resolver.num_accelerators()` determines the number of GPUs the
  strategy uses. If it's zero, the strategy uses the CPU. All workers need to
  use the same number of devices, otherwise the behavior is undefined.
  This strategy is not intended for TPU. Use `tf.distribute.TPUStrategy`
  instead.
  After setting up TF_CONFIG, using this strategy is similar to using
  `tf.distribute.MirroredStrategy` and `tf.distribute.TPUStrategy`.
  ```
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(2, input_shape=(5,)),
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
  def dataset_fn(ctx):
    x = np.random.random((2, 5)).astype(np.float32)
    y = np.random.randint(2, size=(2, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.repeat().batch(1, drop_remainder=True)
  dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
  model.compile()
  model.fit(dist_dataset)
  ```
  You can also write your own training loop:
  ```
  @tf.function
  def train_step(iterator):
    def step_fn(inputs):
      features, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(features, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
    strategy.run(step_fn, args=(next(iterator),))
  for _ in range(NUM_STEP):
    train_step(iterator)
  ```
  See
  [Multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
  for a detailed tutorial.
  __Saving__
  You need to save and checkpoint on all workers instead of just one. This is
  because variables whose synchronization=ON_READ triggers aggregation during
  saving. It's recommended to save to a different path on each worker to avoid
  race conditions. Each worker saves the same thing. See
  tutorial for examples.
  __Known Issues__
  * `tf.distribute.cluster_resolver.TFConfigClusterResolver` does not return the
  correct number of accelerators. The strategy uses all available GPUs if
  `cluster_resolver` is `tf.distribute.cluster_resolver.TFConfigClusterResolver`
  or `None`.
  * In eager mode, the strategy needs to be created before calling any other
  Tensorflow API.
  """
  _collective_key_base = 0
  def __init__(self,
               cluster_resolver=None,
               communication_options=None):
    if communication_options is None:
      communication_options = collective_util.Options()
    super(CollectiveAllReduceStrategy, self).__init__(
        CollectiveAllReduceExtended(
            self,
            cluster_resolver=cluster_resolver,
            communication_options=communication_options))
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "MultiWorkerMirroredStrategy")
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended._num_devices_per_worker)
  @classmethod
  def _from_local_devices(cls, devices, communication_options=None):
    obj = cls(communication_options=communication_options)
    return obj
  @property
  def cluster_resolver(self):
class _CollectiveAllReduceStrategyExperimentalMeta(type):
  @classmethod
  def __instancecheck__(cls, instance):
    return isinstance(instance, CollectiveAllReduceStrategy)
@tf_export("distribute.experimental.MultiWorkerMirroredStrategy", v1=[])
class _CollectiveAllReduceStrategyExperimental(
    CollectiveAllReduceStrategy,
    metaclass=_CollectiveAllReduceStrategyExperimentalMeta):
  __doc__ = CollectiveAllReduceStrategy.__doc__
  @deprecation.deprecated(
      None, "use distribute.MultiWorkerMirroredStrategy instead")
  def __init__(self,
               communication=collective_util.CommunicationImplementation.AUTO,
               cluster_resolver=None):
    communication_options = collective_util.Options(
        implementation=communication)
    super(_CollectiveAllReduceStrategyExperimental,
          self).__init__(cluster_resolver, communication_options)
  @classmethod
  def _from_local_devices(
      cls,
      devices,
      communication=collective_util.CommunicationImplementation.AUTO):
    obj = cls(communication)
    return obj
_CollectiveAllReduceStrategyExperimental.__name__ = CollectiveAllReduceStrategy.__name__
class CollectiveAllReduceStrategyV1(distribute_lib.StrategyV1):
  __doc__ = CollectiveAllReduceStrategy.__doc__
  _collective_key_base = 0
  def __init__(self,
               communication=collective_util.CommunicationImplementation.AUTO,
               cluster_resolver=None):
    communication_options = collective_util.Options(
        implementation=communication)
    super(CollectiveAllReduceStrategyV1, self).__init__(
        CollectiveAllReduceExtended(
            self,
            cluster_resolver=cluster_resolver,
            communication_options=communication_options))
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set(
        "MultiWorkerMirroredStrategy")
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_gpu_per_worker").set(
            self.extended._num_devices_per_worker
            if self.extended._local_device_type == "GPU"
            else 0)
class CollectiveAllReduceExtended(mirrored_strategy.MirroredExtended):
  _enable_check_health = True
  _check_health_interval = 30
  _check_health_initial_timeout = 0
  _check_health_retry_limit = 3
  _check_health_timeout = 10
  def __init__(self, container_strategy, cluster_resolver,
               communication_options):
    if not isinstance(communication_options, collective_util.Options):
      raise ValueError("communication_options must be an instance of "
                       "tf.distribute.experimental.CommunicationOptions")
    self._cluster_resolver = cluster_resolver or TFConfigClusterResolver()
    if not isinstance(self._cluster_resolver, ClusterResolver):
      raise ValueError("cluster_resolver must be an instance of "
                       "tf.distribute.cluster_resolver.ClusterResolver")
    distribute_lib.StrategyExtendedV1.__init__(self, container_strategy)
    self._communication_options = communication_options
    self._initialize_strategy(self._cluster_resolver)
    self._cfer_fn_cache = weakref.WeakKeyDictionary()
    self.experimental_enable_get_next_as_optional = True
    assert isinstance(self._cross_device_ops,
                      cross_device_ops_lib.CollectiveAllReduce)
  def _use_merge_call(self):
    return True
  def _initialize_strategy(self, cluster_resolver):
    if cluster_resolver.cluster_spec().as_dict():
      self._initialize_multi_worker(cluster_resolver)
    else:
      self._initialize_local(cluster_resolver)
  def _initialize_local_devices(self, cluster_resolver, worker_device):
    if isinstance(cluster_resolver, TFConfigClusterResolver):
      num_gpus = context.num_gpus()
      num_tpus = 0
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)
      num_tpus = cluster_resolver.num_accelerators().get("TPU", 0)
    if num_gpus:
      local_device_type = "GPU"
      num_local_devices = num_gpus
    elif num_tpus:
      local_device_type = "TPU"
      num_local_devices = num_tpus
    else:
      local_device_type = "CPU"
      num_local_devices = 1
    local_devices = tuple(
        f"{worker_device}/device:{local_device_type}:{i}"
        for i in range(num_local_devices))
    return local_devices, local_device_type
  def _initialize_local(self, cluster_resolver, devices=None):
    self._is_chief = True
    self._num_workers = 1
    if ops.executing_eagerly_outside_functions():
      try:
        context.context().configure_collective_ops(
            scoped_allocator_enabled_ops=("CollectiveReduce",))
      except RuntimeError:
        logging.warning("Collective ops is not configured at program startup. "
                        "Some performance features may not be enabled.")
      self._collective_ops_configured = True
    if devices:
      local_devices = devices
      if "GPU" in devices[0]:
        local_device_type = "GPU"
      elif "TPU" in devices[0]:
        local_device_type = "TPU"
      else:
        local_device_type = "CPU"
    else:
      local_devices, local_device_type = self._initialize_local_devices(
          cluster_resolver, worker_device="")
    self._worker_device = device_util.canonicalize("/device:CPU:0")
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)
    self._collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=1 + self._collective_key_base)
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=local_devices,
        group_size=len(local_devices),
        options=self._communication_options,
        collective_keys=self._collective_keys)
    self._host_cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=[self._worker_device],
        group_size=self._num_workers,
        options=self._communication_options,
        collective_keys=self._collective_keys)
    super(CollectiveAllReduceExtended, self)._initialize_single_worker(
        local_devices)
    self._cluster_spec = None
    self._task_type = None
    self._task_id = None
    self._id_in_cluster = 0
    self._local_or_standalone_client_mode = True
    self._num_devices_per_worker = len(local_devices)
    self._local_device_type = local_device_type
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()
    logging.info(
        "Single-worker MultiWorkerMirroredStrategy with local_devices "
        "= %r, communication = %s", local_devices,
        self._communication_options.implementation)
  def _initialize_multi_worker(self, cluster_resolver):
    cluster_spec = multi_worker_util.normalize_cluster_spec(
        cluster_resolver.cluster_spec())
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    if task_type is None or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`.")
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._id_in_cluster = multi_worker_util.id_in_cluster(
        self._cluster_spec, self._task_type, self._task_id)
    self._num_workers = multi_worker_util.worker_count(cluster_spec, task_type)
    if not self._num_workers:
      raise ValueError("No `worker`, `chief` or `evaluator` tasks can be found "
                       "in `cluster_spec`.")
    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)
    self._worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)
    if (ops.executing_eagerly_outside_functions() and
        not getattr(self, "_local_or_standalone_client_mode", False)):
      context.context().configure_collective_ops(
          collective_leader=multi_worker_util.collective_leader(
              cluster_spec, task_type, task_id),
          scoped_allocator_enabled_ops=("CollectiveReduce",),
          device_filters=("/job:%s/task:%d" % (task_type, task_id),))
      self._collective_ops_configured = True
      if context.context().coordination_service is None:
        coordinated_jobs = ["chief", "worker"]
        if task_type in coordinated_jobs:
          context.context().configure_coordination_service(
              service_type="standalone",
              service_leader=multi_worker_util.coordination_leader(
                  cluster_spec),
              coordinated_jobs=coordinated_jobs)
    if (context.executing_eagerly() and
        not getattr(self, "_std_server_started", False) and
        not getattr(self, "_local_or_standalone_client_mode", False)):
      config_proto = copy.deepcopy(context.context().config)
      config_proto = self._update_config_proto(config_proto)
      if config_proto.experimental.coordination_config.service_type:
        self._enable_check_health = False
      if hasattr(cluster_resolver, "port"):
        port = cluster_resolver.port
      else:
        port = 0
      server_def = tensorflow_server_pb2.ServerDef(
          cluster=cluster_spec.as_cluster_def(),
          default_session_config=config_proto,
          job_name=task_type,
          task_index=task_id,
          protocol=cluster_resolver.rpc_layer or "grpc",
          port=port)
      context.context().enable_collective_ops(server_def)
      self._std_server_started = True
      context.context().ensure_initialized()
      logging.info(
          "Enabled multi-worker collective ops with available devices: %r",
          context.context().devices())
    local_devices, local_device_type = self._initialize_local_devices(
        cluster_resolver, self._worker_device)
    if local_device_type == "TPU":
      tpu_strategy_util.initialize_tpu_system()
    self._collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=1 + self._collective_key_base)
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=local_devices,
        group_size=len(local_devices) * self._num_workers,
        options=self._communication_options,
        collective_keys=self._collective_keys)
    self._host_cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=[self._worker_device],
        group_size=self._num_workers,
        options=self._communication_options,
        collective_keys=self._collective_keys)
    super(CollectiveAllReduceExtended, self)._initialize_single_worker(
        local_devices)
    self._default_device = "/job:%s/task:%d" % (task_type, task_id)
    self._num_devices_per_worker = len(local_devices)
    self._local_device_type = local_device_type
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()
    if self._enable_check_health and context.executing_eagerly():
      self._start_check_health_thread()
    else:
      logging.info("Check health not enabled.")
    logging.info(
        "MultiWorkerMirroredStrategy with cluster_spec = %r, task_type = %r, "
        "task_id = %r, num_workers = %r, local_devices = %r, "
        "communication = %s", cluster_spec.as_dict(), task_type, task_id,
        self._num_workers, local_devices,
        self._communication_options.implementation)
  def __del__(self):
    self._stop_check_health_thread()
  def _input_workers_with_options(self, options=None):
    host_device = device_util.get_host_for_device(self._worker_device)
    if not options or options.experimental_fetch_to_device:
      return input_lib.InputWorkers([(host_device, self.worker_devices)])
    else:
      return input_lib.InputWorkers([(
          host_device,
          [device_util.get_host_for_device(worker) for worker in
           self.worker_devices])])
  @property
  def _input_workers(self):
    return self._input_workers_with_options()
  def _get_variable_creator_initial_value(self,
                                          replica_id,
                                          device,
                                          primary_var,
                                          **kwargs):
      assert device is not None
      assert primary_var is None
        group_key = self._collective_keys.get_group_key([device])
        group_size = self._num_workers
        collective_instance_key = (
            self._collective_keys.get_instance_key(group_key, device))
        with ops.device(device):
          initial_value = kwargs["initial_value"]
          if callable(initial_value):
            initial_value = initial_value()
          if isinstance(initial_value, base.CheckpointInitialValue):
            initial_value = initial_value.wrapped_value
          assert not callable(initial_value)
          initial_value = ops.convert_to_tensor(
              initial_value, dtype=kwargs.get("dtype", None))
          if self._num_workers > 1:
            if self._is_chief:
              bcast_send = collective_ops.broadcast_send(
                  initial_value, initial_value.shape, initial_value.dtype,
                  group_size, group_key, collective_instance_key)
              with ops.control_dependencies([bcast_send]):
                return array_ops.identity(initial_value)
            else:
              return collective_ops.broadcast_recv(initial_value.shape,
                                                   initial_value.dtype,
                                                   group_size, group_key,
                                                   collective_instance_key)
          return initial_value
      return initial_value_fn
    else:
      return super(CollectiveAllReduceExtended,
                   self)._get_variable_creator_initial_value(
                       replica_id=replica_id,
                       device=device,
                       primary_var=primary_var,
                       **kwargs)
  def _make_input_context(self):
    input_context = distribute_lib.InputContext(
        num_input_pipelines=self._num_workers,
        input_pipeline_id=self._id_in_cluster,
        num_replicas_in_sync=self._num_replicas_in_sync)
    return input_context
  def _experimental_distribute_dataset(self, dataset, options):
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`distribute_datasets_from_function` "
          "of tf.distribute.MirroredStrategy"
      )
    input_context = self._make_input_context()
    return input_util.get_distributed_dataset(
        dataset,
        self._input_workers_with_options(options),
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync,
        input_context=input_context,
        options=options)
  def _distribute_datasets_from_function(self, dataset_fn, options):
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`distribute_datasets_from_function` "
          "of tf.distribute.MirroredStrategy")
    input_context = self._make_input_context()
    return input_util.get_distributed_datasets_from_function(
        dataset_fn=dataset_fn,
        input_workers=self._input_workers_with_options(options),
        input_contexts=[input_context],
        strategy=self._container_strategy(),
        options=options)
  def _experimental_distribute_values_from_function(self, value_fn):
    per_replica_values = []
    num_local_replicas = len(self.worker_devices)
    for local_replica_id in range(num_local_replicas):
      replica_id = (self._id_in_cluster * num_local_replicas +
                    local_replica_id)
      value_context = distribute_lib.ValueContext(
          replica_id, self._num_replicas_in_sync)
      per_replica_values.append(value_fn(value_context))
    return distribute_utils.regroup(per_replica_values, always_wrap=True)
  def _make_dataset_iterator(self, dataset):
    input_context = self._make_input_context()
    return input_lib_v1.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync,
        input_context=input_context)
  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    input_context = self._make_input_context()
    return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers,
                                              [input_context],
                                              self._container_strategy())
  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    if cluster_spec:
      cluster_resolver = SimpleClusterResolver(
          cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
          task_type=task_type,
          task_id=task_id,
          num_accelerators={
              self._local_device_type: self._num_devices_per_worker},
          rpc_layer=self._rpc_layer)
      self._initialize_multi_worker(cluster_resolver)
      assert isinstance(self._cross_device_ops,
                        cross_device_ops_lib.CollectiveAllReduce)
    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))
  def _update_config_proto(self, config_proto):
    updated_config = copy.deepcopy(config_proto)
    rewrite_options = updated_config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append("CollectiveReduce")
    if (not ops.executing_eagerly_outside_functions() and
        self._communication_options.implementation ==
        collective_util.CommunicationImplementation.NCCL):
      updated_config.experimental.collective_nccl = True
    if not self._cluster_spec:
      return updated_config
    assert self._task_type
    assert self._task_id is not None
    updated_config.experimental.collective_group_leader = (
        multi_worker_util.collective_leader(self._cluster_spec, self._task_type,
                                            self._task_id))
    del updated_config.device_filters[:]
    updated_config.device_filters.append(
        "/job:%s/task:%d" % (self._task_type, self._task_id))
    return updated_config
  def _get_cross_device_ops(self, value):
    if isinstance(value, values.DistributedValues):
    else:
      num_devices = 1
    if num_devices == len(self.worker_devices):
      return self._cross_device_ops
    else:
      return self._host_cross_device_ops
  def _gather_to_implementation(self, value, destinations, axis, options):
        value,
        destinations=destinations,
        axis=axis,
        options=options)
  def _reduce_to(self, reduce_op, value, destinations, options):
    if (isinstance(value, values.Mirrored) and
        reduce_op == reduce_util.ReduceOp.MEAN):
      return value
    assert not isinstance(value, values.Mirrored)
    if (isinstance(value, values.DistributedValues) and
        len(self.worker_devices) == 1):
      value = value.values[0]
    if (not isinstance(value, values.DistributedValues) and
        self._num_workers == 1):
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, value, destinations, len(self.worker_devices))
    return self._get_cross_device_ops(value).reduce(
        reduce_op,
        value,
        destinations=destinations,
        options=self._communication_options.merge(options))
  def _replica_ctx_all_reduce(self, reduce_op, value, options=None):
    if options is None:
      options = collective_util.Options()
    if context.executing_eagerly():
      return super()._replica_ctx_all_reduce(reduce_op, value, options)
    replica_context = ds_context.get_replica_context()
    assert replica_context, (
        "`StrategyExtended._replica_ctx_all_reduce` must be called in a "
        "replica context")
        reduce_op,
        value,
        options)
  def _check_health(self):
    while True:
      if self._check_health_thread_should_stop.is_set():
        return
      for job in self._cluster_spec.jobs:
        for task_id in range(self._cluster_spec.num_tasks(job)):
          peer = "/job:{}/replica:0/task:{}".format(job, task_id)
          attempts = 0
          while True:
            attempts += 1
            try:
              context.context().check_collective_ops_peer_health(
                  peer, timeout_in_ms=self._check_health_timeout * 1000)
              break
            except (errors.UnavailableError, errors.FailedPreconditionError,
                    errors.DeadlineExceededError) as e:
              if attempts < self._check_health_retry_limit:
                logging.warning("%s seems down, retrying %d/%d", peer, attempts,
                                self._check_health_retry_limit)
                continue
              logging.error(
                  "Cluster check alive failed, %s is down, "
                  "aborting collectives: %s", peer, e)
              context.context().abort_collective_ops(
                  errors.UNAVAILABLE,
                  "cluster check alive failed, {} is down".format(peer))
              return
              logging.error("Unexpected exception in check alive: %s", e)
              context.context().abort_collective_ops(
                  errors.INTERNAL,
                  "unexecpted exception in check alive: %s" % e)
              return
      time.sleep(self._check_health_interval)
  def _start_check_health_thread(self):
    dummy_value = array_ops.identity([])
    logging.info("Waiting for the cluster, timeout = %s",
                 self._check_health_initial_timeout or "inf")
    try:
      self._host_cross_device_ops.reduce(
          reduce_util.ReduceOp.SUM,
          dummy_value,
          dummy_value,
          options=collective_util.Options(
              timeout_seconds=self._check_health_initial_timeout,
              implementation=collective_util.CommunicationImplementation.RING))
      if context.is_async():
        context.async_wait()
    except errors.DeadlineExceededError:
      raise RuntimeError(
          "Timeout waiting for the cluster, timeout is %d seconds" %
          self._check_health_initial_timeout)
    logging.info("Cluster is ready.")
    self._check_health_thread_should_stop = threading.Event()
    self._check_health_thread = threading.Thread(
        target=self._check_health,
        daemon=True)
    self._check_health_thread.start()
  def _stop_check_health_thread(self):
    if getattr(self, "_check_health_thread", None):
      logging.info("stopping check health thread")
      self._check_health_thread_should_stop.set()
      self._check_health_thread.join()
      self._check_health_thread = None
      logging.info("check health thread stopped")
  def _warn_nccl_no_gpu(self):
    if ((self._communication_options.implementation ==
         collective_util.CommunicationImplementation.NCCL) and
        self._local_device_type != "GPU"):
      logging.warning("Enabled NCCL communication but no GPUs detected/"
                      "specified.")
  def _in_multi_worker_mode(self):
    return self._num_workers > 1
  @property
  def experimental_between_graph(self):
    return True
  @property
  def experimental_should_init(self):
    return True
  @property
  def should_checkpoint(self):
    return self._is_chief
  @property
  def should_save_summary(self):
    return self._is_chief
  @property
  def _num_replicas_in_sync(self):
    return len(self.worker_devices) * self._num_workers
  @property
  def _global_batch_size(self):
    return True
  def _get_replica_id_in_sync_group(self, replica_id):
    return self._id_in_cluster * len(self.worker_devices) + replica_id
  def _get_local_replica_id(self, replica_id_in_sync_group):
    return (replica_id_in_sync_group -
            self._id_in_cluster * len(self.worker_devices))
  def __deepcopy__(self, memo):
    if hasattr(self, "_check_health_thread"):
      raise ValueError(
          "MultiWorkerMirroredStrategy cannot be deep copied in eager mode. "
          "If you're using Estimator and see this error message, call "
          "tf.compat.v1.disable_eager_execution() at the beginning of your "
          "program")
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      setattr(result, k, copy.deepcopy(v, memo))
    return result
