
import tempfile
from absl import flags
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
TMP_WORK_DIR = "tmp_work_dir_placeholder"
NO_WORK_DIR = ""
TEST_HEARTBEAT_INTERVAL_MS = 100
TEST_DISPATCHER_TIMEOUT_MS = 1000
PROTOCOL = "grpc"
TRANSFER_PROTOCOL = flags.DEFINE_string(
    "tf_data_service_test_transfer_protocol", None, "Data plane protocol.")
def all_cluster_configurations():
  with_work_dir = combinations.combine(
      work_dir=TMP_WORK_DIR, fault_tolerant_mode=[True, False])
  without_work_dir = combinations.combine(
      work_dir=NO_WORK_DIR, fault_tolerant_mode=False)
  return with_work_dir + without_work_dir
def _make_worker(dispatcher_address,
                 data_transfer_protocol,
                 shutdown_quiet_period_ms=0,
                 port=0,
                 worker_tags=None):
  defaults = server_lib.WorkerConfig(dispatcher_address=dispatcher_address)
  config_proto = service_config_pb2.WorkerConfig(
      dispatcher_address=dispatcher_address,
      worker_address=defaults.worker_address,
      port=port,
      protocol=PROTOCOL,
      worker_tags=worker_tags,
      heartbeat_interval_ms=TEST_HEARTBEAT_INTERVAL_MS,
      dispatcher_timeout_ms=TEST_DISPATCHER_TIMEOUT_MS,
      data_transfer_protocol=data_transfer_protocol,
      data_transfer_address=defaults.worker_address,
      shutdown_quiet_period_ms=shutdown_quiet_period_ms)
  return server_lib.WorkerServer(config_proto, start=False)
class TestWorker(object):
  def __init__(self,
               dispatcher_address,
               shutdown_quiet_period_ms,
               data_transfer_protocol=None,
               worker_tags=None):
    self._dispatcher_address = dispatcher_address
    self._shutdown_quiet_period_ms = shutdown_quiet_period_ms
    self._server = _make_worker(
        dispatcher_address,
        data_transfer_protocol,
        shutdown_quiet_period_ms,
        worker_tags=worker_tags)
    self._running = False
    self._data_transfer_protocol = data_transfer_protocol
  def stop(self):
    self._server._stop()
    self._running = False
  def start(self):
    self._server.start()
    self._port = int(self._server._address.split(":")[1])
    self._running = True
  def restart(self, use_same_port=True):
    if self._running:
      self.stop()
    port = 0
    if use_same_port:
      port = self._port
    self._server = _make_worker(self._dispatcher_address,
                                self._data_transfer_protocol,
                                self._shutdown_quiet_period_ms, port)
    self._server.start()
    self._port = int(self._server._address.split(":")[1])
    self._running = True
  def join(self):
    self._server.join()
  def num_tasks(self):
    return self._server._num_tasks()
  def worker_address(self):
    return self._server._address
class TestCluster(object):
  def __init__(self,
               num_workers,
               dispatcher_port=0,
               work_dir=TMP_WORK_DIR,
               fault_tolerant_mode=True,
               job_gc_check_interval_ms=None,
               job_gc_timeout_ms=None,
               worker_shutdown_quiet_period_ms=0,
               start=True,
               data_transfer_protocol=None):
    """Creates a tf.data service test cluster.
    Args:
      num_workers: The number of workers to initially add to the cluster.
      dispatcher_port: The port to use for the dispatcher.
      work_dir: The work directory to use for the dispatcher. If set to
        `TMP_WORK_DIR`, the cluster will create a new temporary directory to use
        as the work directory. If set to `NO_WORK_DIR`, no work directory will
        be used.
      fault_tolerant_mode: Whether the dispatcher should write its state to a
        journal so that it can recover from restarts.
      job_gc_check_interval_ms: How often the dispatcher should scan through to
        delete old and unused jobs, in milliseconds.
      job_gc_timeout_ms: How long a job needs to be unused before it becomes a
        candidate for garbage collection, in milliseconds.
      worker_shutdown_quiet_period_ms: When shutting down a worker, how long to
        wait for the gRPC server to process the final requests.
      start: Whether to immediately start the servers in the cluster. If
        `False`, the servers can be started later by calling
        `start_dispatcher()` and `start_workers()`.
      data_transfer_protocol: (Optional.) The protocol to use for transferring
        data with the tf.data service. The default can controlled via
        tf_data_service_test_transfer_protocol flag.
    """
    if work_dir == TMP_WORK_DIR:
      work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    self._worker_shutdown_quiet_period_ms = worker_shutdown_quiet_period_ms
    if not data_transfer_protocol:
      data_transfer_protocol = TRANSFER_PROTOCOL.value
    self._data_transfer_protocol = data_transfer_protocol
    self.dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=dispatcher_port,
            work_dir=work_dir,
            protocol=PROTOCOL,
            fault_tolerant_mode=fault_tolerant_mode,
            job_gc_check_interval_ms=job_gc_check_interval_ms,
            job_gc_timeout_ms=job_gc_timeout_ms),
        start=start)
    self.workers = []
    for _ in range(num_workers):
      self.add_worker(start=start)
  def dispatcher_address(self):
    return self.dispatcher.target.split("://")[1]
  def add_worker(self, start=True):
    worker = TestWorker(self.dispatcher_address(),
                        self._worker_shutdown_quiet_period_ms,
                        self._data_transfer_protocol)
    if start:
      worker.start()
    self.workers.append(worker)
  def start_dispatcher(self):
    self.dispatcher.start()
  def start_workers(self):
    for worker in self.workers:
      worker.start()
  def stop_dispatcher(self):
    self.dispatcher._stop()
  def stop_workers(self):
    for worker in self.workers:
      worker.stop()
  def restart_dispatcher(self):
    if not self.dispatcher._config.fault_tolerant_mode:
      raise ValueError(
          "Trying to restart the dispatcher without fault-tolerance.")
    port = int(self.dispatcher_address().split(":")[1])
    self.dispatcher._stop()
    self.dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=port,
            work_dir=self.dispatcher._config.work_dir,
            protocol=PROTOCOL,
            fault_tolerant_mode=self.dispatcher._config.fault_tolerant_mode))
  def num_registered_workers(self):
    return self.dispatcher._num_workers()
  def num_tasks_on_workers(self):
    return sum(worker.num_tasks() for worker in self.workers)
  def __del__(self):
    self.workers.clear()
    del self.dispatcher
class TestBase(test_base.DatasetTestBase):
  def register_dataset(self, dispatcher_address, dataset):
    compression = "AUTO"
    if TRANSFER_PROTOCOL.value is not None:
      compression = None
    return data_service_ops.register_dataset(
        dispatcher_address, dataset, compression=compression)
  def from_dataset_id(self,
                      processing_mode,
                      cluster,
                      dataset_id,
                      element_spec,
                      job_name=None):
    return data_service_ops.from_dataset_id(
        processing_mode,
        cluster.dispatcher_address(),
        dataset_id,
        element_spec,
        data_transfer_protocol=TRANSFER_PROTOCOL.value,
        job_name=job_name)
  def make_distributed_dataset(self,
                               dataset,
                               cluster,
                               processing_mode="parallel_epochs",
                               job_name=None,
                               consumer_index=None,
                               num_consumers=None,
                               max_outstanding_requests=None,
                               compression="AUTO",
                               target_workers="AUTO"):
    return dataset.apply(
        data_service_ops._distribute(
            processing_mode,
            cluster.dispatcher_address(),
            job_name=job_name,
            consumer_index=consumer_index,
            num_consumers=num_consumers,
            max_outstanding_requests=max_outstanding_requests,
            task_refresh_interval_hint_ms=20,
            data_transfer_protocol=TRANSFER_PROTOCOL.value,
            compression=compression,
            target_workers=target_workers))
  def make_distributed_range_dataset(self,
                                     num_elements,
                                     cluster,
                                     processing_mode="parallel_epochs",
                                     job_name=None,
                                     max_outstanding_requests=None,
                                     compression="AUTO",
                                     target_workers="AUTO"):
    dataset = dataset_ops.Dataset.range(num_elements)
    return self.make_distributed_dataset(
        dataset,
        cluster,
        processing_mode=processing_mode,
        job_name=job_name,
        max_outstanding_requests=max_outstanding_requests,
        compression=compression,
        target_workers=target_workers)
  def make_coordinated_read_dataset(
      self,
      cluster,
      num_consumers,
      sharding_policy=data_service_ops.ShardingPolicy.OFF):
    if sharding_policy not in [
        data_service_ops.ShardingPolicy.OFF,
        data_service_ops.ShardingPolicy.DYNAMIC
    ]:
      raise ValueError(f"Unsupported sharding policy: {sharding_policy}")
    ds = dataset_ops.Dataset.from_tensors(math_ops.cast(0, dtypes.int64))
    ds = ds.concatenate(dataset_ops.Dataset.random())
    def make_group(x):
      x = x % (2**32)
      return dataset_ops.Dataset.range(x*num_consumers, (x+1)*num_consumers)
    ds = ds.flat_map(make_group)
    consumers = []
    for consumer_index in range(num_consumers):
      consumers.append(
          self.make_distributed_dataset(
              ds,
              cluster,
              job_name="test",
              processing_mode=sharding_policy,
              consumer_index=consumer_index,
              num_consumers=num_consumers))
    ds = dataset_ops.Dataset.from_tensor_slices(consumers)
    ds = ds.interleave(
        lambda x: x,
        cycle_length=num_consumers,
        num_parallel_calls=num_consumers)
    return ds
  def checkCoordinatedReadGroups(self, results, num_consumers):
    groups = [
        results[start:start + num_consumers]
        for start in range(0, len(results), num_consumers)
    ]
    incorrect_groups = []
    for group in groups:
      for offset in range(1, len(group)):
        if group[0] + offset != group[offset]:
          incorrect_groups.append(group)
          break
    self.assertEmpty(
        incorrect_groups,
        "Incorrect groups: {}.\nAll groups: {}".format(incorrect_groups,
                                                       groups))
  def read(self, get_next, results, count):
    for _ in range(count):
      results.append(self.evaluate(get_next()))
