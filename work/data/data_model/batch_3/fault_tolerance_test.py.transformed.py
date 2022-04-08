
import threading
import time
from absl.testing import parameterized
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
TMP_WORK_DIR = data_service_test_base.TMP_WORK_DIR
NO_WORK_DIR = data_service_test_base.NO_WORK_DIR
class FaultToleranceTest(data_service_test_base.TestBase,
                         parameterized.TestCase):
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherStop(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    iterator = iter(ds)
    results = []
    results.append(next(iterator).numpy())
    cluster.stop_dispatcher()
    for _ in range(num_elements - 1):
      results.append(next(iterator).numpy())
    self.assertEqual(results, list(range(num_elements)))
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherRestartBeforeReading(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    cluster.restart_dispatcher()
    self.assertDatasetProduces(ds, list(range(num_elements)))
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherRestartDuringReading(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    iterator = iter(ds)
    results = []
    for _ in range(num_elements // 2):
      results.append(next(iterator).numpy())
    cluster.restart_dispatcher()
    for elem in iterator:
      results.append(elem.numpy())
    self.assertEqual(list(range(num_elements)), results)
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherRestartDuringDistributedEpoch(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, processing_mode="distributed_epoch")
    iterator = iter(ds)
    results = []
    for _ in range(num_elements // 2):
      results.append(next(iterator).numpy())
    cluster.restart_dispatcher()
    for elem in iterator:
      results.append(elem.numpy())
    self.assertEqual(list(range(num_elements)), results)
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherRestartDuringDistributedEpochRepeat(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    repetitions = 5
    breakpoints = [50, 250, 450, 500]
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.repeat(repetitions)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    iterator = iter(ds)
    results = []
    for breakpoint_ in breakpoints:
      for _ in range(len(results), breakpoint_):
        results.append(next(iterator).numpy())
      cluster.restart_dispatcher()
    self.assertCountEqual(repetitions * list(range(num_elements)), results)
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherRestartBetweenIterations(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds = self.make_distributed_range_dataset(100, cluster)
    self.assertDatasetProduces(ds, list(range(num_elements)))
    cluster.restart_dispatcher()
    self.assertDatasetProduces(ds, list(range(num_elements)))
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherManyRestarts(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements_start = 10
    num_elements_end = 15
    datasets = []
    for num_elements in range(num_elements_start, num_elements_end):
      datasets.append(
          self.make_distributed_range_dataset(num_elements, cluster))
      cluster.restart_dispatcher()
    for ds, num_elements in zip(datasets,
                                range(num_elements_start, num_elements_end)):
      self.assertDatasetProduces(ds, list(range(num_elements)))
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherAndWorkerRestart(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    cluster.restart_dispatcher()
    cluster.workers[0].restart()
    self.assertDatasetProduces(ds, list(range(num_elements)))
    cluster.restart_dispatcher()
    cluster.workers[0].restart()
    self.assertDatasetProduces(ds, list(range(num_elements)))
  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherAndMultiWorkerRestart(self):
    num_workers = 2
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    num_elements = 100
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    iterator = iter(ds)
    results = []
    cluster.restart_dispatcher()
    for worker_index in range(num_workers):
      cluster.workers[worker_index].restart()
    for elem in iterator:
      results.append(elem.numpy())
    self.assertCountEqual(num_workers * list(range(num_elements)), results)
    cluster.restart_dispatcher()
    for worker_index in range(num_workers):
      cluster.workers[worker_index].restart()
    for elem in iterator:
      results.append(elem.numpy())
    self.assertCountEqual(num_workers * list(range(num_elements)), results)
  @combinations.generate(test_base.eager_only_combinations())
  def testStartServersLate(self):
    try:
      dispatcher_port = portpicker.pick_unused_port()
    except:
      raise self.skipTest("Flakes in portpicker library do not represent "
                          "TensorFlow errors.")
    cluster = data_service_test_base.TestCluster(
        num_workers=1, dispatcher_port=dispatcher_port, start=False)
    def start_servers():
      time.sleep(0.5)
      cluster.start_dispatcher()
      cluster.start_workers()
    start_servers_thread = threading.Thread(target=start_servers, daemon=True)
    start_servers_thread.start()
    num_elements = 10
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    results = [elem.numpy() for elem in ds]
    self.assertEqual(list(range(num_elements)), results)
    start_servers_thread.join()
  @combinations.generate(test_base.eager_only_combinations())
  def testAddWorkerMidJob(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    iterator = iter(ds)
    results = []
    for _ in range(num_elements // 2):
      results.append(next(iterator).numpy())
    cluster.add_worker()
    while cluster.num_registered_workers() < 2:
    for elem in iterator:
      results.append(elem.numpy())
    self.assertCountEqual(2 * list(range(num_elements)), results)
  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(use_same_port=[True, False]),
                         data_service_test_base.all_cluster_configurations()))
  def testRestartWorker(self, use_same_port, work_dir, fault_tolerant_mode):
    cluster = data_service_test_base.TestCluster(
        num_workers=1,
        work_dir=work_dir,
        fault_tolerant_mode=fault_tolerant_mode)
    num_elements = 100
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    iterator = iter(ds)
    midpoint = num_elements // 2
    for i in range(midpoint):
      self.assertEqual(i, next(iterator).numpy())
    cluster.workers[0].restart(use_same_port=use_same_port)
    while True:
      val = next(iterator).numpy()
      if val == 0:
        break
    for i in range(1, num_elements // 2):
      val = next(iterator).numpy()
      self.assertEqual(i, val)
  @combinations.generate(test_base.eager_only_combinations())
  def testChangeProcessingModeAfterRestart(self):
    self.skipTest("b/170910141")
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    range_dataset = dataset_ops.Dataset.range(num_elements)
    ds = range_dataset.apply(
        data_service_ops.distribute(
            processing_mode="parallel_epochs",
            service=cluster.dispatcher_address(),
            job_name="test"))
    iterator = iter(ds)
    for i in range(num_elements // 2):
      self.assertEqual(i, next(iterator).numpy())
    cluster.restart_dispatcher()
    ds = range_dataset.apply(
        data_service_ops.distribute(
            processing_mode="distributed_epoch",
            service=cluster.dispatcher_address(),
            job_name="test"))
    with self.assertRaisesOpError("already an existing job with that name "
                                  "using processing mode <parallel_epochs>"):
      next(iter(ds)).numpy()
  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(work_dir=[TMP_WORK_DIR, NO_WORK_DIR])))
  def testDistributeLargeGraphThenRegisterWorker(self, work_dir):
    cluster = data_service_test_base.TestCluster(
        num_workers=0, work_dir=work_dir, fault_tolerant_mode=False)
    tensor = array_ops.ones((2, 1000, 1000), dtype=dtypes.float32)
    ds = dataset_ops.Dataset.from_tensors(tensor)
    ds = self.make_distributed_dataset(ds, cluster)
    it = iter(ds)
    cluster.add_worker()
    self.assertAllEqual(next(it), tensor)
if __name__ == "__main__":
  test.main()
