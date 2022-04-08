
import time
from absl.testing import parameterized
from tensorflow.python.data.experimental.kernel_tests.service import multi_process_cluster
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
_COLOCATED_WORKER_TAG = "COLOCATED"
class WorkerTagsTest(data_service_test_base.TestBase, parameterized.TestCase):
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testReadFromLocalWorker(self, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(dataset, list(range(num_elements)))
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testReadFromLocalAndNonTpuWorkers(self, num_local_workers,
                                        num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_remote_worker(worker_tags=None)
    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(
        dataset, (num_local_workers + 1) * list(range(num_elements)),
        assert_items_equal=True)
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testLocalWorkerHasNoTag(self, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=0,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_local_worker(worker_tags=None)
    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(dataset, list(range(num_elements)))
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testReadFromLocalAndNonTpuWorkers_DynamicSharding(self, num_local_workers,
                                                        num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=3,
        worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_remote_worker(worker_tags=None)
    num_elements = 100
    dataset = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.DYNAMIC)
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=True)
  @combinations.generate(test_base.default_test_combinations())
  def testReadFromLocalWorker_StaticSharding(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1,
        num_remote_workers=3,
        worker_addresses=["localhost:%port%"] * 5,
        worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_remote_worker(worker_tags=None)
    num_elements = 100
    dataset = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, list(range(0, num_elements, 5)))
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[1, 3])))
  def testCoordinatedRead(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    num_consumers = 4
    dataset = self.make_coordinated_read_dataset(cluster, num_consumers)
    get_next = self.getNext(dataset)
    results = [self.evaluate(get_next()) for _ in range(200)]
    self.checkCoordinatedReadGroups(results, num_consumers)
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[1, 3])))
  def testAddRemoteWorkersMidJob(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    num_elements = 300
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    get_next = self.getNext(dataset)
    results = [self.evaluate(get_next()) for _ in range(100)]
    cluster.start_remote_worker(worker_tags=None)
    cluster.start_remote_worker(worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_remote_worker(worker_tags=None)
    cluster.start_remote_worker(worker_tags=[_COLOCATED_WORKER_TAG])
    expect_num_workers_to_read = num_local_workers + 2
    while cluster._dispatcher._num_workers() < (num_local_workers +
                                                num_remote_workers + 4):
    results += self.getIteratorOutput(get_next)
    self.assertCountEqual(
        results, expect_num_workers_to_read * list(range(num_elements)))
  @combinations.generate(test_base.default_test_combinations())
  def testMultipleTags(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1,
        num_remote_workers=3,
        worker_tags=[_COLOCATED_WORKER_TAG, "COLOCATED_2", "COLOCATED_3"])
    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(dataset, list(range(num_elements)))
  @combinations.generate(test_base.default_test_combinations())
  def testUnusedTags(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1,
        num_remote_workers=3,
        worker_tags=["Unused tag 1", "Unused tag 2", "Unused tag 3"])
    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(
        dataset, 4 * list(range(num_elements)), assert_items_equal=True)
  @combinations.generate(test_base.default_test_combinations())
  def testInvalidTag(self):
    with self.assertRaisesRegex(RuntimeError, "Worker tags cannot be empty."):
      _ = multi_process_cluster.MultiProcessCluster(
          num_local_workers=1,
          num_remote_workers=3,
          worker_tags=["", _COLOCATED_WORKER_TAG])
if __name__ == "__main__":
  multi_process_cluster.test_main()
