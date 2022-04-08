
import contextlib
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import _pywrap_tf_cluster as tf_cluster
class Cluster(object):
  def __init__(self,
               allow_soft_placement=True,
               disable_detailed_stats=True,
               disable_timeline=True,
               devices=None):
    self._tf_cluster = None
    self._generate_timeline = not disable_timeline
    if devices is None:
      self._tf_cluster = tf_cluster.TF_NewCluster(allow_soft_placement,
                                                  disable_detailed_stats)
    else:
      devices_serialized = [device.SerializeToString() for device in devices]
      self._tf_cluster = tf_cluster.TF_NewVirtualCluster(devices_serialized)
  def Shutdown(self):
    if self._tf_cluster is not None:
      tf_cluster.TF_ShutdownCluster(self._tf_cluster)
      self._tf_cluster = None
  def __del__(self):
    self.Shutdown()
  @property
  def tf_cluster(self):
    return self._tf_cluster
  def ListDevices(self):
    if self._tf_cluster is None:
      return []
    return [device_properties_pb2.NamedDevice.FromString(device)
            for device in tf_cluster.TF_ListDevices(self._tf_cluster)]
  def ListAvailableOps(self):
    return tf_cluster.TF_ListAvailableOps()
  def GetSupportedDevices(self, item):
    return tf_cluster.TF_GetSupportedDevices(self._tf_cluster, item.tf_item)
  def EstimatePerformance(self, device):
    return tf_cluster.TF_EstimatePerformance(device.SerializeToString())
  def MeasureCosts(self, item):
    op_perf_bytes_list, run_time, step_stats_bytes = tf_cluster.TF_MeasureCosts(
        item.tf_item, self._tf_cluster, self._generate_timeline)
    op_perfs = [op_performance_data_pb2.OpPerformance.FromString(op_perf_bytes)
                for op_perf_bytes in op_perf_bytes_list]
    return (op_perfs, run_time,
            step_stats_pb2.StepStats.FromString(step_stats_bytes))
  def DeterminePeakMemoryUsage(self, item):
    return tf_cluster.TF_DeterminePeakMemoryUsage(item.tf_item,
                                                  self._tf_cluster)
@contextlib.contextmanager
def Provision(allow_soft_placement=True,
              disable_detailed_stats=True,
              disable_timeline=True,
              devices=None):
  cluster = Cluster(allow_soft_placement, disable_detailed_stats,
                    disable_timeline, devices)
  yield cluster
  cluster.Shutdown()
