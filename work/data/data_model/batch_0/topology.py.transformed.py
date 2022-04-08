
import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
def _tpu_device_name(job, task, device):
  if job is None:
    return "/task:%d/device:TPU:%d" % (task, device)
  else:
    return "/job:%s/task:%d/device:TPU:%d" % (job, task, device)
def _tpu_host_device_name(job, task):
  if job is None:
    return "/task:%d/device:CPU:0" % task
  else:
    return "/job:%s/task:%d/device:CPU:0" % (job, task)
@tf_export("tpu.experimental.Topology")
class Topology(object):
  def __init__(self, serialized=None, mesh_shape=None, device_coordinates=None):
    self._serialized = serialized
    if serialized:
      self._parse_topology(serialized)
    else:
      self._mesh_shape = np.asarray(mesh_shape, dtype=np.int32)
      self._device_coordinates = np.asarray(device_coordinates, np.int32)
      if len(self._mesh_shape) != 4 or any(self._mesh_shape < 1):
        raise ValueError("`mesh_shape` must be a sequence of 4 positive "
                         f"entries; got `mesh_shape={self._mesh_shape}`")
      if (len(self._device_coordinates.shape) != 3 or
          self._device_coordinates.shape[2] != len(self._mesh_shape)):
        raise ValueError(
            "`device_coordinates` must be a rank 3 int32 array "
            "with minor dimension equal to the `mesh_shape` rank"
            "got device_coordinates={} len(device_coordinates)={} device_coordinates.shape[2]={} mesh_shape={}, len(mesh_shape)={}"
            .format(self._device_coordinates.shape,
                    len(self._device_coordinates.shape),
                    self._device_coordinates.shape[2], self._mesh_shape,
                    len(self._mesh_shape)))
    self._topology_tasks, self._topology_devices = self._invert_topology()
    self._missing_devices = np.argwhere(self._topology_tasks < 0)
  def _parse_topology(self, serialized):
    proto = topology_pb2.TopologyProto()
    proto.ParseFromString(serialized)
    self._mesh_shape = np.array(proto.mesh_shape, dtype=np.int32)
    if len(self._mesh_shape) != 4 or any(self._mesh_shape < 1):
      raise ValueError("`mesh_shape` must be a vector of size 4 with positive "
                       "entries; got {}".format(self._mesh_shape))
    if proto.num_tasks < 0:
      raise ValueError("`num_tasks` must be >= 0; got {}".format(
          proto.num_tasks))
    if proto.num_tpu_devices_per_task < 0:
      raise ValueError("`num_tpu_devices_per_task` must be >= 0; got {}".format(
          proto.num_tpu_devices_per_task))
    expected_coordinates_size = (
        proto.num_tasks * proto.num_tpu_devices_per_task * len(
            proto.mesh_shape))
    if len(proto.device_coordinates) != expected_coordinates_size:
      raise ValueError("`device_coordinates` must have shape num_tasks ({}) * "
                       "num_tpu_devices_per_task ({}) * len(mesh_shape) ({}); "
                       "got shape {}".format(proto.num_tasks,
                                             proto.num_tpu_devices_per_task,
                                             proto.mesh_shape,
                                             len(proto.device_coordinates)))
    coords = np.array(proto.device_coordinates, dtype=np.int32)
    if any(coords < 0):
      raise ValueError(
          "All values in `device_coordinates` must be >= 0, got {}"
          .format(coords))
    coords = coords.reshape((proto.num_tasks, proto.num_tpu_devices_per_task,
                             len(proto.mesh_shape)))
    self._device_coordinates = coords
  def _invert_topology(self):
    tasks = np.full(list(self.mesh_shape), -1, dtype=np.int32)
    devices = np.full(list(self.mesh_shape), -1, dtype=np.int32)
    for task in range(self.device_coordinates.shape[0]):
      for device in range(self.device_coordinates.shape[1]):
        x, y, z, core = self.device_coordinates[task, device, :]
        tasks[x, y, z, core] = task
        devices[x, y, z, core] = device
    return tasks, devices
  @property
  def mesh_shape(self):
    return self._mesh_shape
  @property
  def mesh_rank(self):
    return len(self._mesh_shape)
  @property
  def device_coordinates(self):
    """Describes the mapping from TPU devices to topology coordinates.
    Returns:
      A rank 3 int32 array with shape `[tasks, devices, axis]`.
      `tasks` is the number of tasks in the TPU cluster, `devices` is the number
      of TPU devices per task, and `axis` is the number of axes in the TPU
      cluster topology. Each entry gives the `axis`-th coordinate in the
      topology of a task/device pair. TPU topologies are 4-dimensional, with
      dimensions `(x, y, z, core number)`.
    """
    return self._device_coordinates
  @property
  def missing_devices(self):
    return self._missing_devices
  def task_ordinal_at_coordinates(self, device_coordinates):
    return self._topology_tasks[tuple(device_coordinates)]
  def tpu_device_ordinal_at_coordinates(self, device_coordinates):
    return self._topology_devices[tuple(device_coordinates)]
  def cpu_device_name_at_coordinates(self, device_coordinates, job=None):
    return _tpu_host_device_name(
        job, self._topology_tasks[tuple(device_coordinates)])
  def tpu_device_name_at_coordinates(self, device_coordinates, job=None):
    return _tpu_device_name(job,
                            self._topology_tasks[tuple(device_coordinates)],
                            self._topology_devices[tuple(device_coordinates)])
  @property
  def num_tasks(self):
    return self._device_coordinates.shape[0]
  @property
  def num_tpus_per_task(self):
    return self._device_coordinates.shape[1]
  def serialized(self):
    if self._serialized is None:
      proto = topology_pb2.TopologyProto()
      proto.mesh_shape[:] = list(self._mesh_shape)
      proto.num_tasks = self._device_coordinates.shape[0]
      proto.num_tpu_devices_per_task = self._device_coordinates.shape[1]
      proto.device_coordinates.extend(list(self._device_coordinates.flatten()))
      self._serialized = proto.SerializeToString()
    return self._serialized
