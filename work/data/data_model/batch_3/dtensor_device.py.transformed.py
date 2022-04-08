
import contextlib
import logging
import os
import threading
from typing import List, Set
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import resource_variable_ops
_DT_CLIENT_ID = "DTENSOR_CLIENT_ID"
_DT_NUM_CLIENTS = "DTENSOR_NUM_CLIENTS"
_DT_JOB_NAME = "DTENSOR_JOB_NAME"
_next_device_number = 0
_next_device_number_lock = threading.Lock()
class DTensorDevice(object):
  def __init__(self, meshes: List[layout_lib.Mesh], is_async=True):
    """Create a new DTensorDevice which executes ops on `underlying_device`.
    Args:
      meshes: A list of `Mesh` objects indicating groups of devices to execute
        on. These may also be registered lazily.
      is_async: Indicates whether DTensor operations on this client will return
        immediately (with "non-ready" handles) or block until executed. This is
        on by default and is exposed as an option for ease of debugging.
    """
    if any(not isinstance(mesh, layout_lib.Mesh) for mesh in meshes):
      raise TypeError(
          "Expected a flat list of Mesh objects, got {}".format(meshes))
    global _next_device_number, _next_device_number_lock
    ctx = context.context()
    with _next_device_number_lock:
      self.name = "{}/device:CUSTOM:{}".format(ctx.host_address_space(),
                                               _next_device_number)
      _next_device_number += 1
    device, device_info = _pywrap_dtensor_device.Allocate(self.name)
    context.register_custom_device(device, self.name, device_info)
    self._device_info = device_info
    self._current_output_layout = None
    self._is_async = is_async
    self._meshes = set()
    self._mesh_lock = threading.Lock()
    for mesh in meshes:
      self._register_mesh(mesh)
  def _num_clients(self):
    return int(os.environ.get(_DT_NUM_CLIENTS, "1"))
  def _client_id(self):
    return int(os.environ.get(_DT_CLIENT_ID, "0"))
  def _job_name(self):
    return os.environ.get(_DT_JOB_NAME, "localhost")
  def _full_job_name(self):
    if self._job_name() == "localhost":
      return "localhost/replica:0/task:0"
    return self._job_name() + "/replica:0/task:" + str(self._client_id())
  def _create_host_array(self, shape, host_id):
    num_global_devices = np.prod(shape)
    global_device_ids = np.arange(num_global_devices).reshape(shape)
    local_device_list = [
        tf_device.DeviceSpec(
            job=self._full_job_name(), device_type="CPU", device_index=0)
    ]
    num_local_devices = len(local_device_list)
    local_device_ids = [
        x + host_id * num_local_devices for x in range(num_local_devices)
    ]
    return global_device_ids, local_device_ids, local_device_list
  def _create_embedding_host_mesh(self, tpu_mesh: layout_lib.Mesh):
    if tpu_mesh.device_type().upper() != "TPU":
      raise ValueError("Must pass input of a tpu mesh.")
    ts_local_device_ids = []
    ts_local_devices = []
    for local_device_str in tpu_mesh.local_devices():
      if not local_device_str.endswith("TPU:0"):
        continue
      device_spec = tf_device.DeviceSpec.from_string(local_device_str)
      ts_local_device_ids.append(device_spec.task)
      ts_local_devices.append(device_spec.replace(device_type="CPU"))
    if not ts_local_device_ids or not ts_local_device_ids:
      logging.info(
          "Cannot create tpu system mesh as %s has no `TPU:0` local device "
          "found", tpu_mesh.to_string())
      return None
    ts_global_device_ids = np.arange(self._num_clients())
    return layout_lib.Mesh(
        global_device_ids=ts_global_device_ids,
        local_device_ids=ts_local_device_ids,
        local_devices=ts_local_devices)
  def _register_mesh(self, mesh: layout_lib.Mesh):
    with self._mesh_lock:
      if mesh not in self._meshes:
        _pywrap_dtensor_device.AddMesh(self._device_info, mesh.to_string(),
                                       self._is_async, False)
        self._meshes.add(mesh)
        if mesh.device_type().upper() == "TPU":
          logging.info(
              "Registering virtual 1:1 mapped host mesh %s for mesh %s",
              mesh.host_mesh().to_string(), mesh.to_string())
          _pywrap_dtensor_device.AddMesh(self._device_info,
                                         mesh.host_mesh().to_string(),
                                         self._is_async, True)
          self._meshes.add(mesh.host_mesh())
          embedding_host_mesh = self._create_embedding_host_mesh(mesh)
          if embedding_host_mesh:
            logging.info(
                "Registering embedding host mesh %s on each client for mesh %s ",
                embedding_host_mesh.to_string(), mesh.to_string())
            _pywrap_dtensor_device.AddMesh(self._device_info,
                                           embedding_host_mesh.to_string(),
                                           self._is_async, False)
            self._meshes.add(embedding_host_mesh)
  @property
  def meshes(self) -> Set[layout_lib.Mesh]:
    return self._meshes
  def copy_to_mesh(self, tensor, new_layout, source_layout=None) -> ops.Tensor:
    self._register_mesh(new_layout.mesh)
    with ops.device(self.name):
      return gen_dtensor_ops.copy_to_mesh(
          tensor,
          layout=new_layout.to_string(),
          source_layout=source_layout.to_string() if source_layout else "")
  def pack(self, tensors, layout):
    """Returns a packed tensor handle for dtensor_device.
    Packing and unpacking are inverse operations:
    * unpack(pack(tensors)) == tensors
    * pack(unpack(dtensor)) == dtensor
    1. For any DTensor on the mesh, `unpack` returns the raw components placed
       on each underlying device.
    2. Packing these raw components in the same order returns a DTensor which
       should be identical to the original DTensor--both the content value and
       the layout.
    N.B. It is the callers responsibility to ensure that the underlying values
    for Pack() adhere to the specified layout. See examples below for more
    detail.
    N.B. [Shape, Rank, and Scalar]: The rank of the DTensor is the same as the
    rank of it's raw components, i.e., rank is preserved.  This leads to a
    consistent interpretation for packing scalar values into a DTensor. The only
    valid layout for a scalar value is fully replicated, and the individual
    components must be identical scalars.
    N.B. Each input `tensor[i]` will be copied to `layout.mesh.local_device[i]`
    if not already on the local device.
    For example, assume we have a mesh [X(2), Y(3)], which has in total 6
    underlying devices. Futuremore, assume that the device location mapping is
    the following:
       device_ID  |  location X, Y
               0     0, 0
               1     0, 1
               2     0, 2
               3     1, 0
               4     1, 1
               5     1, 2
    1. For 1-D vector DTensor with shape [128] with layout [mesh.X] and value as
       range(128), the raw components will have shape [64] each, and the raw
       components will be:
       device_ID  |  raw component
               0     range(64)
               1     range(64)
               2     range(64)
               3     range(64, 128)
               4     range(64, 128)
               5     range(64, 128)
       This also means for a 1-D DTensor with shape [2] and layout [mesh.X], the
       raw components have shape [1] rather than the shape for scalar value.
    2. For 2-D vector DTensor with shape [2, 3] with layout [mesh.X, mesh.Y] and
       value as range(6), this is basically a fully-sharded DTensor.
       From global view, the content looks like
           [
             [0.0, 1.0, 2.0],
             [3.0, 4.0, 5.0],
           ]
       The raw components will have shape [1, 1] each, and have the following
       content:
       device_ID  |  raw component
               0     [[0.0]]
               1     [[1.0]]
               2     [[2.0]]
               3     [[3.0]]
               4     [[4.0]]
               5     [[5.0]]
    3. For a scalar value 123.0 DTensor, it can only have one legitimate layout
       `[]` (no dimension, but fully replicated).
       The raw components will have shape [] each, and have the following
       content:
       device_ID  |  raw component
               0     123.0
               1     123.0
               2     123.0
               3     123.0
               4     123.0
               5     123.0
       Again, caller of `pack` is expected to provide 6 identical value raw
       components with scalar shapes.
    4. For 3-D vector DTensor with shape [2, 2, 3] with layout
       [X, unsharded, unsharded] and value as range(12),
       From global view, the content looks like
           [
             [
               [0.0, 1.0, 2.0],
               [3.0, 4.0, 5.0],
             ],
             [
               [6.0, 7.0, 8.0],
               [9.0, 10., 11.],
             ],
           ]
       The raw components will have shape [1, 2, 3] each, and have the following
       content:
       device_ID  |  raw component
               0     range(6).reshape([1, 2, 3])
               1     range(6).reshape([1, 2, 3])
               2     range(6).reshape([1, 2, 3])
               3     range(6, 12).reshape([1, 2, 3])
               4     range(6, 12).reshape([1, 2, 3])
               5     range(6, 12).reshape([1, 2, 3])
    Raises:
      RuntimeError: When not called eagerly.
    """
    if not context.executing_eagerly():
      raise RuntimeError("Pack must be called eagerly.")
    if any(
        issubclass(type(t), resource_variable_ops.BaseResourceVariable)
        for t in tensors):
      raise TypeError(
          "Received Variable input to Pack, Variable is not supported.")
    self._register_mesh(layout.mesh)
    with ops.device(self.name):
      if all(isinstance(t, sparse_tensor.SparseTensor) for t in tensors):
        if not all(t.shape == tensors[0].shape for t in tensors):
          raise TypeError("All input SparseTensors to Pack must be same shape.")
        is_sparse = True
        tensors = [t.indices for t in tensors] + [t.values for t in tensors] + [
            ops.convert_to_tensor(t.shape, dtype=dtypes.int64) for t in tensors
        ]
      elif any(isinstance(t, sparse_tensor.SparseTensor) for t in tensors):
        raise TypeError("Cannot Pack SparseTensors with Tensors.")
      else:
        is_sparse = False
      try:
        return _pywrap_dtensor_device.Pack(
            tensors,
            layout.to_string(),
            self._device_info,
            is_sparse)
  def unpack(self, tensor):
    """Returns the raw tensor components of the packed tensor.
    Packing and unpacking are inverse operations:
    * unpack(pack(tensors)) == tensors
    * pack(unpack(dtensor)) == dtensor
    See documentation for pack for more details.
    Raises:
      RuntimeError: When not called eagerly.
    """
    if not context.executing_eagerly():
      raise RuntimeError("Unpack must be called eagerly.")
    if issubclass(type(tensor), resource_variable_ops.BaseResourceVariable):
      raise TypeError(
          "Received Variable input to unpack, Variable is not supported.")
    try:
      tensors = _pywrap_dtensor_device.Unpack(
          tensor,
          self._device_info)
    is_sparse = _pywrap_dtensor_device.IsSparseDTensor(
        tensor,
        self._device_info)
    if is_sparse:
      result = []
      for i in range(len(tensors) // 3):
        result.append(
            sparse_tensor.SparseTensor(tensors[i],
                                       tensors[i + len(tensors) // 3],
                                       tensors[i + 2 * len(tensors) // 3]))
      return result
    else:
      return tensors
  def fetch_layout(self, tensor):
    if not context.executing_eagerly():
      raise RuntimeError("FetchLayout must be called eagerly.")
    if issubclass(type(tensor), resource_variable_ops.BaseResourceVariable):
      tensor = tensor.read_value()
    try:
      layout_string = _pywrap_dtensor_device.FetchLayout(
          tensor,
          self._device_info)
    return layout_lib.Layout.from_string(layout_string)
  def set_same_shape_policy(self, enabled):
    _pywrap_dtensor_device.SetSameShapePolicy(self._device_info, enabled)
  def set_tpu_core_ids(self, mesh_name, tpu_core_ids):
    _pywrap_dtensor_device.SetTPUCoreIDs(self._device_info, mesh_name,
                                         tpu_core_ids)
  def clear_tpu_core_ids(self):
    _pywrap_dtensor_device.ClearTPUCoreIDs(self._device_info)
  def tpu_core_ids_to_locations(self, tpu_core_ids):
    return _pywrap_dtensor_device.TPUCoreIDsToLocations(
        self._device_info,
        tpu_core_ids)
  def tpu_core_locations_to_ids(self, tpu_core_locations):
    return _pywrap_dtensor_device.TPUCoreLocationsToIDs(
        self._device_info,
        tpu_core_locations)
  @contextlib.contextmanager
  def _experimental_default_mesh(self, mesh: layout_lib.Mesh):
    self._register_mesh(mesh)
    _pywrap_dtensor_device.ExperimentalSetDefaultMesh(
        self._device_info,
        mesh.to_string().encode("utf-8"))
    yield
    _pywrap_dtensor_device.ExperimentalClearDefaultMesh(self._device_info)
  @contextlib.contextmanager
  def _default_layout(self, layout: layout_lib.Layout):
    self._register_mesh(layout.mesh)
    try:
      previous_default = self._current_output_layout
      self._current_output_layout = layout.to_string().encode("utf-8")
      _pywrap_dtensor_device.ExperimentalSetDefaultLayout(
          self._device_info, self._current_output_layout)
      if context.executing_eagerly():
        graph = None
        previous_graph_size = None
        with ops.device(self.name):
          yield
      else:
        graph = ops.get_default_graph()
        previous_graph_size = len(graph.get_operations())
        yield
    finally:
              "_layout",
              attr_value_pb2.AttrValue(
                  list=attr_value_pb2.AttrValue.ListValue(
                      s=[self._current_output_layout])))
              "_mesh",
              attr_value_pb2.AttrValue(
                  s=layout.mesh.to_string().encode("utf-8")))
      if self._current_output_layout is None:
        _pywrap_dtensor_device.ExperimentalClearDefaultLayout(self._device_info)
      else:
        _pywrap_dtensor_device.ExperimentalSetDefaultLayout(
            self._device_info, self._current_output_layout.decode("utf-8"))
