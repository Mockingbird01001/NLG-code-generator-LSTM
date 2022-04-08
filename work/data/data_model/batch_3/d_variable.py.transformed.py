
import functools
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util.tf_export import tf_export
class DSaveSpec(saveable_object.SaveSpec):
  def __init__(self,
               tensor,
               slice_spec,
               name,
               global_shape,
               layout,
               dtype=None,
               device=None):
    super().__init__(
        tensor=tensor,
        slice_spec=slice_spec,
        name=name,
        dtype=dtype,
        device=device)
    self.global_shape = global_shape
    self.layout = layout
class _DVariableSaveable(saveable_object.SaveableObject):
  def __init__(self, dvariable, name):
    with ops.device(dvariable.device):
      original_layout = api.fetch_layout(dvariable)
    self._original_layout = original_layout
    self._dvariable = dvariable
    def pack(tensors, layout):
      with ops.device(dvariable.device):
        return api.pack(tensors, layout)
    host_layout = layout_lib.Layout(original_layout.sharding_specs,
                                    original_layout.mesh.host_mesh())
    def get_host_dvariable():
      if original_layout.mesh.device_type().upper() != 'CPU':
        with ops.device(dvariable.device):
          host_dvariable = DVariable(
              api.pack(api.unpack(dvariable.read_value()), host_layout))
      else:
        host_dvariable = dvariable
      return (math_ops.cast(host_dvariable, dtypes.bfloat16)
              if self.should_cast(host_dvariable) else host_dvariable)
    num_local_devices = original_layout.mesh.num_local_devices()
    super(_DVariableSaveable, self).__init__(
        None,
        [
            DSaveSpec(
                tensor=get_host_dvariable,
                slice_spec=pack([''] * num_local_devices,
                                layout_lib.Layout.replicated(
                                    original_layout.mesh.host_mesh(), rank=0)),
                name=pack([name] * num_local_devices,
                          layout_lib.Layout.replicated(
                              original_layout.mesh.host_mesh(), rank=0)),
                global_shape=dvariable.shape,
                layout=host_layout.to_string(),
                dtype=dtypes.bfloat16
                if self.should_cast(dvariable) else dvariable.dtype,
                device=dvariable.device)
        ],
        name)
  def should_cast(self, v):
    return self._dvariable.save_as_bf16 and v.dtype == dtypes.float32
  def restore(self, restored_tensors, restored_shapes):
    tensor, = restored_tensors
    @def_function.function
    def _restore(t):
      with ops.device(self._dvariable.device):
        return api.copy_to_mesh(t, self._original_layout)
    if self._original_layout.mesh.device_type().upper() != 'CPU':
      tensor = _restore(tensor)
    return self._dvariable.assign(
        math_ops.cast(tensor, dtype=self._dvariable.dtype) if self._dvariable
        .save_as_bf16 else tensor)
@tf_export('experimental.dtensor.DVariable', v1=[])
class DVariable(resource_variable_ops.ResourceVariable):
  def __init__(self, initial_value, *args, dtype=None, **kwargs):
    if callable(initial_value):
      initial_value = initial_value()
    initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
    variable_device = initial_value.device
    self._save_as_bf16 = False
    with ops.device(variable_device):
      super(DVariable, self).__init__(
          initial_value, *args, dtype=dtype, **kwargs)
      self.layout = None
      if context.executing_eagerly():
        try:
          self.layout = api.fetch_layout(initial_value)
        except (errors.InvalidArgumentError, errors.NotFoundError):
          self.layout = None
          pass
  @property
  def save_as_bf16(self):
    return self._save_as_bf16
  @save_as_bf16.setter
  def save_as_bf16(self, save_as_bf16):
    self._save_as_bf16 = save_as_bf16 and self.dtype == dtypes.float32
  def _gather_saveables_for_checkpoint(self):
    return {
        trackable.VARIABLE_VALUE_KEY:
            functools.partial(_DVariableSaveable, self)
    }
