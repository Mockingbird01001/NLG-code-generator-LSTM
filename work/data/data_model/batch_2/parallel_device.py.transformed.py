
import threading
import weakref
from tensorflow.python import _pywrap_parallel_device
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
_next_device_number = 0
_next_device_number_lock = threading.Lock()
_all_parallel_devices = weakref.WeakValueDictionary()
def unpack(tensor):
  parallel_device = _all_parallel_devices.get(tensor.device, None)
  if parallel_device is None:
    raise ValueError("{} is not a parallel device".format(tensor.device))
  return parallel_device.unpack(tensor)
class ParallelDevice(object):
  def __init__(self, components):
    global _next_device_number, _next_device_number_lock
    self.components = tuple(device_util.canonicalize(d) for d in components)
    if not self.components:
      raise ValueError("ParallelDevice requires at least one component.")
    ctx = context.context()
    with _next_device_number_lock:
      self._name = "{}/device:CUSTOM:{}".format(ctx.host_address_space(),
                                                _next_device_number)
      _next_device_number += 1
    device, device_info = _pywrap_parallel_device.GetParallelDeviceCapsules(
        self._name, self.components)
    context.register_custom_device(device, self._name, device_info)
    self._device_ids = None
    self._device_scope = None
    _all_parallel_devices[self._name] = self
  def _pack_tensor(self, *tensors):
    for tensor in tensors:
      if not isinstance(tensor, (ops.Tensor, composite_tensor.CompositeTensor,
                                 variables.Variable)):
        raise ValueError(
            ("Every component to pack onto the ParallelDevice must already be "
             "a tensor, got {}. Consider running `tf.constant` or "
             "`tf.convert_to_tensor` first on literal values.")
            .format(tensors))
    with ops.device(None):
      tensors = [t.read_value() if isinstance(t, variables.Variable)
                 else t for t in tensors]
    with ops.device(self._name):
      return tpu_ops.tpu_replicated_input(inputs=tensors)
  def pack(self, tensors):
    """Create a tensor on the parallel device from a sequence of tensors.
    Args:
      tensors: A list of tensors, one per device in `self.components`. The list
        can contain composite tensors and nests (lists, dicts, etc. supported by
        `tf.nest`) with the same structure for each device, but every component
        of nests must already be a `tf.Tensor` or composite. Passing
        `tf.Variable` objects reads their value, it does not share a mutable
        reference between the packed and unpacked forms.
    Returns:
      A tensor placed on the ParallelDevice. For nested structures, returns a
      single structure containing tensors placed on the ParallelDevice (same
      structure as each component of `tensors`).
    Raises:
      ValueError: If the length of `tensors` does not match the number of
        component devices, or if there are non-tensor inputs.
    """
    self._assert_eager()
    if len(tensors) != len(self.components):
      raise ValueError(
          ("Creating a parallel tensor requires one tensor per component. "
           "Got {} but was expecting {}.")
          .format(len(tensors), len(self.components)))
    return nest.map_structure(self._pack_tensor, *tensors,
                              expand_composites=True)
  def _unpack_tensor(self, parallel_tensor):
    if not isinstance(parallel_tensor, (
        ops.Tensor, composite_tensor.CompositeTensor, variables.Variable)):
      raise ValueError(
          "Expected a tensor, got {}.".format(parallel_tensor))
    with ops.device(self._name):
      return tpu_ops.tpu_replicated_output(
          parallel_tensor, num_replicas=len(self.components))
  def unpack(self, parallel_tensor):
    self._assert_eager()
    unpacked_components = [[] for _ in range(len(self.components))]
    for tensor in nest.flatten(parallel_tensor, expand_composites=True):
      for accumulator, unpacked_tensor in zip(
          unpacked_components, self._unpack_tensor(tensor)):
        accumulator.append(unpacked_tensor)
    return [nest.pack_sequence_as(parallel_tensor, unpacked,
                                  expand_composites=True)
            for unpacked in unpacked_components]
  @property
  def device_ids(self):
    if self._device_ids is None:
      with ops.init_scope():
        device_ids_list = []
        for index, device in enumerate(self.components):
          with ops.device(device):
            device_ids_list.append(
                array_ops.identity(constant_op.constant(index)))
        self._device_ids = self.pack(device_ids_list)
    return self._device_ids
  def _assert_eager(self):
    if not context.executing_eagerly():
      raise NotImplementedError(
          "ParallelDevice is currently not supported inside `tf.function`. It "
          "can however run calls to a `tf.function` in parallel:\n\n"
          "with ParallelDevice() as p:\n  f()")
  def __enter__(self):
    if self._device_scope is not None:
      raise AssertionError(
          "Re-entered a ParallelDevice scope without first exiting it.")
    self._assert_eager()
    self._device_scope = ops.device(self._name)
    self._device_scope.__enter__()
    return self
  def __exit__(self, typ, exc, tb):
    self._device_scope.__exit__(typ, exc, tb)
    self._device_scope = None
