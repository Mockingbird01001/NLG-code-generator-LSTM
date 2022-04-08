
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import registration
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.training.saving import saveable_hook
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
class _SingleDeviceSaver(object):
  __slots__ = ["_saveable_objects"]
  def __init__(self, saveable_objects):
    saveable_objects = list(saveable_objects)
    for saveable in saveable_objects:
      if not isinstance(saveable, saveable_object.SaveableObject):
        raise ValueError(f"Expected a list of SaveableObjects, got {saveable}.")
    self._saveable_objects = saveable_objects
  def save(self, file_prefix, options=None):
    options = options or checkpoint_options.CheckpointOptions()
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in self._saveable_objects:
      for spec in saveable.specs:
        tensor = spec.tensor
        if tensor is not None:
          tensor_names.append(spec.name)
          tensors.append(tensor)
          tensor_slices.append(spec.slice_spec)
    save_device = options.experimental_io_device or "cpu:0"
    with ops.device(save_device):
      return io_ops.save_v2(file_prefix, tensor_names, tensor_slices, tensors)
  def restore(self, file_prefix, options=None):
    options = options or checkpoint_options.CheckpointOptions()
    restore_specs = []
    tensor_structure = []
    for saveable in self._saveable_objects:
      saveable_tensor_structure = []
      tensor_structure.append(saveable_tensor_structure)
      for spec in saveable.specs:
        saveable_tensor_structure.append(spec.name)
        restore_specs.append((spec.name, spec.slice_spec, spec.dtype))
    tensor_names, tensor_slices, tensor_dtypes = zip(*restore_specs)
    restore_device = options.experimental_io_device or "cpu:0"
    with ops.device(restore_device):
      restored_tensors = io_ops.restore_v2(
          file_prefix, tensor_names, tensor_slices, tensor_dtypes)
    structured_restored_tensors = nest.pack_sequence_as(
        tensor_structure, restored_tensors)
    restore_ops = {}
    for saveable, restored_tensors in zip(self._saveable_objects,
                                          structured_restored_tensors):
      restore_ops[saveable.name] = saveable.restore(
          restored_tensors, restored_shapes=None)
    return restore_ops
def sharded_filename(filename_tensor, shard, num_shards):
  return gen_io_ops.sharded_filename(filename_tensor, shard, num_shards)
def registered_saver_filename(filename_tensor, saver_name):
  return string_ops.string_join(
      [filename_tensor, constant_op.constant(f"-{saver_name}")])
def _get_mapped_registered_save_fn(fn, trackables, call_with_mapped_captures):
  def save_fn(file_prefix):
    return fn(trackables=trackables, file_prefix=file_prefix)
  if call_with_mapped_captures is None:
    return save_fn
  else:
    tf_fn = def_function.function(save_fn, autograph=False)
    concrete = tf_fn.get_concrete_function(
        file_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))
    def save_fn_with_replaced_captures(file_prefix):
      return call_with_mapped_captures(concrete, [file_prefix])
    return save_fn_with_replaced_captures
def _get_mapped_registered_restore_fn(fn, trackables,
                                      call_with_mapped_captures):
  def restore_fn(merged_prefix):
    return fn(trackables=trackables, merged_prefix=merged_prefix)
  if call_with_mapped_captures is None:
    return restore_fn
  else:
    tf_fn = def_function.function(restore_fn, autograph=False)
    concrete = tf_fn.get_concrete_function(
        merged_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))
    def restore_fn_with_replaced_captures(merged_prefix):
      return call_with_mapped_captures(concrete, [merged_prefix])
    return restore_fn_with_replaced_captures
class MultiDeviceSaver(object):
  def __init__(self,
               saveable_objects,
               registered_savers=None,
               call_with_mapped_captures=None):
    self._before_save_callbacks = []
    self._after_restore_callbacks = []
    saveable_objects = list(saveable_objects)
    saveables_by_device = {}
    for saveable in saveable_objects:
      is_saveable = isinstance(saveable, saveable_object.SaveableObject)
      is_hook = isinstance(saveable, saveable_hook.SaveableHook)
      if not is_saveable and not is_hook:
        raise ValueError(
            f"Expected a dictionary of SaveableObjects, got {saveable}.")
      if is_hook:
        self._before_save_callbacks.append(saveable.before_save)
        self._after_restore_callbacks.append(saveable.after_restore)
      if is_saveable:
        host_device = saveable_object_util.set_cpu0(saveable.device)
        saveables_by_device.setdefault(host_device, []).append(saveable)
    self._single_device_savers = {
        device: _SingleDeviceSaver(saveables)
        for device, saveables in saveables_by_device.items()}
    self._registered_savers = {}
    if registered_savers:
      for registered_name, trackables in registered_savers.items():
        save_fn = _get_mapped_registered_save_fn(
            registration.get_save_function(registered_name),
            trackables, call_with_mapped_captures)
        restore_fn = _get_mapped_registered_restore_fn(
            registration.get_restore_function(registered_name),
            trackables, call_with_mapped_captures)
        self._registered_savers[registered_name] = (save_fn, restore_fn)
  def to_proto(self):
    filename_tensor = array_ops.placeholder(
        shape=[], dtype=dtypes.string, name="saver_filename")
    save_tensor = self._traced_save(filename_tensor)
    restore_op = self._traced_restore(filename_tensor).op
    return saver_pb2.SaverDef(
        filename_tensor_name=filename_tensor.name,
        save_tensor_name=save_tensor.name,
        restore_op_name=restore_op.name,
        version=saver_pb2.SaverDef.V2)
  @def_function.function(
      input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),),
      autograph=False)
  def _traced_save(self, file_prefix):
    save_op = self.save(file_prefix)
    with ops.device("cpu:0"):
      with ops.control_dependencies([save_op]):
        return array_ops.identity(file_prefix)
  @def_function.function(
      input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),),
      autograph=False)
  def _traced_restore(self, file_prefix):
    restore_ops = self.restore(file_prefix)
    with ops.device("cpu:0"):
      with ops.control_dependencies(restore_ops.values()):
        return array_ops.identity(file_prefix)
  def save(self, file_prefix, options=None):
    options = options or checkpoint_options.CheckpointOptions()
    for callback in self._before_save_callbacks:
      callback()
    with ops.device("CPU"):
      sharded_suffix = array_ops.where(
          string_ops.regex_full_match(file_prefix, "^s3://.*"),
          constant_op.constant(".part"),
          constant_op.constant("_temp/part"))
      tmp_checkpoint_prefix = string_ops.string_join(
          [file_prefix, sharded_suffix])
      registered_paths = {
          saver_name: registered_saver_filename(file_prefix, saver_name)
          for saver_name in self._registered_savers
      }
    def save_fn():
      saved_prefixes = []
      for saver_name, (save_fn, _) in self._registered_savers.items():
        maybe_saved_prefixes = save_fn(registered_paths[saver_name])
        if maybe_saved_prefixes is not None:
          flattened_saved_prefixes = nest.flatten(maybe_saved_prefixes)
          if not all(
              tensor_util.is_tf_type(x) and x.dtype == dtypes.string
              for x in flattened_saved_prefixes):
            raise ValueError(
                "Registered saver can only return `None` or "
                f"string type tensors. Got {maybe_saved_prefixes}.")
          saved_prefixes.extend(flattened_saved_prefixes)
      num_shards = len(self._single_device_savers)
      sharded_saves = []
      num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
      last_device = None
      for shard, (device, saver) in enumerate(
          sorted(self._single_device_savers.items())):
        last_device = device
        with ops.device(saveable_object_util.set_cpu0(device)):
          shard_prefix = sharded_filename(tmp_checkpoint_prefix, shard,
                                          num_shards_tensor)
        saved_prefixes.append(shard_prefix)
        with ops.device(device):
          sharded_saves.append(saver.save(shard_prefix, options))
      with ops.control_dependencies(sharded_saves):
        merge_device = (
            options.experimental_io_device or
            saveable_object_util.set_cpu0(last_device))
        with ops.device(merge_device):
          return gen_io_ops.merge_v2_checkpoints(
              saved_prefixes, file_prefix, delete_old_dirs=True)
    if context.executing_eagerly() and len(self._single_device_savers) > 1:
      @def_function.function(jit_compile=False)
      def tf_function_save():
        save_fn()
      tf_function_save()
    else:
      return save_fn()
  def restore(self, file_prefix, options=None):
    options = options or checkpoint_options.CheckpointOptions()
    def restore_fn():
      restore_ops = {}
      for device, saver in sorted(self._single_device_savers.items()):
        with ops.device(device):
          restore_ops.update(saver.restore(file_prefix, options))
      for _, (_, restore_fn) in self._registered_savers.items():
        restore_fn(file_prefix)
      return restore_ops
    if context.executing_eagerly() and len(self._single_device_savers) > 1:
      @def_function.function(jit_compile=False)
      def tf_function_restore():
        restore_fn()
        return {}
      restore_ops = tf_function_restore()
    else:
      restore_ops = restore_fn()
    for callback in self._after_restore_callbacks:
      callback()
    return restore_ops
