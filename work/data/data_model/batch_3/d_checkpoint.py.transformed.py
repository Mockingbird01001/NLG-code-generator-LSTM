
from typing import Dict, List, Optional
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import save_restore
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import graph_view as graph_view_lib
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
  def __init__(self,
               mesh: layout.Mesh,
               saveable_objects: List[saveable_object.SaveableObject]):
    super().__init__(saveable_objects)
    self._mesh = mesh
  def save(
      self,
      file_prefix: str,
      options: Optional[checkpoint_options.CheckpointOptions] = None
  ) -> Optional[ops.Operation]:
    if options is not None and options.experimental_io_device is not None:
      raise ValueError(
          "Specified experimental_io_device in DTensor checkpoint is not supported."
      )
    del options
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in self._saveable_objects:
      for spec in saveable.specs:
        tensor = spec.tensor
        if tensor is not None:
          if api.device_name() != spec.device:
            tensor = api.pack(
                [tensor] * self._mesh.host_mesh().num_local_devices(),
                layout.Layout.replicated(
                    self._mesh.host_mesh(), rank=tensor.shape.rank))
          tensor_names.append(spec.name)
          tensors.append(tensor)
          tensor_slices.append(spec.slice_spec)
    return save_restore.sharded_save(
        self._mesh, file_prefix, tensor_names, tensor_slices, tensors)
  def restore(
      self,
      file_prefix: str,
      options: Optional[checkpoint_options.CheckpointOptions] = None
  ) -> Dict[str, ops.Operation]:
    if options is not None and options.experimental_io_device is not None:
      raise ValueError(
          "Specified experimental_io_device in DTensor checkpoint is not "
          "supported.")
    del options
    restore_specs = []
    tensor_structure = []
    for saveable in self._saveable_objects:
      saveable_tensor_structure = []
      tensor_structure.append(saveable_tensor_structure)
      for spec in saveable.specs:
        saveable_tensor_structure.append(spec.name)
        if isinstance(spec, d_variable.DSaveSpec):
          restore_specs.append((spec.name, spec.slice_spec, spec.dtype,
                                spec.layout, spec.global_shape))
        elif isinstance(spec, saveable_object.SaveSpec):
          restore_specs.append(
              (spec.name, spec.slice_spec, spec.dtype,
               layout.Layout.replicated(self._mesh.host_mesh(),
                                        spec.tensor.shape.rank).to_string(),
               spec.tensor.shape.as_list()))
    tensor_names, tensor_slices, tensor_dtypes, layouts, global_shapes = zip(
        *restore_specs)
    with ops.device(api.device_name()):
      restored_tensors = gen_dtensor_ops.d_tensor_restore_v2(
          prefix=file_prefix,
          tensor_names=tensor_names,
          shape_and_slices=tensor_slices,
          input_shapes=global_shapes,
          input_layouts=layouts,
          dtypes=tensor_dtypes)
    structured_restored_tensors = nest.pack_sequence_as(tensor_structure,
                                                        restored_tensors)
    restore_ops = {}
    for saveable, restored_tensors in zip(self._saveable_objects,
                                          structured_restored_tensors):
      restore_ops[saveable.name] = saveable.restore(
          restored_tensors, restored_shapes=None)
    return restore_ops
  def __init__(self, mesh: layout.Mesh, **kwargs):
    super().__init__(**kwargs)
    self._mesh = mesh
  def restore_saveables(
      self,
      tensor_saveables: Dict[str, saveable_object.SaveableObject],
      python_saveables: List[base.PythonStateSaveable],
      registered_savers: Optional[Dict[str, Dict[str, base.Trackable]]] = None
  ) -> Optional[List[ops.Operation]]:
    del registered_savers
    restore_ops = []
    reader = None
    for saveable in python_saveables:
      if reader is None:
        reader = py_checkpoint_reader.NewCheckpointReader(self.save_path_string)
      spec_names = [spec.name for spec in saveable.specs]
      saveable.python_restore([reader.get_tensor(name) for name in spec_names])
    if tensor_saveables:
      validated_saveables = saveable_object_util.validate_and_slice_inputs(
          tensor_saveables)
      validated_names = set(saveable.name for saveable in validated_saveables)
      if set(tensor_saveables.keys()) != validated_names:
        raise AssertionError(
            ("Saveable keys changed when validating. Got back %s, was "
             "expecting %s") % (tensor_saveables.keys(), validated_names))
      new_restore_ops = _DSaver(self._mesh, validated_saveables).restore(
          self.save_path_tensor, self.options)
      if not context.executing_eagerly():
        for name, restore_op in sorted(new_restore_ops.items()):
          restore_ops.append(restore_op)
          assert name not in self.restore_ops_by_name
          self.restore_ops_by_name[name] = restore_op
    return restore_ops
def saver_with_op_caching(mesh, obj, attached_dependencies=None):
  if context.executing_eagerly():
    saveables_cache = None
  else:
    saveables_cache = object_identity.ObjectIdentityWeakKeyDictionary()
  return DTrackableSaver(
      mesh,
      graph_view_lib.ObjectGraphView(
          weakref.ref(obj),
          saveables_cache=saveables_cache,
          attached_dependencies=attached_dependencies))
class DTrackableSaver(util.TrackableSaver):
  def __init__(self, mesh: layout.Mesh, graph_view):
    super(DTrackableSaver, self).__init__(graph_view)
    self._mesh = mesh
  def _save_cached_when_graph_building(self, file_prefix, object_graph_tensor,
                                       options, update_ckpt_state=False):
    """Create or retrieve save ops, overrides parents's private method.
    Args:
      file_prefix: The prefix for saved checkpoint files.
      object_graph_tensor: A `Tensor` to which the current object graph will be
        fed.
      options: `CheckpointOptions` object.
      update_ckpt_state: Optional bool flag. Indiciate whether the internal
        checkpoint state needs to be updated. This is used for async checkpoint,
        which DTrackableSaver currently does not support.
    TODO(chienchunh): Implement async checkpoint for DTrackableSaver.
    Returns:
      A two-element tuple with a filename tensor and a feed_dict of tensors to
      feed when running it (if graph building). The feed dict contains the
      current object graph and any Python state to be saved in the
      checkpoint. When executing eagerly only the first argument is meaningful.
    """
    (named_saveable_objects, graph_proto,
     feed_additions, unused_registered_savers) = self._gather_saveables(
         object_graph_tensor=object_graph_tensor)
    if (self._last_save_object_graph != graph_proto
        or context.executing_eagerly() or ops.inside_function()):
      saver = _DSaver(self._mesh, named_saveable_objects)
      save_op = saver.save(file_prefix, options=options)
      with ops.device("/cpu:0"):
        with ops.control_dependencies([save_op]):
          self._cached_save_operation = array_ops.identity(file_prefix)
      self._last_save_object_graph = graph_proto
    return self._cached_save_operation, feed_additions
  def restore(self, save_path, options=None):
    options = options or checkpoint_options.CheckpointOptions()
    if save_path is None:
      return util.InitializationOnlyStatus(self._graph_view, ops.uid())
    reader = py_checkpoint_reader.NewCheckpointReader(save_path)
    graph_building = not context.executing_eagerly()
    if graph_building:
      dtype_map = None
    else:
      dtype_map = reader.get_variable_to_dtype_map()
    try:
      object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
    except errors_impl.NotFoundError:
          save_path=save_path,
          dtype_map=dtype_map)
      if not graph_building:
        for existing_trackable in self._graph_view.list_objects():
          existing_trackable._maybe_initialize_trackable()
          existing_trackable._name_based_restores.add(restore_coordinator)
          existing_trackable._name_based_attribute_restore(restore_coordinator)
      return util.NameBasedSaverStatus(
          restore_coordinator, graph_view=self._graph_view)
    if graph_building:
      if self._file_prefix_placeholder is None:
        self._file_prefix_placeholder = api.pack(
            [constant_op.constant("model")] * self._mesh.num_local_devices(),
            layout.Layout.replicated(self._mesh.host_mesh(), rank=0))
      file_prefix_tensor = self._file_prefix_placeholder
      file_prefix_feed_dict = {self._file_prefix_placeholder: save_path}
    else:
      file_prefix_tensor = api.pack(
          [constant_op.constant(save_path)] * self._mesh.num_local_devices(),
          layout.Layout.replicated(self._mesh.host_mesh(), rank=0))
      file_prefix_feed_dict = None
    object_graph_proto = (trackable_object_graph_pb2.TrackableObjectGraph())
    object_graph_proto.ParseFromString(object_graph_string)
    checkpoint = _DCheckpointRestoreCoordinator(
        mesh=self._mesh,
        object_graph_proto=object_graph_proto,
        save_path=save_path,
        save_path_tensor=file_prefix_tensor,
        reader=reader,
        restore_op_cache=self._restore_op_cache,
        graph_view=self._graph_view,
        options=options)
    base.CheckpointPosition(
        checkpoint=checkpoint, proto_id=0).restore(self._graph_view.root)
    if self._graph_view.attached_dependencies:
      for ref in self._graph_view.attached_dependencies:
        if ref.name == "root":
          continue
        proto_id = None
        for proto_ref in object_graph_proto.nodes[0].children:
          if proto_ref.local_name == ref.name:
            proto_id = proto_ref.node_id
            break
        if proto_id in checkpoint.object_by_proto_id:
          continue
        base.CheckpointPosition(
            checkpoint=checkpoint, proto_id=proto_id).restore(ref.ref)
    load_status = util.CheckpointLoadStatus(
        checkpoint,
        graph_view=self._graph_view,
        feed_dict=file_prefix_feed_dict)
    return load_status
@tf_export("experimental.dtensor.DTensorCheckpoint", v1=[])
class DTensorCheckpoint(util.Checkpoint):
  def __init__(self,
               mesh: layout.Mesh,
               root=None,
               **kwargs):
    super(DTensorCheckpoint, self).__init__(root=root, **kwargs)
    self._mesh = mesh
    saver_root = self
    attached_dependencies = None
    self._save_assign_op = None
    if root:
      util._assert_trackable(root, "root")
      saver_root = root
      attached_dependencies = []
      kwargs["root"] = root
      root._maybe_initialize_trackable()
      self._save_counter = data_structures.NoDependency(
          root._lookup_dependency("save_counter"))
      self._root = data_structures.NoDependency(root)
    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      setattr(self, k, v)
      converted_v = getattr(self, k)
      util._assert_trackable(converted_v, k)
      if root:
        attached_dependencies = attached_dependencies or []
        child = root._lookup_dependency(k)
        if child is None:
          attached_dependencies.append(base.TrackableReference(k, converted_v))
        elif child != converted_v:
          raise ValueError(
              "Cannot create a Checkpoint with keyword argument {name} if "
              "root.{name} already exists.".format(name=k))
    self._saver = saver_with_op_caching(mesh, saver_root, attached_dependencies)
