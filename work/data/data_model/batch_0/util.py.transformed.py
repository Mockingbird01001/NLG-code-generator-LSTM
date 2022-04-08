
import abc
import collections
import functools
import glob
import os
import threading
import time
import weakref
import six
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import graph_view as graph_view_lib
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
_SESSION_PROVIDER = None
_END_TIME_OF_LAST_WRITE = None
_END_TIME_OF_LAST_WRITE_LOCK = threading.Lock()
_CHECKPOINT_V1 = "checkpoint_v1"
_CHECKPOINT_V2 = "checkpoint_v2"
_ASYNC_CHECKPOINT_THREAD = None
def _get_duration_microseconds(start_time_seconds, end_time_seconds):
  if end_time_seconds < start_time_seconds:
    return 0
  return round((end_time_seconds - start_time_seconds) * 1000000)
@tf_export("__internal__.tracking.register_session_provider", v1=[])
def register_session_provider(session_provider):
  global _SESSION_PROVIDER
  _SESSION_PROVIDER = session_provider
def get_session():
  session = ops.get_default_session()
  if session is None:
    global _SESSION_PROVIDER
    if _SESSION_PROVIDER is not None:
  return session
def _get_checkpoint_size(prefix):
  size = 0
  files = glob.glob("{}*".format(prefix))
  for file in files:
    size += metrics.CalculateFileSize(file)
  return size
class ObjectGraphProtoPrettyPrinter(object):
  __slots__ = ["_object_graph_proto", "_node_name_cache"]
  def __init__(self, object_graph_proto):
    self._object_graph_proto = object_graph_proto
    self._node_name_cache = None
  @property
  def node_names(self):
    if self._node_name_cache is not None:
      return self._node_name_cache
    path_to_root = {}
    path_to_root[0] = ("(root)",)
    to_visit = collections.deque([0])
    while to_visit:
      node_id = to_visit.popleft()
      obj = self._object_graph_proto.nodes[node_id]
      for child in obj.children:
        if child.node_id not in path_to_root:
          path_to_root[child.node_id] = (
              path_to_root[node_id] + (child.local_name,))
          to_visit.append(child.node_id)
    node_names = {}
    for node_id, path_to_root in path_to_root.items():
      node_names[node_id] = ".".join(path_to_root)
    for node_id, node in enumerate(self._object_graph_proto.nodes):
      for slot_reference in node.slot_variables:
        node_names[slot_reference.slot_variable_node_id] = (
            f"{node_names[node_id]}'s state '{slot_reference.slot_name}' for "
            f"{node_names[slot_reference.original_variable_node_id]}")
    self._node_name_cache = node_names
    return node_names
class _CheckpointRestoreCoordinatorDeleter(object):
  __slots__ = [
      "expect_partial", "object_graph_proto", "matched_proto_ids",
      "unused_attributes"
  ]
  def __init__(self, expect_partial, object_graph_proto, matched_proto_ids,
               unused_attributes):
    self.expect_partial = expect_partial
    self.object_graph_proto = object_graph_proto
    self.matched_proto_ids = matched_proto_ids
    self.unused_attributes = unused_attributes
  def set_expect_partial(self, expect_partial):
    self.expect_partial = expect_partial
  def __del__(self):
    if self.expect_partial:
      return
    if logging is None:
      log_fn = print
    else:
      log_fn = logging.warning
    unused_nodes_in_checkpoint = []
    unrestored_attributes_in_object = []
    pretty_printer = ObjectGraphProtoPrettyPrinter(self.object_graph_proto)
    for node_id, node in enumerate(self.object_graph_proto.nodes):
      if not node.attributes:
        continue
      if node_id not in self.matched_proto_ids:
        unused_nodes_in_checkpoint.append(pretty_printer.node_names[node_id])
    for node_id, attribute_name in self.unused_attributes.items():
      unrestored_attributes_in_object.append((
          pretty_printer.node_names[node_id], attribute_name))
    if unused_nodes_in_checkpoint or unrestored_attributes_in_object:
      log_fn("Detecting that an object or model or tf.train.Checkpoint is being"
             " deleted with unrestored values. See the following logs for the "
             "specific values in question. To silence these warnings, use "
             "`status.expect_partial()`. See "
             "for details about the status object returned by the restore "
             "function.")
      for node_path in unused_nodes_in_checkpoint:
        log_fn("Value in checkpoint could not be found in the restored object: "
               f"{node_path}")
      for node_path, attr in unrestored_attributes_in_object:
        log_fn("An attribute in the restored object could not be found in the "
               f"checkpoint. Object: {node_path}, attribute: {attr}")
class _CheckpointRestoreCoordinator(object):
  def __init__(self, object_graph_proto, save_path, save_path_tensor, reader,
               restore_op_cache, graph_view, options):
    """Specify the checkpoint being loaded.
    Args:
      object_graph_proto: The TrackableObjectGraph protocol buffer associated
        with this checkpoint.
      save_path: A string, the path to the checkpoint, as returned by
        `tf.train.latest_checkpoint`.
      save_path_tensor: A string `Tensor` which contains or will be fed the save
        path.
      reader: A `CheckpointReader` for `save_path`. If None,
        `_CheckpointRestoreCoordinator` will initialize one itself.
      restore_op_cache: A dictionary shared between
        `_CheckpointRestoreCoordinator`s for the same Python objects, used to
        look up restore ops by name to avoid re-creating them across multiple
        `restore()` calls.
      graph_view: A graph_view_lib.ObjectGraphView object for the restored
        objects.
      options: A CheckpointOptions object.
    """
    self.options = options
    self.object_graph_proto = object_graph_proto
    self.restore_uid = ops.uid()
    self.unused_attributes = {}
    self.object_by_proto_id = weakref.WeakValueDictionary()
    self.matched_proto_ids = set()
    self.all_python_objects = object_identity.ObjectIdentityWeakSet()
    self.save_path_tensor = save_path_tensor
    self.save_path_string = save_path
    self.reader = reader
    if self.reader is None:
      self.reader = py_checkpoint_reader.NewCheckpointReader(save_path)
    self.dtype_map = reader.get_variable_to_dtype_map()
    self.shape_map = reader.get_variable_to_shape_map()
    self.restore_ops = []
    self.restore_ops_by_name = restore_op_cache
    self.graph_view = graph_view
    self.new_restore_ops_callback = None
    self.deferred_slot_restorations = {}
    self.slot_restorations = {}
    self.expect_partial_attr = False
    for node_index, node in enumerate(self.object_graph_proto.nodes):
      for slot_reference in node.slot_variables:
        self.slot_restorations.setdefault(
            slot_reference.original_variable_node_id, []).append(
                    optimizer_id=node_index,
                    slot_variable_id=slot_reference.slot_variable_node_id,
                    slot_name=slot_reference.slot_name))
    self._deleter = _CheckpointRestoreCoordinatorDeleter(
        self.expect_partial_attr,
        self.object_graph_proto,
        self.matched_proto_ids,
        self.unused_attributes)
  @property
  def expect_partial(self):
    return self.expect_partial_attr
  @expect_partial.setter
  def expect_partial(self, expect_partial):
    self.expect_partial_attr = expect_partial
    self._deleter.set_expect_partial(expect_partial)
  def new_restore_ops(self, new_ops):
    self.restore_ops.extend(new_ops)
    if self.new_restore_ops_callback:
  def restore_saveables(self,
                        tensor_saveables,
                        python_saveables,
                        registered_savers=None):
    restore_ops = []
    for saveable in python_saveables:
      spec_names = [spec.name for spec in saveable.specs]
      saveable.python_restore(
          [self.reader.get_tensor(name) for name in spec_names])
    if tensor_saveables or registered_savers:
      validated_saveables = saveable_object_util.validate_and_slice_inputs(
          tensor_saveables)
      validated_names = set(saveable.name for saveable in validated_saveables)
      if set(tensor_saveables.keys()) != validated_names:
        raise AssertionError(
            "Saveable keys changed when validating. Got back "
            f"{tensor_saveables.keys()}, was expecting {validated_names}")
      new_restore_ops = functional_saver.MultiDeviceSaver(
          validated_saveables,
          registered_savers).restore(self.save_path_tensor, self.options)
      if not context.executing_eagerly():
        for name, restore_op in sorted(new_restore_ops.items()):
          restore_ops.append(restore_op)
          assert name not in self.restore_ops_by_name
          self.restore_ops_by_name[name] = restore_op
    return restore_ops
class _NameBasedRestoreCoordinator(object):
  def __init__(self, save_path, dtype_map=None):
    self.save_path = save_path
    self.dtype_map = dtype_map
    self.unused_attributes = object_identity.ObjectIdentityWeakKeyDictionary()
    self.restore_uid = ops.uid()
  def globally_named_object_attributes(self, trackable):
    """Create globally named SaveableObjects from attributes.
    If an object's attribute has no global name specified (default construction
    for the SaveableObject factory), records the failure in
    `self.unused_attributes` (which can then be used to make status assertions
    fail; see `NameBasedSaverStatus`).
    Args:
      trackable: An object to save.
    Yields:
      SaveableObjects for `trackable`'s attributes.
    """
    for attribute_name, saveable_factory in (
      if callable(saveable_factory):
        try:
          saveable = saveable_factory()
        except TypeError:
          self.unused_attributes.setdefault(trackable,
                                            []).append(attribute_name)
          continue
      else:
        saveable = saveable_factory
      names_to_saveables = saveable_object_util.op_list_to_dict(
          [saveable], convert_variable_to_tensor=False)
      for name, op in names_to_saveables.items():
        for saveable_object in saveable_object_util.saveable_objects_for_op(
            op=op, name=name):
          yield saveable_object
  def eager_restore(self, trackable):
    assert context.executing_eagerly()
    for saveable in self.globally_named_object_attributes(trackable):
      restored_tensors = []
      tensor_missing = False
      for spec in saveable.specs:
        if spec.name in self.dtype_map:
          with ops.device("cpu:0"):
            restored, = io_ops.restore_v2(
                prefix=self.save_path,
                tensor_names=[spec.name],
                shape_and_slices=[""],
                dtypes=[self.dtype_map[spec.name]],
                name="%s_checkpoint_read" % (spec.name,))
          restored_tensors.append(array_ops.identity(restored))
        else:
          tensor_missing = True
      if tensor_missing:
        self.unused_attributes.setdefault(trackable, []).append(saveable.name)
      else:
        saveable.restore(
            restored_tensors=restored_tensors, restored_shapes=None)
def _default_getter(name,
                    shape,
                    dtype,
                    initializer=None,
                    partition_info=None,
                    **kwargs):
  dtype = dtypes.as_dtype(dtype)
  shape_object = tensor_shape.as_shape(shape)
  with ops.init_scope():
    if initializer is None:
      initializer, initializing_from_value = (
              name=name,
              shape=shape_object,
              dtype=dtype))
    else:
      initializing_from_value = not callable(initializer)
    variable_dtype = dtype.base_dtype
    if initializing_from_value:
      if shape is not None:
        raise ValueError("If initializer is a constant, do not specify shape.")
      initial_value = initializer
    else:
      if isinstance(initializer, type(init_ops.Initializer)):
        initializer = initializer(dtype=dtype)
      shape_list = None if shape is None else shape_object.as_list()
      if "partition_info" in tf_inspect.getargspec(initializer).args:
        initial_value = functools.partial(initializer,
                                          shape_list,
                                          dtype=dtype,
                                          partition_info=partition_info)
      else:
        initial_value = functools.partial(initializer,
                                          shape_list,
                                          dtype=dtype)
    return variables.VariableV1(
        initial_value=initial_value,
        name=name,
        dtype=variable_dtype,
        use_resource=True,
        **kwargs)
def add_variable(trackable,
                 name,
                 shape=None,
                 dtype=dtypes.float32,
                 initializer=None,
                 trainable=True):
      name=name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      getter=_default_getter,
      trainable=trainable)
def object_metadata(save_path):
  """Retrieves information about the objects in a checkpoint.
  Example usage:
  ```python
  object_graph = tf.contrib.checkpoint.object_metadata(
      tf.train.latest_checkpoint(checkpoint_directory))
  ckpt_variable_names = set()
  for node in object_graph.nodes:
    for attribute in node.attributes:
      ckpt_variable_names.add(attribute.full_name)
  ```
  Args:
    save_path: The path to the checkpoint, as returned by `save` or
      `tf.train.latest_checkpoint`.
  Returns:
    A parsed `tf.contrib.checkpoint.TrackableObjectGraph` protocol buffer.
  Raises:
    ValueError: If an object graph was not found in the checkpoint.
  """
  reader = py_checkpoint_reader.NewCheckpointReader(save_path)
  try:
    object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
  except errors_impl.NotFoundError:
    raise ValueError(
        f"The specified checkpoint \"{save_path}\" does not appear to be "
        "object-based (saved with TF2) since it is missing the key "
        f"\"{base.OBJECT_GRAPH_PROTO_KEY}\". Likely it was created with the "
        "TF1 name-based saver and does not contain an object dependency graph.")
  object_graph_proto = (trackable_object_graph_pb2.TrackableObjectGraph())
  object_graph_proto.ParseFromString(object_graph_string)
  return object_graph_proto
def list_objects(root_trackable):
  """Traverse the object graph and list all accessible objects.
  Looks for `Trackable` objects which are dependencies of
  `root_trackable`. Includes slot variables only if the variable they are
  slotting for and the optimizer are dependencies of `root_trackable`
  (i.e. if they would be saved with a checkpoint).
  Args:
    root_trackable: A `Trackable` object whose dependencies should be flattened.
  Returns:
    A flat list of objects.
  """
  return graph_view_lib.ObjectGraphView(root_trackable).list_objects()
def gather_initializers(root_trackable):
  """Traverse the object graph and find initialization ops.
  Looks for `Trackable` objects which are dependencies of
  `root_trackable` and which have an `initializer` property. Includes
  initializers for slot variables only if the variable they are slotting for and
  the optimizer are dependencies of `root_trackable` (i.e. if they would be
  saved with a checkpoint).
  Args:
    root_trackable: A `Trackable` object to gather initializers for.
  Returns:
    A list of initialization ops.
  """
  trackable_objects = list_objects(root_trackable)
  return [
      c.initializer
      for c in trackable_objects
      if hasattr(c, "initializer") and c.initializer is not None
  ]
@tf_contextlib.contextmanager
def capture_dependencies(template):
  """Capture variables created within this scope as `Template` dependencies.
  Requires that `template.variable_scope` is active.
  This scope is intended as a compatibility measure, allowing a trackable
  object to add dependencies on variables created in a block of code which is
  not aware of object-based saving (and instead uses variable names
  heavily). This is how `Template` objects add dependencies on variables and
  sub-`Template`s. Where possible, use `tf.compat.v1.make_template` directly.
  Args:
    template: The `Template` object to register dependencies with.
  Yields:
    None (when used as a context manager).
  """
  name_prefix = template.variable_scope.name
  def _trackable_custom_creator(next_creator,
                                name,
                                initial_value,
                                trackable_parent=None,
                                **kwargs):
    """A variable creation hook which adds Trackable dependencies.
    Set for example during a `Template`'s first wrapped function
    execution. Ensures that (a) `template` depends on any trackable
    objects using their own `capture_dependencies` scope inside this scope which
    create variables, and (b) that any variables not in a more deeply nested
    scope are added as dependencies directly.
    The `trackable_parent` argument is passed between custom creators but
    ignored when the variable object itself is created. This argument indicates
    (if not `None`) that a more deeply nested scope has already added the
    variable as a dependency, and that parent scopes should add a dependency on
    that object rather than on the variable directly.
    Args:
      next_creator: See `variable_scope.variable_creator_scope`; the next
        creator in the chain.
      name: The (full, scope-influenced) name of the variable. The `name_prefix`
        itself is stripped for the purposes of object-based dependency tracking,
        but scopes opened within this scope are respected.
      initial_value: See `variable_scope.variable_creator_scope`. Taken
        explicitly so the argument can be re-named and used with
        `Trackable._add_variable_with_custom_getter`.
      trackable_parent: If not None, a more deeply nested trackable object and
        its name prefix which were passed to `capture_dependencies` to add a
        dependency on (rather than depending on the variable directly).
      **kwargs: Passed through to the next creator.
    Returns:
      The output of `next_creator`: the fetched/created variable object.
    """
    def _call_next_creator_renaming_initializer(initializer, **inner_kwargs):
      return next_creator(initial_value=initializer, name=name, **inner_kwargs)
    if name is not None and name.startswith(name_prefix):
      scope_stripped_name = name[len(name_prefix) + 1:]
      if not trackable_parent:
            initializer=initial_value,
            name=scope_stripped_name,
            getter=_call_next_creator_renaming_initializer,
            overwrite=True,
            trackable_parent=(template, name_prefix),
            **kwargs)
      else:
        parent_object, parent_name_prefix = trackable_parent
            parent_object,
            name=parent_name_prefix[len(name_prefix) + 1:],
            overwrite=True)
    return next_creator(
        name=name,
        initial_value=initial_value,
        trackable_parent=(template, name_prefix),
        **kwargs)
  with variable_scope.variable_creator_scope(_trackable_custom_creator):
    yield
class _LoadStatus(object):
  @abc.abstractmethod
  def assert_consumed(self):
    pass
  @abc.abstractmethod
  def assert_existing_objects_matched(self):
    pass
  @abc.abstractmethod
  def assert_nontrivial_match(self):
    pass
  @abc.abstractmethod
  def run_restore_ops(self, session=None):
    pass
  @abc.abstractmethod
  def initialize_or_restore(self, session=None):
    pass
  def expect_partial(self):
    return self
@tf_export("__internal__.tracking.streaming_restore", v1=[])
def streaming_restore(status, session=None):
  """When graph building, runs restore ops as soon as they come in.
  Args:
    status: A _LoadStatus objects from an object-based saver's restore().
      Streaming restore from name-based checkpoints is not currently supported.
    session: A session to run new restore ops in.
  """
  if context.executing_eagerly():
    return
  if session is None:
    session = get_session()
  if isinstance(status, NameBasedSaverStatus):
    raise NotImplementedError(
        "Streaming restore not supported from name-based checkpoints when "
        "graph building. File a feature request if this limitation bothers "
        "you. As a workaround, consider either using tf.train.Checkpoint to "
        "load name-based checkpoints or enabling eager execution.")
  status.run_restore_ops(session=session)
  status._checkpoint.new_restore_ops_callback = (
      lambda ops: session.run(ops, feed_dict=status._feed_dict))
def _objects_with_attributes(full_list):
class CheckpointLoadStatus(_LoadStatus):
  """Checks the status of checkpoint loading and manages restore ops.
  Returned from `Saver.restore`. Since `restore` may defer the loading of values
  in the checkpoint which don't yet have corresponding Python objects,
  `CheckpointLoadStatus` provides a callback to verify that checkpoint loading
  is complete (`assert_consumed`).
  When graph building, `restore` does not run restore ops itself since their
  creation may be deferred. The `run_restore_ops` method must be called once all
  Python objects with values to restore have been created and added to the
  dependency graph (this does not necessarily have to be the whole checkpoint;
  calling `run_restore_ops` while `assert_consumed` fails is supported and will
  partially restore the checkpoint).
  See `Saver.restore` for usage examples.
  """
  def __init__(self, checkpoint, feed_dict, graph_view):
    self._checkpoint = checkpoint
    self._feed_dict = feed_dict
    self._object_graph_view = graph_view
    self._root = graph_view.root
  def assert_consumed(self):
    pretty_printer = ObjectGraphProtoPrettyPrinter(
        self._checkpoint.object_graph_proto)
    self.assert_existing_objects_matched()
    for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
      if not node.attributes:
        continue
      trackable = self._checkpoint.object_by_proto_id.get(node_id, None)
      if trackable is None:
        raise AssertionError(
            "Unresolved object in checkpoint "
            f"{pretty_printer.node_names[node_id]}: {node}")
    if self._checkpoint.slot_restorations:
      raise AssertionError(
          f"Unresolved slot restorations: {self._checkpoint.slot_restorations}")
    if self._checkpoint.unused_attributes:
      unused_attribute_messages = []
      for node_id, attribute in six.iteritems(
          self._checkpoint.unused_attributes):
        obj = self._checkpoint.object_by_proto_id[node_id]
        unused_attribute_messages.append(
            f"{pretty_printer.node_names[node_id]} ({obj}): {attribute}")
      joined_attribute_messages = "\n".join(unused_attribute_messages)
      raise AssertionError(
          "Unused attributes in these objects (the attributes exist in the "
          f"checkpoint but were not restored):\n{joined_attribute_messages}")
    return self
  def assert_existing_objects_matched(self):
    """Asserts that trackable Python objects have been matched.
    Note that this is a weaker assertion than `assert_consumed`. It will only
    fail for existing Python objects which are (transitive) dependencies of the
    root object and which do not have an entry in the checkpoint.
    It will not fail, for example, if a `tf.keras.Layer` object has not yet been
    built and so has not created any `tf.Variable` objects.
    Returns:
      `self` for chaining.
    Raises:
      AssertionError: If a Python object exists in the transitive dependencies
        of the root object but does not have a value in the checkpoint.
    """
    for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
      trackable = self._checkpoint.object_by_proto_id.get(node_id, None)
      if (trackable is not None and
        raise AssertionError(
            f"Object {node} not assigned a value from checkpoint.")
    for trackable_object in self._object_graph_view.list_objects():
      if (isinstance(trackable_object,
                     data_structures.TrackableDataStructure) and
        continue
      self._checkpoint.all_python_objects.add(trackable_object)
    unused_python_objects = (
        object_identity.ObjectIdentitySet(
            _objects_with_attributes(
                self._checkpoint.all_python_objects)) -
        object_identity.ObjectIdentitySet(
            self._checkpoint.object_by_proto_id.values()))
    if unused_python_objects:
      num_unused_python_objects = len(list(unused_python_objects))
      num_variables_to_show = min(10, num_unused_python_objects)
      raise AssertionError(
          f"Found {num_unused_python_objects} Python objects that were "
          "not bound to checkpointed values, likely due to changes in the "
          f"Python program. Showing {num_variables_to_show} of "
          f"{num_unused_python_objects} unmatched objects: "
          f"{list(unused_python_objects)[:num_variables_to_show]}")
    return self
  def assert_nontrivial_match(self):
    for trackable_object in self._object_graph_view.list_objects():
      self._checkpoint.all_python_objects.add(trackable_object)
    if len(self._checkpoint.object_by_proto_id) <= 1:
      unused_python_objects = (
          object_identity.ObjectIdentitySet(
              _objects_with_attributes(self._checkpoint.all_python_objects)) -
          object_identity.ObjectIdentitySet(
              self._checkpoint.object_by_proto_id.values()))
      if unused_python_objects:
        raise AssertionError(
            "Nothing except the root object matched a checkpointed value. "
            "Typically this means that the checkpoint does not match the "
            "Python program. The following objects have no matching "
            f"checkpointed value: {list(unused_python_objects)}")
      else:
        raise AssertionError(
            "Nothing to load. No dependencies have been added to "
            f"{self._object_graph_view.root} yet.")
    return self
  def run_restore_ops(self, session=None):
    if context.executing_eagerly():
    if session is None:
      session = get_session()
    session.run(self._checkpoint.restore_ops, feed_dict=self._feed_dict)
  def initialize_or_restore(self, session=None):
    """Run operations to initialize or restore objects in the dependency graph.
    Any objects in the dependency graph which have initializers but are not in
    the checkpoint will have those initializers run, unless those variables are
    being restored by a later call to `tf.train.Checkpoint.restore()`.
    This method has a sibling in `InitializationOnlyStatus` which instead
    initializes variables. That type is returned if no checkpoint is specified
    in `Saver.restore`.
    Args:
      session: The session to run init/restore ops in. If `None`, uses the
        default session.
    """
    if context.executing_eagerly():
    if session is None:
      session = get_session()
    all_objects = self._object_graph_view.list_objects()
    already_initialized_objects = object_identity.ObjectIdentitySet(
        self._checkpoint.object_by_proto_id.values())
    initializers_for_non_restored_variables = [
        c.initializer for c in all_objects
        if hasattr(c, "initializer")
        and c not in already_initialized_objects
        and (getattr(c, "_update_uid", self._checkpoint.restore_uid - 1)
             < self._checkpoint.restore_uid)
    ]
    self.run_restore_ops(session=session)
    session.run(initializers_for_non_restored_variables)
  def expect_partial(self):
    self._checkpoint.expect_partial = True
    return self
class InitializationOnlyStatus(_LoadStatus):
  def __init__(self, object_graph_view, restore_uid):
    self._restore_uid = restore_uid
    self._object_graph_view = object_graph_view
    self._root = object_graph_view.root
  def assert_consumed(self):
    raise AssertionError(
        "No checkpoint specified (save_path=None); nothing is being restored.")
  def assert_existing_objects_matched(self):
    raise AssertionError(
        "No checkpoint specified (save_path=None); nothing is being restored.")
  def assert_nontrivial_match(self):
    raise AssertionError(
        "No checkpoint specified (save_path=None); nothing is being restored.")
  def run_restore_ops(self, session=None):
    raise AssertionError(
        "No checkpoint specified, so no restore ops are available "
        "(save_path=None to Saver.restore).")
  def initialize_or_restore(self, session=None):
    """Runs initialization ops for variables.
    Objects which would be saved by `Saver.save` will be initialized, unless
    those variables are being restored by a later call to
    `tf.train.Checkpoint.restore()`.
    This method does nothing when executing eagerly (initializers get run
    eagerly).
    Args:
      session: The session to run initialization ops in. If `None`, uses the
        default session.
    """
    if context.executing_eagerly():
    if session is None:
      session = get_session()
    trackable_objects = self._object_graph_view.list_objects()
    initializers = [
        c.initializer for c in trackable_objects
        if hasattr(c, "initializer") and c.initializer is not None
        and (getattr(c, "_update_uid", self._restore_uid - 1)
             < self._restore_uid)
    ]
    session.run(initializers)
_DEPRECATED_RESTORE_INSTRUCTIONS = (
    "Restoring a name-based tf.train.Saver checkpoint using the object-based "
    "restore API. This mode uses global names to match variables, and so is "
    "somewhat fragile. It also adds new restore ops to the graph each time it "
    "is called when graph building. Prefer re-encoding training checkpoints in "
    "the object-based format: run save() on the object-based saver (the same "
    "one this message is coming from) and use that checkpoint in the future.")
class NameBasedSaverStatus(_LoadStatus):
  @deprecation.deprecated(
      date=None, instructions=_DEPRECATED_RESTORE_INSTRUCTIONS)
  def __init__(self, checkpoint, object_graph_view):
    self._checkpoint = checkpoint
    self._object_graph_view = object_graph_view
    self._optionally_restored = []
    self._root = object_graph_view.root
  def add_to_optionally_restored(self, var):
    """Add a variable to the list of optionally restored variables.
    There are situations where certain variables should be ignored in assertions
    such as assert_existing_objects_matched(). One example is that of a
    checkpoint saved with train.Saver(), and restored with train.Checkpoint():
    it is possible for the train.Saver() checkpoint to be missing the internal
    `save_counter` variable, which we want to ignore on restore.
    Args:
      var: The variable to treat as optionally restored.
    """
    self._optionally_restored.append(var)
  def assert_consumed(self):
    unused_attributes = list(self._checkpoint.unused_attributes.items())
    unused_attributes = [
        a for a in unused_attributes
        if all(a[0] is not x for x in self._optionally_restored)
    ]
    if unused_attributes:
      unused_attribute_strings = [
          f"\n    {obj}: {attributes}" for obj, attributes in unused_attributes]
      raise AssertionError(
          "Some objects had attributes which were not restored: "
          f"{unused_attribute_strings}")
    for trackable in self._object_graph_view.list_objects():
      trackable._maybe_initialize_trackable()
      if trackable._update_uid < self._checkpoint.restore_uid:
        raise AssertionError(f"Object not restored: {trackable}")
    return self
  def assert_existing_objects_matched(self):
    return self.assert_consumed()
  def assert_nontrivial_match(self):
    return self.assert_consumed()
  def _gather_saveable_objects(self):
    objects = self._object_graph_view.list_objects()
    saveable_objects = []
    for trackable in objects:
      trackable._maybe_initialize_trackable()
      if trackable._update_uid < self._checkpoint.restore_uid:
        trackable._update_uid = self._checkpoint.restore_uid
      else:
        continue
      saveable_objects.extend(
          self._checkpoint.globally_named_object_attributes(trackable))
    return saveable_objects
  def run_restore_ops(self, session=None):
    if context.executing_eagerly():
    if session is None:
      session = get_session()
    with ops.device("/cpu:0"):
      saveables = self._gather_saveable_objects()
      v1_saver_lib.Saver(saveables).restore(
          sess=session, save_path=self._checkpoint.save_path)
  def initialize_or_restore(self, session=None):
    self.run_restore_ops(session=session)
class _SessionWithFeedDictAdditions(session_lib.SessionInterface):
  def __init__(self, session, feed_additions):
    self._wrapped_session = session
    self._feed_additions = feed_additions
  def run(self, fetches, feed_dict=None, **kwargs):
    if feed_dict is None:
      feed_dict = {}
    else:
      feed_dict = feed_dict.copy()
    feed_dict.update(self._feed_additions)
    return self._wrapped_session.run(
        fetches=fetches, feed_dict=feed_dict, **kwargs)
@tf_export("__internal__.tracking.TrackableSaver", v1=[])
class TrackableSaver(object):
  """Saves and restores a `Trackable` object and its dependencies.
  See `Trackable` for details of dependency management. `Saver` wraps
  `tf.compat.v1.train.Saver` for saving, including extra information about the
  graph of
  dependencies between Python objects. When restoring, it uses this information
  about the save-time dependency graph to more robustly match objects with their
  checkpointed values. When executing eagerly, it supports restoring variables
  on object creation (see `Saver.restore`).
  Values in a checkpoint are mapped to `Trackable` Python objects
  (`Variable`s, `Optimizer`s, `Layer`s) based on the names provided when the
  checkpoint was written. To avoid breaking existing checkpoints when modifying
  a class, dependency names (the names of attributes to which `Trackable`
  objects are assigned) may not change. These names are local to objects, in
  contrast to the `Variable.name`-based save/restore from
  `tf.compat.v1.train.Saver`, and
  so allow additional program transformations.
  """
  def __init__(self, graph_view):
    self._file_prefix_placeholder = None
    self._object_graph_feed_tensor = None
    self._last_save_object_graph = None
    self._file_prefix_feed_tensor = None
    self._cached_save_operation = None
    self._restore_op_cache = {}
    self._graph_view = graph_view
  def _gather_saveables(self, object_graph_tensor=None):
    named_saveable_objects, graph_proto, feed_additions, registered_savers = (
        self._graph_view.serialize_object_graph_with_registered_savers())
    if object_graph_tensor is None:
      with ops.device("/cpu:0"):
        object_graph_tensor = constant_op.constant(
            graph_proto.SerializeToString(), dtype=dtypes.string)
    else:
      feed_additions.update(
          {object_graph_tensor: graph_proto.SerializeToString()})
    assert base.OBJECT_GRAPH_PROTO_KEY not in named_saveable_objects
    named_saveable_objects.append(
        base.NoRestoreSaveable(
            tensor=object_graph_tensor, name=base.OBJECT_GRAPH_PROTO_KEY))
    return (named_saveable_objects, graph_proto, feed_additions,
            registered_savers)
  def _save_cached_when_graph_building(self,
                                       file_prefix,
                                       object_graph_tensor,
                                       options,
                                       update_ckpt_state=False):
    """Create or retrieve save ops.
    Args:
      file_prefix: The prefix for saved checkpoint files.
      object_graph_tensor: A `Tensor` to which the current object graph will be
        fed.
      options: `CheckpointOptions` object.
      update_ckpt_state: Optional bool flag. Indiciate whether the internal
        checkpoint state needs to be updated.
    Returns:
      A two-element tuple with a filename tensor and a feed_dict of tensors to
      feed when running it (if graph building). The feed dict contains the
      current object graph and any Python state to be saved in the
      checkpoint. When executing eagerly only the first argument is meaningful.
    """
    (named_saveable_objects, graph_proto, feed_additions,
     registered_savers) = self._gather_saveables(
         object_graph_tensor=object_graph_tensor)
    def _run_save():
      if (self._last_save_object_graph != graph_proto
          or context.executing_eagerly() or ops.inside_function()):
        saver = functional_saver.MultiDeviceSaver(named_saveable_objects,
                                                  registered_savers)
        save_op = saver.save(file_prefix, options=options)
        with ops.device("/cpu:0"):
          with ops.control_dependencies([save_op]):
            self._cached_save_operation = array_ops.identity(file_prefix)
        self._last_save_object_graph = graph_proto
      return self._cached_save_operation, feed_additions
    def _copy_tensors():
      for saveable in named_saveable_objects:
        original_device = saveable.device
        with ops.device(original_device):
          for spec in saveable.specs:
            tensor = spec.tensor
            device = spec.device
            if tensor is not None:
              with ops.device(saveable_object_util.set_cpu0(device)):
                spec.device = saveable_object_util.set_cpu0(device)
    def _async_save_fn():
      _run_save()
      if update_ckpt_state:
        _update_checkpoint_state_internal(
            _convert_file_name_tensor_to_string(file_prefix))
    if options.experimental_enable_async_checkpoint:
      _copy_tensors()
      global _ASYNC_CHECKPOINT_THREAD
      if _ASYNC_CHECKPOINT_THREAD is not None:
        _ASYNC_CHECKPOINT_THREAD.join()
      _ASYNC_CHECKPOINT_THREAD = threading.Thread(target=_async_save_fn)
      _ASYNC_CHECKPOINT_THREAD.start()
      self._cached_save_operation = file_prefix
      return self._cached_save_operation, feed_additions
    return _run_save()
  def save(self, file_prefix, checkpoint_number=None, session=None,
           options=None, update_ckpt_state=False):
    """Save a training checkpoint.
    The saved checkpoint includes variables created by this object and any
    Trackable objects it depends on at the time `Saver.save()` is called.
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `checkpoint_number`, if provided.
      checkpoint_number: An integer variable or Tensor, used to number
        checkpoints. Typically this value is saved along with other variables in
        training checkpoints, which will happen automatically if it was created
        by `root_trackable` or one of its dependencies (via
        `Trackable._add_variable`).
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.
      options: Optional `tf.train.CheckpointOptions` object.
      update_ckpt_state: Optional bool flag. Indiciate whether the internal
        checkpoint state needs to be updated. Set this to True only if calling
        from tf.train.Checkpoint.save() to enable updating the checkpoint state.
        By default this is set to False, i.e., not updating checkpoint state.
    Returns:
      The full path to the checkpoint.
    """
    options = options or checkpoint_options.CheckpointOptions()
    feed_dict = {}
    use_session = (not context.executing_eagerly() and
                   not ops.inside_function())
    if checkpoint_number:
      file_prefix = "%s-%d" % (file_prefix, checkpoint_number)
    if use_session:
      if self._object_graph_feed_tensor is None:
        with ops.device("/cpu:0"):
          self._object_graph_feed_tensor = constant_op.constant(
              "", dtype=dtypes.string)
          self._file_prefix_feed_tensor = constant_op.constant(
              "", dtype=dtypes.string)
      object_graph_tensor = self._object_graph_feed_tensor
      file_prefix_tensor = self._file_prefix_feed_tensor
      feed_dict[file_prefix_tensor] = file_prefix
    else:
      with ops.device("/cpu:0"):
        file_prefix_tensor = ops.convert_to_tensor(
            file_prefix, dtype=dtypes.string)
      object_graph_tensor = None
    if not tensor_util.is_tensor(file_prefix):
      file_io.recursive_create_dir(os.path.dirname(file_prefix))
    save_path, new_feed_additions = self._save_cached_when_graph_building(
        file_prefix_tensor, object_graph_tensor, options, update_ckpt_state)
    if new_feed_additions:
      feed_dict.update(new_feed_additions)
    if not use_session:
      session = None
    elif session is None:
      session = get_session()
    if session:
      return session.run(save_path, feed_dict=feed_dict)
    else:
      return save_path
  def restore(self, save_path, options=None):
    """Restore a training checkpoint.
    Restores `root_trackable` and any objects that it tracks
    (transitive). Either assigns values immediately if variables to restore have
    been created already, or defers restoration until the variables are
    created. Dependencies added to the `root_trackable` passed to the
    constructor after this call will be matched if they have a corresponding
    object in the checkpoint.
    When building a graph, restorations are added to the graph but not run.
    ```python
    saver = Saver(root)
    saver.restore(path)
    ```
    To ensure that loading is complete and no more deferred restorations will
    take place, you can use the `assert_consumed()` method of the status object
    returned by the `restore` call.
    The assert will raise an exception unless every object was matched and all
    checkpointed values have a matching variable object.
    ```python
    saver = Saver(root)
    saver.restore(path).assert_consumed()
    ```
    When graph building, `assert_consumed()` indicates that all of the restore
    ops which will be created for this checkpoint have been created. They can be
    run via the `run_restore_ops()` function of the status object:
    ```python
    saver.restore(path).assert_consumed().run_restore_ops()
    ```
    If the checkpoint has not been consumed completely, then the list of restore
    ops will grow as more objects are added to the dependency graph.
    Name-based `tf.compat.v1.train.Saver` checkpoints can be loaded using this
    method. There is no deferred loading, and names are used to match
    variables. No restore ops are created/run until `run_restore_ops()` or
    `initialize_or_restore()` are called on the returned status object, even
    when executing eagerly. Re-encode name-based checkpoints using this
    object-based `Saver.save` as soon as possible.
    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If None (as when there is no latest
        checkpoint for `tf.train.latest_checkpoint` to return), returns an
        object which may run initializers for objects in the dependency graph.
        If the checkpoint was written by the name-based
        `tf.compat.v1.train.Saver`, names are used to match variables.
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      A load status object, which can be used to make assertions about the
      status of checkpoint restoration and run initialization/restore ops
      (of type `CheckpointLoadStatus`, or `InitializationOnlyStatus` if
      `save_path` is `None`).
      If `save_path` points to a name-based checkpoint, a `NameBasedSaverStatus`
      object is returned which runs restore ops from a name-based saver.
    Raises:
      RuntimeError: When a checkpoint file saved by async checkpoint is not
        available upon restore().
    """
    options = options or checkpoint_options.CheckpointOptions()
    if save_path is None:
      return InitializationOnlyStatus(self._graph_view, ops.uid())
    global _ASYNC_CHECKPOINT_THREAD
    if _ASYNC_CHECKPOINT_THREAD is not None:
      _ASYNC_CHECKPOINT_THREAD.join()
    reader = py_checkpoint_reader.NewCheckpointReader(save_path)
    graph_building = not context.executing_eagerly()
    if graph_building:
      dtype_map = None
    else:
      dtype_map = reader.get_variable_to_dtype_map()
    try:
      object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
    except errors_impl.NotFoundError:
      restore_coordinator = _NameBasedRestoreCoordinator(
          save_path=save_path,
          dtype_map=dtype_map)
      if not graph_building:
        for existing_trackable in self._graph_view.list_objects():
          existing_trackable._maybe_initialize_trackable()
          existing_trackable._name_based_restores.add(restore_coordinator)
          existing_trackable._name_based_attribute_restore(restore_coordinator)
      return NameBasedSaverStatus(
          restore_coordinator,
          object_graph_view=self._graph_view)
    if graph_building:
      if self._file_prefix_placeholder is None:
        with ops.device("/cpu:0"):
          self._file_prefix_placeholder = constant_op.constant("model")
      file_prefix_tensor = self._file_prefix_placeholder
      file_prefix_feed_dict = {self._file_prefix_placeholder: save_path}
    else:
      with ops.device("/cpu:0"):
        file_prefix_tensor = constant_op.constant(save_path)
      file_prefix_feed_dict = None
    object_graph_proto = (trackable_object_graph_pb2.TrackableObjectGraph())
    object_graph_proto.ParseFromString(object_graph_string)
    checkpoint = _CheckpointRestoreCoordinator(
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
        if proto_id is None:
          continue
        base.CheckpointPosition(
            checkpoint=checkpoint, proto_id=proto_id).restore(ref.ref)
    load_status = CheckpointLoadStatus(
        checkpoint,
        graph_view=self._graph_view,
        feed_dict=file_prefix_feed_dict)
    return load_status
def frozen_saver(root_trackable):
  named_saveable_objects, registered_savers = graph_view_lib.ObjectGraphView(
      root_trackable).frozen_saveables_and_savers()
  return functional_saver.MultiDeviceSaver(named_saveable_objects,
                                           registered_savers)
def saver_with_op_caching(obj, attached_dependencies=None):
  if context.executing_eagerly():
    saveables_cache = None
  else:
    saveables_cache = object_identity.ObjectIdentityWeakKeyDictionary()
  return TrackableSaver(
      graph_view_lib.ObjectGraphView(
          obj, saveables_cache=saveables_cache,
          attached_dependencies=attached_dependencies))
def _assert_trackable(obj, name):
  if not isinstance(
      obj, (base.Trackable, def_function.Function)):
    raise ValueError(
        f"`Checkpoint` was expecting {name} to be a trackable object (an "
        f"object derived from `Trackable`), got {obj}. If you believe this "
        "object should be trackable (i.e. it is part of the "
        "TensorFlow Python API and manages state), please open an issue.")
def _update_checkpoint_state_internal(file_path):
  checkpoint_management.update_checkpoint_state_internal(
      save_dir=os.path.dirname(file_path),
      model_checkpoint_path=file_path,
      all_model_checkpoint_paths=[file_path],
      save_relative_paths=True)
def _convert_file_name_tensor_to_string(tensor):
  output = tensor
  if tensor_util.is_tf_type(output):
    if context.executing_eagerly():
      output = compat.as_str(output.numpy())
  else:
    output = compat.as_str(output)
  return output
@tf_export(v1=["train.Checkpoint"])
class CheckpointV1(tracking.AutoTrackable):
  """Groups trackable objects, saving and restoring them.
  `Checkpoint`'s constructor accepts keyword arguments whose values are types
  that contain trackable state, such as `tf.compat.v1.train.Optimizer`
  implementations, `tf.Variable`, `tf.keras.Layer` implementations, or
  `tf.keras.Model` implementations. It saves these values with a checkpoint, and
  maintains a `save_counter` for numbering checkpoints.
  Example usage when graph building:
  ```python
  import tensorflow as tf
  import os
  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  train_op = optimizer.minimize( ... )
  with tf.compat.v1.Session() as session:
    status.initialize_or_restore(session)
    for _ in range(num_training_steps):
      session.run(train_op)
    checkpoint.save(file_prefix=checkpoint_prefix)
  ```
  Example usage with eager execution enabled:
  ```python
  import tensorflow as tf
  import os
  tf.compat.v1.enable_eager_execution()
  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  for _ in range(num_training_steps):
  checkpoint.save(file_prefix=checkpoint_prefix)
  ```
  `Checkpoint.save` and `Checkpoint.restore` write and read object-based
  checkpoints, in contrast to `tf.compat.v1.train.Saver` which writes and reads
  `variable.name` based checkpoints. Object-based checkpointing saves a graph of
  dependencies between Python objects (`Layer`s, `Optimizer`s, `Variable`s,
  etc.) with named edges, and this graph is used to match variables when
  restoring a checkpoint. It can be more robust to changes in the Python
  program, and helps to support restore-on-create for variables when executing
  eagerly. Prefer `tf.train.Checkpoint` over `tf.compat.v1.train.Saver` for new
  code.
  `Checkpoint` objects have dependencies on the objects passed as keyword
  arguments to their constructors, and each dependency is given a name that is
  identical to the name of the keyword argument for which it was created.
  TensorFlow classes like `Layer`s and `Optimizer`s will automatically add
  dependencies on their variables (e.g. "kernel" and "bias" for
  `tf.keras.layers.Dense`). Inheriting from `tf.keras.Model` makes managing
  dependencies easy in user-defined classes, since `Model` hooks into attribute
  assignment. For example:
  ```python
  class Regress(tf.keras.Model):
    def __init__(self):
      super(Regress, self).__init__()
      self.input_transform = tf.keras.layers.Dense(10)
    def call(self, inputs):
      x = self.input_transform(inputs)
  ```
  This `Model` has a dependency named "input_transform" on its `Dense` layer,
  which in turn depends on its variables. As a result, saving an instance of
  `Regress` using `tf.train.Checkpoint` will also save all the variables created
  by the `Dense` layer.
  When variables are assigned to multiple workers, each worker writes its own
  section of the checkpoint. These sections are then merged/re-indexed to behave
  as a single checkpoint. This avoids copying all variables to one worker, but
  does require that all workers see a common filesystem.
  While `tf.keras.Model.save_weights` and `tf.train.Checkpoint.save` save in the
  same format, note that the root of the resulting checkpoint is the object the
  save method is attached to. This means saving a `tf.keras.Model` using
  `save_weights` and loading into a `tf.train.Checkpoint` with a `Model`
  attached (or vice versa) will not match the `Model`'s variables. See the
  [guide to training
  checkpoints](https://www.tensorflow.org/guide/checkpoint) for
  details. Prefer `tf.train.Checkpoint` over `tf.keras.Model.save_weights` for
  training checkpoints.
  Attributes:
    save_counter: Incremented when `save()` is called. Used to number
      checkpoints.
  """
  def __init__(self, **kwargs):
    super(CheckpointV1, self).__init__()
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      if _END_TIME_OF_LAST_WRITE is None:
        _END_TIME_OF_LAST_WRITE = time.time()
    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      setattr(self, k, v)
      if not isinstance(
          getattr(self, k), (base.Trackable, def_function.Function)):
        raise ValueError(
            "`Checkpoint` was expecting a trackable object (an object "
            f"derived from `Trackable`), got {v}. If you believe this "
            "object should be trackable (i.e. it is part of the "
            "TensorFlow Python API and manages state), please open an issue.")
    self._save_assign_op = None
    self._saver = saver_with_op_caching(self)
  def _maybe_create_save_counter(self):
    if self._save_counter is None:
      with ops.device("/cpu:0"):
        self._save_counter = data_structures.NoDependency(
            add_variable(
                self,
                name="save_counter",
                initializer=0,
                dtype=dtypes.int64,
                trainable=False))
  def write(self, file_prefix, session=None):
    """Writes a training checkpoint.
    The checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.write()` is
    called.
    `write` does not number checkpoints, increment `save_counter`, or update the
    metadata used by `tf.train.latest_checkpoint`. It is primarily intended for
    use by higher level checkpoint management utilities. `save` provides a very
    basic implementation of these features.
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.
    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """
    start_time = time.time()
    output = self._saver.save(file_prefix=file_prefix, session=session)
    end_time = time.time()
    metrics.AddCheckpointWriteDuration(
        api_label=_CHECKPOINT_V1,
        microseconds=_get_duration_microseconds(start_time, end_time))
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      metrics.AddTrainingTimeSaved(
          api_label=_CHECKPOINT_V1,
          microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE,
                                                  end_time))
      _END_TIME_OF_LAST_WRITE = end_time
    if tensor_util.is_tf_type(output):
      if context.executing_eagerly():
        output = compat.as_str(output.numpy())
    else:
      output = compat.as_str(output)
    metrics.RecordCheckpointSize(
        api_label=_CHECKPOINT_V1, filesize=_get_checkpoint_size(output))
    return output
  @property
  def save_counter(self):
    self._maybe_create_save_counter()
    return self._save_counter
  def save(self, file_prefix, session=None):
    """Saves a training checkpoint and provides basic checkpoint management.
    The saved checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.save()` is
    called.
    `save` is a basic convenience wrapper around the `write` method,
    sequentially numbering checkpoints using `save_counter` and updating the
    metadata used by `tf.train.latest_checkpoint`. More advanced checkpoint
    management, for example garbage collection and custom numbering, may be
    provided by other utilities which also wrap `write`
    (`tf.train.CheckpointManager` for example).
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `Checkpoint.save_counter`.
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.
    Returns:
      The full path to the checkpoint.
    """
    graph_building = not context.executing_eagerly()
    if graph_building:
      if ops.inside_function():
        raise NotImplementedError(
            "Calling tf.train.Checkpoint.save() from a function is not "
            "supported, as save() modifies saving metadata in ways not "
            "supported by TensorFlow Operations. Consider using "
            "tf.train.Checkpoint.write(), a lower-level API which does not "
            "update metadata. tf.train.latest_checkpoint and related APIs will "
            "not see this checkpoint.")
      if session is None:
        session = get_session()
      if self._save_counter is None:
        session.run(self.save_counter.initializer)
    if not graph_building or self._save_assign_op is None:
      with ops.colocate_with(self.save_counter):
        assign_op = self.save_counter.assign_add(1, read_value=True)
      if graph_building:
        self._save_assign_op = data_structures.NoDependency(assign_op)
    if graph_building:
      checkpoint_number = session.run(self._save_assign_op)
    else:
      checkpoint_number = assign_op.numpy()
    file_path = self.write(
        "%s-%d" % (file_prefix, checkpoint_number), session=session)
    checkpoint_management.update_checkpoint_state_internal(
        save_dir=os.path.dirname(file_prefix),
        model_checkpoint_path=file_path,
        all_model_checkpoint_paths=[file_path],
        save_relative_paths=True)
    return file_path
  def restore(self, save_path):
    """Restore a training checkpoint.
    Restores this `Checkpoint` and any objects it depends on.
    When executing eagerly, either assigns values immediately if variables to
    restore have been created already, or defers restoration until the variables
    are created. Dependencies added after this call will be matched if they have
    a corresponding object in the checkpoint (the restore request will queue in
    any trackable object waiting for the expected dependency to be added).
    When graph building, restoration ops are added to the graph but not run
    immediately.
    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path)
    ```
    To ensure that loading is complete and no more deferred restorations will
    take place, you can use the `assert_consumed()` method of the status object
    returned by `restore`.
    The assert will raise an exception if any Python objects in the dependency
    graph were not found in the checkpoint, or if any checkpointed values do not
    have a matching Python object:
    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path).assert_consumed()
    ```
    When graph building, `assert_consumed()` indicates that all of the restore
    ops that will be created for this checkpoint have been created. They can be
    run via the `run_restore_ops()` method of the status object:
    ```python
    checkpoint.restore(path).assert_consumed().run_restore_ops()
    ```
    If the checkpoint has not been consumed completely, then the list of restore
    ops will grow as more objects are added to the dependency graph.
    To check that all variables in the Python object have restored values from
    checkpoint, use `assert_existing_objects_matched()`. This assertion is
    useful when called after the variables in your graph have been created.
    Name-based `tf.compat.v1.train.Saver` checkpoints can be loaded using this
    method. Names are used to match variables. No restore ops are created/run
    until `run_restore_ops()` or `initialize_or_restore()` are called on the
    returned status object when graph building, but there is restore-on-creation
    when executing eagerly. Re-encode name-based checkpoints using
    `tf.train.Checkpoint.save` as soon as possible.
    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If None (as when there is no latest
        checkpoint for `tf.train.latest_checkpoint` to return), returns an
        object which may run initializers for objects in the dependency graph.
        If the checkpoint was written by the name-based
        `tf.compat.v1.train.Saver`, names are used to match variables.
    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration and run initialization/restore ops.
      The returned status object has the following methods:
      * `assert_consumed()`:
          Raises an exception if any variables are unmatched: either
          checkpointed values which don't have a matching Python object or
          Python objects in the dependency graph with no values in the
          checkpoint. This method returns the status object, and so may be
          chained with `initialize_or_restore` or `run_restore_ops`.
      * `assert_existing_objects_matched()`:
          Raises an exception if any existing Python objects in the dependency
          graph are unmatched. Unlike `assert_consumed`, this assertion will
          pass if values in the checkpoint have no corresponding Python
          objects. For example a `tf.keras.Layer` object which has not yet been
          built, and so has not created any variables, will pass this assertion
          but will fail `assert_consumed`. Useful when loading part of a larger
          checkpoint into a new Python program, e.g. a training checkpoint with
          a `tf.compat.v1.train.Optimizer` was saved but only the state required
          for inference is being loaded. This method returns the status object,
          and so may be chained with `initialize_or_restore` or
          `run_restore_ops`.
      * `assert_nontrivial_match()`: Asserts that something aside from the root
          object was matched. This is a very weak assertion, but is useful for
          sanity checking in library code where objects may exist in the
          checkpoint which haven't been created in Python and some Python
          objects may not have a checkpointed value.
      * `expect_partial()`: Silence warnings about incomplete checkpoint
          restores. Warnings are otherwise printed for unused parts of the
          checkpoint file or object when the `Checkpoint` object is deleted
          (often at program shutdown).
      * `initialize_or_restore(session=None)`:
          When graph building, runs variable initializers if `save_path` is
          `None`, but otherwise runs restore operations. If no `session` is
          explicitly specified, the default session is used. No effect when
          executing eagerly (variables are initialized or restored eagerly).
      * `run_restore_ops(session=None)`:
          When graph building, runs restore operations. If no `session` is
          explicitly specified, the default session is used. No effect when
          executing eagerly (restore operations are run eagerly). May only be
          called when `save_path` is not `None`.
    """
    start_time = time.time()
    status = self._saver.restore(save_path=save_path)
    self._maybe_create_save_counter()
    if isinstance(status, NameBasedSaverStatus):
      status.add_to_optionally_restored(self.save_counter)
    metrics.AddCheckpointReadDuration(
        api_label=_CHECKPOINT_V1,
        microseconds=_get_duration_microseconds(start_time, time.time()))
    return status
@tf_export("train.Checkpoint", v1=[])
class Checkpoint(tracking.AutoTrackable):
  """Manages saving/restoring trackable values to disk.
  TensorFlow objects may contain trackable state, such as `tf.Variable`s,
  `tf.keras.optimizers.Optimizer` implementations, `tf.data.Dataset` iterators,
  `tf.keras.Layer` implementations, or  `tf.keras.Model` implementations.
  These are called **trackable objects**.
  A `Checkpoint` object can be constructed to save either a single or group of
  trackable objects to a checkpoint file. It maintains a `save_counter` for
  numbering checkpoints.
  Example:
  ```python
  model = tf.keras.Model(...)
  checkpoint = tf.train.Checkpoint(model)
  save_path = checkpoint.save('/tmp/training_checkpoints')
  checkpoint.restore(save_path)
  ```
  Example 2:
  ```python
  import tensorflow as tf
  import os
  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  for _ in range(num_training_steps):
  checkpoint.save(file_prefix=checkpoint_prefix)
  ```
  `Checkpoint.save()` and `Checkpoint.restore()` write and read object-based
  checkpoints, in contrast to TensorFlow 1.x's `tf.compat.v1.train.Saver` which
  writes and
  reads `variable.name` based checkpoints. Object-based checkpointing saves a
  graph of dependencies between Python objects (`Layer`s, `Optimizer`s,
  `Variable`s, etc.) with named edges, and this graph is used to match variables
  when restoring a checkpoint. It can be more robust to changes in the Python
  program, and helps to support restore-on-create for variables.
  `Checkpoint` objects have dependencies on the objects passed as keyword
  arguments to their constructors, and each dependency is given a name that is
  identical to the name of the keyword argument for which it was created.
  TensorFlow classes like `Layer`s and `Optimizer`s will automatically add
  dependencies on their own variables (e.g. "kernel" and "bias" for
  `tf.keras.layers.Dense`). Inheriting from `tf.keras.Model` makes managing
  dependencies easy in user-defined classes, since `Model` hooks into attribute
  assignment. For example:
  ```python
  class Regress(tf.keras.Model):
    def __init__(self):
      super(Regress, self).__init__()
      self.input_transform = tf.keras.layers.Dense(10)
    def call(self, inputs):
      x = self.input_transform(inputs)
  ```
  This `Model` has a dependency named "input_transform" on its `Dense` layer,
  which in turn depends on its variables. As a result, saving an instance of
  `Regress` using `tf.train.Checkpoint` will also save all the variables created
  by the `Dense` layer.
  When variables are assigned to multiple workers, each worker writes its own
  section of the checkpoint. These sections are then merged/re-indexed to behave
  as a single checkpoint. This avoids copying all variables to one worker, but
  does require that all workers see a common filesystem.
  This function differs slightly from the Keras Model `save_weights` function.
  `tf.keras.Model.save_weights` creates a checkpoint file with the name
  specified in `filepath`, while `tf.train.Checkpoint` numbers the checkpoints,
  using `filepath` as the prefix for the checkpoint file names. Aside from this,
  `model.save_weights()` and `tf.train.Checkpoint(model).save()` are equivalent.
  See the [guide to training
  checkpoints](https://www.tensorflow.org/guide/checkpoint) for
  details.
  Attributes:
    save_counter: Incremented when `save()` is called. Used to number
      checkpoints.
  """
  def __init__(self, root=None, **kwargs):
    """Creates a training checkpoint for a single or group of objects.
    Args:
      root: The root object to checkpoint. `root` may be a trackable object or
        `WeakRef` of a trackable object.
      **kwargs: Keyword arguments are set as attributes of this object, and are
        saved with the checkpoint. All `kwargs` must be trackable objects, or a
        nested structure of trackable objects (`list`, `dict`, or `tuple`).
    Raises:
      ValueError: If `root` or the objects in `kwargs` are not trackable. A
        `ValueError` is also raised if the `root` object tracks different
        objects from the ones listed in attributes in kwargs (e.g.
        `root.child = A` and `tf.train.Checkpoint(root, child=B)` are
        incompatible).
    """
    super(Checkpoint, self).__init__()
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      if _END_TIME_OF_LAST_WRITE is None:
        _END_TIME_OF_LAST_WRITE = time.time()
    attached_dependencies = None
    self._save_assign_op = None
    if root:
      trackable_root = root() if isinstance(root, weakref.ref) else root
      _assert_trackable(trackable_root, "root")
      attached_dependencies = []
      kwargs["root"] = root
      trackable_root._maybe_initialize_trackable()
      self._save_counter = data_structures.NoDependency(
          trackable_root._lookup_dependency("save_counter"))
    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      setattr(self, k, v)
      converted_v = getattr(self, k)
      if isinstance(converted_v, weakref.ref):
        converted_v = converted_v()
      _assert_trackable(converted_v, k)
      if root:
        child = trackable_root._lookup_dependency(k)
        if child is None:
          attached_dependencies.append(
              base.WeakTrackableReference(k, converted_v))
        elif child != converted_v:
          raise ValueError(
              f"Cannot create a Checkpoint with keyword argument {k} if "
              f"root.{k} already exists.")
    self._saver = saver_with_op_caching(root if root else self,
                                        attached_dependencies)
    self._attached_dependencies = data_structures.NoDependency(
        attached_dependencies)
  def _maybe_create_save_counter(self):
    if self._save_counter is None:
      with ops.device("/cpu:0"):
        self._save_counter = data_structures.NoDependency(
            add_variable(
                self,
                name="save_counter",
                initializer=0,
                dtype=dtypes.int64,
                trainable=False))
        if self._attached_dependencies is not None:
          self._attached_dependencies.append(
              base.TrackableReference("save_counter", self._save_counter))
          if isinstance(self.root, weakref.ref):
            root = self.root()
          else:
            root = self.root
          if restore:
            restore[0].restore(self._save_counter)
  def write(self, file_prefix, options=None):
    """Writes a training checkpoint.
    The checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.write()` is
    called.
    `write` does not number checkpoints, increment `save_counter`, or update the
    metadata used by `tf.train.latest_checkpoint`. It is primarily intended for
    use by higher level checkpoint management utilities. `save` provides a very
    basic implementation of these features.
    Checkpoints written with `write` must be read with `read`.
    Example usage:
    ```
    step = tf.Variable(0, name="step")
    checkpoint = tf.Checkpoint(step=step)
    checkpoint.write("/tmp/ckpt")
    checkpoint.read("/tmp/ckpt")
    options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint.write("/tmp/ckpt", options=options)
    checkpoint.read("/tmp/ckpt", options=options)
    ```
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """
    return self._write(file_prefix, options)
  def _write(self, file_prefix, options=None, update_ckpt_state=False):
    """Internal method that implements Checkpoint.write().
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      options: Optional `tf.train.CheckpointOptions` object.
      update_ckpt_state: Optional bool flag. Indiciate whether the internal
        checkpoint state needs to be updated.
    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """
    start_time = time.time()
    options = options or checkpoint_options.CheckpointOptions()
    output = self._saver.save(
        file_prefix=file_prefix,
        options=options,
        update_ckpt_state=update_ckpt_state)
    end_time = time.time()
    metrics.AddCheckpointWriteDuration(
        api_label=_CHECKPOINT_V2,
        microseconds=_get_duration_microseconds(start_time, end_time))
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      metrics.AddTrainingTimeSaved(
          api_label=_CHECKPOINT_V2,
          microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE,
                                                  end_time))
      _END_TIME_OF_LAST_WRITE = end_time
    output = _convert_file_name_tensor_to_string(output)
    if not options.experimental_enable_async_checkpoint:
      metrics.RecordCheckpointSize(
          api_label=_CHECKPOINT_V2, filesize=_get_checkpoint_size(output))
    return output
  @property
  def save_counter(self):
    self._maybe_create_save_counter()
    return self._save_counter
  def save(self, file_prefix, options=None):
    """Saves a training checkpoint and provides basic checkpoint management.
    The saved checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.save()` is
    called.
    `save` is a basic convenience wrapper around the `write` method,
    sequentially numbering checkpoints using `save_counter` and updating the
    metadata used by `tf.train.latest_checkpoint`. More advanced checkpoint
    management, for example garbage collection and custom numbering, may be
    provided by other utilities which also wrap `write` and `read`.
    (`tf.train.CheckpointManager` for example).
    ```
    step = tf.Variable(0, name="step")
    checkpoint = tf.train.Checkpoint(step=step)
    checkpoint.save("/tmp/ckpt")
    checkpoint.restore("/tmp/ckpt-1")
    options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint.save("/tmp/ckpt", options=options)
    checkpoint.restore("/tmp/ckpt-1", options=options)
    ```
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `Checkpoint.save_counter`.
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      The full path to the checkpoint.
    """
    options = options or checkpoint_options.CheckpointOptions()
    graph_building = not context.executing_eagerly()
    if graph_building:
      if options.experimental_enable_async_checkpoint:
        raise NotImplementedError(
            "Async checkpoint is not supported for non-eager mode. ")
      if ops.inside_function():
        raise NotImplementedError(
            "Calling tf.train.Checkpoint.save() from a function is not "
            "supported, as save() modifies saving metadata in ways not "
            "supported by TensorFlow Operations. Consider using "
            "tf.train.Checkpoint.write(), a lower-level API which does not "
            "update metadata. tf.train.latest_checkpoint and related APIs will "
            "not see this checkpoint.")
      session = get_session()
      if self._save_counter is None:
        session.run(self.save_counter.initializer)
    if not graph_building or self._save_assign_op is None:
      with ops.colocate_with(self.save_counter):
        assign_op = self.save_counter.assign_add(1, read_value=True)
      if graph_building:
        self._save_assign_op = data_structures.NoDependency(assign_op)
    if graph_building:
      checkpoint_number = session.run(self._save_assign_op)
    else:
      checkpoint_number = assign_op.numpy()
    file_path = self._write(
        "%s-%d" % (file_prefix, checkpoint_number),
        options=options,
        update_ckpt_state=True)
    if not options.experimental_enable_async_checkpoint:
      _update_checkpoint_state_internal(file_path)
    if not graph_building and not options.experimental_enable_async_checkpoint:
      context.async_wait()
    return file_path
  def read(self, save_path, options=None):
    """Reads a training checkpoint written with `write`.
    Reads this `Checkpoint` and any objects it depends on.
    This method is just like `restore()` but does not expect the `save_counter`
    variable in the checkpoint. It only restores the objects that the checkpoint
    already depends on.
    The method is primarily intended for use by higher level checkpoint
    management utilities that use `write()` instead of `save()` and have their
    own mechanisms to number and track checkpoints.
    Example usage:
    ```python
    ckpt = tf.train.Checkpoint(v=tf.Variable(1.))
    path = ckpt.write('/tmp/my_checkpoint')
    checkpoint.read(path).assert_consumed()
    options = tf.train.CheckpointOptions(
        experimental_io_device="/job:localhost")
    checkpoint.read(path, options=options)
    ```
    Args:
      save_path: The path to the checkpoint as returned by `write`.
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration.  See `restore` for details.
    """
    start_time = time.time()
    options = options or checkpoint_options.CheckpointOptions()
    result = self._saver.restore(save_path=save_path, options=options)
    metrics.AddCheckpointReadDuration(
        api_label=_CHECKPOINT_V2,
        microseconds=_get_duration_microseconds(start_time, time.time()))
    return result
  def restore(self, save_path, options=None):
    """Restores a training checkpoint.
    Restores this `Checkpoint` and any objects it depends on.
    This method is intended to be used to load checkpoints created by `save()`.
    For checkpoints created by `write()` use the `read()` method which does not
    expect the `save_counter` variable added by `save()`.
    `restore()` either assigns values immediately if variables to restore have
    been created already, or defers restoration until the variables are
    created. Dependencies added after this call will be matched if they have a
    corresponding object in the checkpoint (the restore request will queue in
    any trackable object waiting for the expected dependency to be added).
    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path)
    options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint.restore(path, options=options)
    ```
    To ensure that loading is complete and no more deferred restorations will
    take place, use the `assert_consumed()` method of the status object returned
    by `restore()`:
    ```python
    checkpoint.restore(path, options=options).assert_consumed()
    ```
    The assert will raise an error if any Python objects in the dependency graph
    were not found in the checkpoint, or if any checkpointed values do not have
    a matching Python object.
    Name-based `tf.compat.v1.train.Saver` checkpoints from TensorFlow 1.x can be
    loaded using this method. Names are used to match variables. Re-encode
    name-based checkpoints using `tf.train.Checkpoint.save` as soon as possible.
    **Loading from SavedModel checkpoints**
    To load values from a SavedModel, just pass the SavedModel directory
    to checkpoint.restore:
    ```python
    model = tf.keras.Model(...)
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(path).expect_partial()
    ```
    This example calls `expect_partial()` on the loaded status, since
    SavedModels saved from Keras often generates extra keys in the checkpoint.
    Otherwise, the program prints a lot of warnings about unused keys at exit
    time.
    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If the checkpoint was written by the
        name-based `tf.compat.v1.train.Saver`, names are used to match
        variables. This path may also be a SavedModel directory.
      options: Optional `tf.train.CheckpointOptions` object.
    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration.
      The returned status object has the following methods:
      * `assert_consumed()`:
          Raises an exception if any variables are unmatched: either
          checkpointed values which don't have a matching Python object or
          Python objects in the dependency graph with no values in the
          checkpoint. This method returns the status object, and so may be
          chained with other assertions.
      * `assert_existing_objects_matched()`:
          Raises an exception if any existing Python objects in the dependency
          graph are unmatched. Unlike `assert_consumed`, this assertion will
          pass if values in the checkpoint have no corresponding Python
          objects. For example a `tf.keras.Layer` object which has not yet been
          built, and so has not created any variables, will pass this assertion
          but fail `assert_consumed`. Useful when loading part of a larger
          checkpoint into a new Python program, e.g. a training checkpoint with
          a `tf.compat.v1.train.Optimizer` was saved but only the state required
          for
          inference is being loaded. This method returns the status object, and
          so may be chained with other assertions.
      * `assert_nontrivial_match()`: Asserts that something aside from the root
          object was matched. This is a very weak assertion, but is useful for
          sanity checking in library code where objects may exist in the
          checkpoint which haven't been created in Python and some Python
          objects may not have a checkpointed value.
      * `expect_partial()`: Silence warnings about incomplete checkpoint
          restores. Warnings are otherwise printed for unused parts of the
          checkpoint file or object when the `Checkpoint` object is deleted
          (often at program shutdown).
    Raises:
      NotFoundError: if the a checkpoint or SavedModel cannot be found at
        `save_path`.
    """
    orig_save_path = save_path
    if save_path is not None and gfile.IsDirectory(save_path) and (
        (gfile.Exists(utils_impl.get_saved_model_pb_path(save_path)) or
         gfile.Exists(utils_impl.get_saved_model_pbtxt_path(save_path)))):
      save_path = utils_impl.get_variables_path(save_path)
    try:
      status = self.read(save_path, options=options)
      if context.executing_eagerly():
    except errors_impl.NotFoundError as e:
      raise errors_impl.NotFoundError(
          None, None,
          f"Error when restoring from checkpoint or SavedModel at "
          f"{orig_save_path}: {e.message}"
          f"\nPlease double-check that the path is correct. You may be missing "
          "the checkpoint suffix (e.g. the '-1' in 'path/to/ckpt-1').")
    self._maybe_create_save_counter()
    if isinstance(status, NameBasedSaverStatus):
      status.add_to_optionally_restored(self.save_counter)
    return status
