
import contextlib
import functools
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_resource_variable_ops import *
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
acd.register_read_only_resource_op("ReadVariableOp")
acd.register_read_only_resource_op("VariableShape")
acd.register_read_only_resource_op("ResourceGather")
acd.register_read_only_resource_op("ResourceGatherNd")
acd.register_read_only_resource_op("_ReadVariablesOp")
get_resource_handle_data = handle_data_util.get_resource_handle_data
def get_eager_safe_handle_data(handle):
  assert isinstance(handle, ops.Tensor)
  if isinstance(handle, ops.EagerTensor):
  else:
    return get_resource_handle_data(handle)
def _set_handle_shapes_and_types(tensor, handle_data, graph_mode):
  if not graph_mode:
    return
  shapes, types = zip(*[(pair.shape, pair.dtype)
                        for pair in handle_data.shape_and_type])
  ranks = [len(s.dim) if not s.unknown_rank else -1 for s in shapes]
  shapes = [
      if not s.unknown_rank else None for s in shapes
  ]
  pywrap_tf_session.TF_GraphSetOutputHandleShapesAndTypes_wrapper(
      shapes,
      ranks,
      types)
def _combine_handle_data(handle, initial_value):
  """Concats HandleData from tensors `handle` and `initial_value`.
  Args:
    handle: A `Tensor` of dtype `resource`.
    initial_value: A `Tensor`.
  Returns:
    A `CppShapeInferenceResult.HandleData`.  If `initial_value` has dtype
    `variant`, the `HandleData` contains the concatenation of the shape_and_type
    from both `handle` and `initial_value`.
  Raises:
    RuntimeError: If handle, which was returned by VarHandleOp, either has
      no handle data, or its len(handle_data.shape_and_type) != 1.
  """
  assert handle.dtype == dtypes.resource
  variable_handle_data = get_eager_safe_handle_data(handle)
  if initial_value.dtype != dtypes.variant:
    return variable_handle_data
  extra_handle_data = get_eager_safe_handle_data(initial_value)
  if extra_handle_data is not None and extra_handle_data.is_set:
    if (variable_handle_data is None or not variable_handle_data.is_set or
        len(variable_handle_data.shape_and_type) != 1):
      raise RuntimeError(
          "Expected VarHandleOp to return a length==1 shape_and_type, "
          f"but saw: '{variable_handle_data}'")
    variable_handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)
  return variable_handle_data
def _variable_handle_from_shape_and_dtype(shape,
                                          dtype,
                                          shared_name,
                                          name,
                                          graph_mode,
                                          initial_value=None):
  if container is None:
    container = ""
  shape = tensor_shape.as_shape(shape)
  dtype = dtypes.as_dtype(dtype)
  if not graph_mode:
    if shared_name is not None:
          "Using an explicit shared_name is not allowed when executing eagerly."
      )
    shared_name = context.anonymous_name()
  handle = gen_resource_variable_ops.var_handle_op(
      shape=shape,
      dtype=dtype,
      shared_name=shared_name,
      name=name,
      container=container)
  if initial_value is None:
    initial_value = handle
  if graph_mode:
    full_handle_data = _combine_handle_data(handle, initial_value)
    _set_handle_shapes_and_types(handle, full_handle_data, graph_mode)
    return handle
  else:
    handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
    handle_data.is_set = True
    handle_data.shape_and_type.append(
        cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
            shape=shape.as_proto(), dtype=dtype.as_datatype_enum))
    if initial_value is not None and initial_value.dtype == dtypes.variant:
      extra_handle_data = get_eager_safe_handle_data(initial_value)
      if extra_handle_data is not None and extra_handle_data.is_set:
        if (not handle_data.is_set or len(handle_data.shape_and_type) != 1):
          raise RuntimeError(
              "Expected VarHandleOp to return a length==1 shape_and_type, "
              f"but saw: '{handle_data}'")
        handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)
    _set_handle_shapes_and_types(handle, handle_data, graph_mode)
    return handle
def eager_safe_variable_handle(initial_value, shape, shared_name, name,
                               graph_mode):
  """Creates a variable handle with information to do shape inference.
  The dtype is read from `initial_value` and stored in the returned
  resource tensor's handle data.
  If `initial_value.dtype == tf.variant`, we additionally extract the handle
  data (if any) from `initial_value` and append it to the `handle_data`.
  In this case, the returned tensor's handle data is in the form
  ```
  is_set: true
  shape_and_type {
    shape {
      // initial_value.shape
    }
    dtype: DT_VARIANT
  }
  shape_and_type {
    // handle_data(initial_value).shape_and_type[0]
  }
  shape_and_type {
    // handle_data(initial_value).shape_and_type[1]
  }
  ...
  ```
  Ops that read from this tensor, such as `ReadVariableOp` and
  `AssignVariableOp`, know that `handle_data(handle).shape_and_type[1:]`
  correspond to the handle data of the variant(s) stored in the Variable.
  Args:
    initial_value: A `Tensor`.
    shape: The shape of the handle data. Can be `TensorShape(None)` (i.e.
      unknown shape).
    shared_name: A string.
    name: A string.
    graph_mode: A python bool.
  Returns:
    The handle, a `Tensor` of type `resource`.
  """
  dtype = initial_value.dtype.base_dtype
  return _variable_handle_from_shape_and_dtype(shape, dtype, shared_name, name,
                                               graph_mode, initial_value)
@contextlib.contextmanager
def _handle_graph(handle):
  if (context.executing_eagerly() or isinstance(handle, ops.EagerTensor) or
      ops.has_default_graph()):
    yield
  else:
    with handle.graph.as_default():
      yield
class EagerResourceDeleter:
  __slots__ = ["_handle", "_handle_device", "_context"]
  def __init__(self, handle, handle_device):
    if not isinstance(handle, ops.Tensor):
      raise ValueError(
          (f"Passed handle={handle} to EagerResourceDeleter. Was expecting "
           f"the handle to be a `tf.Tensor`."))
    self._handle = handle
    self._handle_device = handle_device
    self._context = context.context()
  def __del__(self):
    try:
      if isinstance(self._handle, ops.EagerTensor) and self._handle.is_packed:
        return
      with context.eager_mode():
        with ops.device(self._handle_device):
          gen_resource_variable_ops.destroy_resource_op(
              self._handle, ignore_lookup_error=True)
    except TypeError:
    except AttributeError:
def shape_safe_assign_variable_handle(handle, shape, value, name=None):
  with _handle_graph(handle):
    value_tensor = ops.convert_to_tensor(value)
  shape.assert_is_compatible_with(value_tensor.shape)
  return gen_resource_variable_ops.assign_variable_op(
      handle, value_tensor, name=name)
def _maybe_set_handle_data(dtype, handle, tensor):
  if dtype == dtypes.variant:
    handle_data = get_eager_safe_handle_data(handle)
    if handle_data.is_set and len(handle_data.shape_and_type) > 1:
          cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData(
              is_set=True, shape_and_type=handle_data.shape_and_type[1:]))
def variable_accessed(variable):
  if hasattr(ops.get_default_graph(), "watch_variable"):
    ops.get_default_graph().watch_variable(variable)
  if variable.trainable:
    tape.variable_accessed(variable)
class BaseResourceVariable(variables.VariableV1, core.Tensor):
      self,
      trainable=None,
      shape=None,
      dtype=None,
      handle=None,
      constraint=None,
      synchronization=None,
      aggregation=None,
      distribute_strategy=None,
      name=None,
      unique_id=None,
      handle_name=None,
      graph_element=None,
      initial_value=None,
      initializer_op=None,
      is_initialized_op=None,
      cached_value=None,
      save_slice_info=None,
      caching_device=None,
      in_graph_mode=None,
      validate_shape=True,
      **unused_kwargs):
    """Creates a variable from a handle.
    Args:
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      shape: The variable's shape. This shape can be set to tf.TensorShape(None)
        in order to assign values of different shapes to this variable.
        Otherwise (i.e. if the shape is fully determined), it will trigger run
        time checks to ensure that each assignment is of the same shape.
      dtype: The variable's dtype.
      handle: The variable's handle
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: The distribution strategy this variable was created
        under.
      name: The name for this variable.
      unique_id: Internal. Unique ID for this variable's handle.
      handle_name: The name for the variable's handle.
      graph_element: Optional, required only in session.run-mode. Pre-created
        tensor which reads this variable's value.
      initial_value: Optional. Variable's initial value.
      initializer_op: Operation which assigns the variable's initial value.
      is_initialized_op: Pre-created operation to check whether this variable is
        initialized.
      cached_value: Pre-created operation to read this variable in a specific
        device.
      save_slice_info: Metadata for variable partitioning.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      in_graph_mode: whether we are executing in TF1 graph mode. If None, will
        detect within the function. This is to avoid repeated init_scope()
        conetxt entrances which can add up.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
    """
    if in_graph_mode is None:
      with ops.init_scope():
        self._in_graph_mode = not context.executing_eagerly()
    else:
      self._in_graph_mode = in_graph_mode
    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            synchronization, aggregation, trainable, name))
    self._trainable = trainable
    self._synchronization = synchronization
    self._aggregation = aggregation
    self._save_slice_info = save_slice_info
    self._initial_value = initial_value
    self._initializer_op = initializer_op
    self._is_initialized_op = is_initialized_op
    self._graph_element = graph_element
    self._caching_device = caching_device
    self._cached_value = cached_value
    self._distribute_strategy = distribute_strategy
    self._shape = tensor_shape.as_shape(shape)
    self._dtype = dtypes.as_dtype(dtype)
    self._handle = handle
    self._unique_id = unique_id
    if handle_name is None:
      self._handle_name = "Variable:0"
    else:
      self._handle_name = handle_name + ":0"
    self._constraint = constraint
    self._cached_shape_as_list = None
    self._validate_shape = validate_shape
  def __repr__(self):
    if context.executing_eagerly() and not self._in_graph_mode:
      try:
        with ops.device(self.device):
          value_text = ops.value_text(self.read_value(), is_repr=True)
        value_text = "numpy=<unavailable>"
      return "<tf.Variable '%s' shape=%s dtype=%s, %s>" % (
          self.name, self.get_shape(), self.dtype.name, value_text)
    else:
      return "<tf.Variable '%s' shape=%s dtype=%s>" % (
          self.name, self.get_shape(), self.dtype.name)
  def __tf_tracing_type__(self, signature_context):
    return signature_context.make_reference_type(
  @contextlib.contextmanager
  def _assign_dependencies(self):
    if self._cached_value is not None:
      with ops.control_dependencies([self._cached_value]):
        yield
    else:
      yield
  def __array__(self, dtype=None):
    """Allows direct conversion to a numpy array.
    >>> np.array(tf.Variable([1.0]))
    array([1.], dtype=float32)
    Returns:
      The variable value as a numpy array.
    """
    return np.asarray(self.numpy(), dtype=dtype)
  def __nonzero__(self):
    return self.__bool__()
  def __bool__(self):
    return bool(self.read_value())
  def __copy__(self):
    return self
  def __deepcopy__(self, memo):
    if not context.executing_eagerly():
      raise NotImplementedError(
          "__deepcopy__() is only available when eager execution is enabled.")
    copied_variable = ResourceVariable(
        initial_value=self.read_value(),
        trainable=self._trainable,
        constraint=self._constraint,
        dtype=self._dtype,
        name=self._shared_name,
        distribute_strategy=self._distribute_strategy,
        synchronization=self.synchronization,
        aggregation=self.aggregation)
    memo[self._unique_id] = copied_variable
    return copied_variable
  @property
  def dtype(self):
    return self._dtype
  @property
  def device(self):
    return self.handle.device
  @property
  def graph(self):
    return self.handle.graph
  @property
  def name(self):
    return self._handle_name
  @property
  def shape(self):
    return self._shape
  def set_shape(self, shape):
    self._shape = self._shape.merge_with(shape)
  def _shape_as_list(self):
    if self.shape.ndims is None:
      return None
    return [dim.value for dim in self.shape.dims]
  def _shape_tuple(self):
    shape = self._shape_as_list()
    if shape is None:
      return None
    return tuple(shape)
  @property
  def create(self):
    if not self._in_graph_mode:
      raise RuntimeError("This operation is not supported "
                         "when eager execution is enabled.")
    return self._initializer_op
  @property
  def handle(self):
    return self._handle
  def value(self):
    if self._cached_value is not None:
      return self._cached_value
    with ops.colocate_with(None, ignore_existing=True):
      return self._read_variable_op()
  def _as_graph_element(self):
    return self._graph_element
  @property
  def initializer(self):
    return self._initializer_op
  @property
  def initial_value(self):
    if context.executing_eagerly():
      raise RuntimeError("This property is not supported "
                         "when eager execution is enabled.")
    return self._initial_value
  @property
  def constraint(self):
    return self._constraint
  @property
  def op(self):
    return self.handle.op
  @property
  def trainable(self):
    return self._trainable
  @property
  def synchronization(self):
    return self._synchronization
  @property
  def aggregation(self):
    return self._aggregation
  def eval(self, session=None):
    if context.executing_eagerly():
      raise RuntimeError("This operation is not supported "
                         "when eager execution is enabled.")
    return self._graph_element.eval(session=session)
  def numpy(self):
    if context.executing_eagerly():
      return self.read_value().numpy()
    raise NotImplementedError(
        "numpy() is only available when eager execution is enabled.")
  @deprecated(None, "Prefer Dataset.range instead.")
  def count_up_to(self, limit):
    """Increments this variable until it reaches `limit`.
    When that Op is run it tries to increment the variable by `1`. If
    incrementing the variable would bring it above `limit` then the Op raises
    the exception `OutOfRangeError`.
    If no error is raised, the Op outputs the value of the variable before
    the increment.
    This is essentially a shortcut for `count_up_to(self, limit)`.
    Args:
      limit: value at which incrementing the variable raises an error.
    Returns:
      A `Tensor` that will hold the variable value before the increment. If no
      other Op modifies this variable, the values produced will all be
      distinct.
    """
    return gen_state_ops.resource_count_up_to(
        self.handle, limit=limit, T=self.dtype)
  def _map_resources(self, save_options):
    new_variable = None
      with ops.device(self.device):
        new_variable = copy_to_graph_uninitialized(self)
    else:
      new_variable = copy_to_graph_uninitialized(self)
    obj_map = {self: new_variable}
    resource_map = {self.handle: new_variable.handle}
    return obj_map, resource_map
  def _read_variable_op(self):
    variable_accessed(self)
    def read_and_set_handle():
      result = gen_resource_variable_ops.read_variable_op(
          self.handle, self._dtype)
      _maybe_set_handle_data(self._dtype, self.handle, result)
      return result
    if getattr(self, "_caching_device", None) is not None:
      with ops.colocate_with(None, ignore_existing=True):
        with ops.device(self._caching_device):
          result = read_and_set_handle()
    else:
      result = read_and_set_handle()
    if not context.executing_eagerly():
      tape.record_operation(
          "ReadVariableOp", [result], [self.handle],
          backward_function=lambda x: [x],
          forward_function=lambda x: [x])
    return result
  def read_value(self):
    with ops.name_scope("Read"):
      value = self._read_variable_op()
    return array_ops.identity(value)
  def sparse_read(self, indices, name=None):
    with ops.name_scope("Gather" if name is None else name) as name:
      variable_accessed(self)
      value = gen_resource_variable_ops.resource_gather(
          self.handle, indices, dtype=self._dtype, name=name)
      if self._dtype == dtypes.variant:
        handle_data = get_eager_safe_handle_data(self.handle)
        if handle_data.is_set and len(handle_data.shape_and_type) > 1:
              cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData(
                  is_set=True, shape_and_type=handle_data.shape_and_type[1:]))
    return array_ops.identity(value)
  def gather_nd(self, indices, name=None):
    with ops.name_scope("GatherNd" if name is None else name) as name:
      if self.trainable:
        variable_accessed(self)
      value = gen_resource_variable_ops.resource_gather_nd(
          self.handle, indices, dtype=self._dtype, name=name)
    return array_ops.identity(value)
  def to_proto(self, export_scope=None):
    if context.executing_eagerly():
      raise RuntimeError("This operation is not supported "
                         "when eager execution is enabled.")
    if export_scope is None or self.handle.name.startswith(export_scope):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(self.handle.name,
                                                   export_scope)
      if self._initial_value is not None:
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      var_def.initializer_name = ops.strip_name_scope(self.initializer.name,
                                                      export_scope)
      if self._cached_value is not None:
        var_def.snapshot_name = ops.strip_name_scope(self._cached_value.name,
                                                     export_scope)
      else:
        var_def.snapshot_name = ops.strip_name_scope(self._graph_element.name,
                                                     export_scope)
      var_def.is_resource = True
      var_def.trainable = self.trainable
      var_def.synchronization = self.synchronization.value
      var_def.aggregation = self.aggregation.value
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(
            self._save_slice_info.to_proto(export_scope=export_scope))
      return var_def
    else:
      return None
  @staticmethod
  def from_proto(variable_def, import_scope=None):
    if context.executing_eagerly():
      raise RuntimeError("This operation is not supported "
                         "when eager execution is enabled.")
    return ResourceVariable(
        variable_def=variable_def, import_scope=import_scope)
  __array_priority__ = 100
  def is_initialized(self, name=None):
    """Checks whether a resource variable has been initialized.
    Outputs boolean scalar indicating whether the tensor has been initialized.
    Args:
      name: A name for the operation (optional).
    Returns:
      A `Tensor` of type `bool`.
    """
    return gen_resource_variable_ops.var_is_initialized_op(self.handle, name)
  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    with _handle_graph(self.handle), self._assign_dependencies():
      assign_sub_op = gen_resource_variable_ops.assign_sub_variable_op(
          self.handle,
          ops.convert_to_tensor(delta, dtype=self.dtype),
          name=name)
    if read_value:
      return self._lazy_read(assign_sub_op)
    return assign_sub_op
  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    with _handle_graph(self.handle), self._assign_dependencies():
      assign_add_op = gen_resource_variable_ops.assign_add_variable_op(
          self.handle,
          ops.convert_to_tensor(delta, dtype=self.dtype),
          name=name)
    if read_value:
      return self._lazy_read(assign_add_op)
    return assign_add_op
  def _lazy_read(self, op):
    variable_accessed(self)
    return _UnreadVariable(
        handle=self.handle,
        dtype=self.dtype,
        shape=self._shape,
        in_graph_mode=self._in_graph_mode,
        parent_op=op,
        unique_id=self._unique_id)
  def assign(self, value, use_locking=None, name=None, read_value=True):
    with _handle_graph(self.handle):
      value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
      if not self._shape.is_compatible_with(value_tensor.shape):
        if self.name is None:
          tensor_name = ""
        else:
          tensor_name = " " + str(self.name)
        raise ValueError(
            (f"Cannot assign value to variable '{tensor_name}': Shape mismatch."
             f"The variable shape {self._shape}, and the "
             f"assigned value shape {value_tensor.shape} are incompatible."))
      kwargs = {}
      if forward_compat.forward_compatible(2022, 3, 23):
        validate_shape = self._validate_shape and self._shape.is_fully_defined()
        kwargs["validate_shape"] = validate_shape
      assign_op = gen_resource_variable_ops.assign_variable_op(
          self.handle, value_tensor, name=name, **kwargs)
      if read_value:
        return self._lazy_read(assign_op)
    return assign_op
  def __reduce__(self):
    return functools.partial(
        ResourceVariable,
        initial_value=self.numpy(),
        trainable=self.trainable,
        name=self._shared_name,
        dtype=self.dtype,
        constraint=self.constraint,
        distribute_strategy=self._distribute_strategy), ()
  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_sub(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))
  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_add(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))
  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_max(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))
  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_min(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))
  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_mul(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))
  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_div(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))
  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_update(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))
  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        state_ops.batch_scatter_update(
            self,
            sparse_delta.indices,
            sparse_delta.values,
            use_locking=use_locking,
            name=name))
  def scatter_nd_sub(self, indices, updates, name=None):
    """Applies sparse subtraction to individual values or slices in a Variable.
    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.
    `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```
    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:
    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_sub(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```
    The resulting update to ref would look like this:
        [1, -9, 3, -6, -6, 6, 7, -4]
    See `tf.scatter_nd` for more details about how to make updates to
    slices.
    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_sub(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))
  def scatter_nd_add(self, indices, updates, name=None):
    """Applies sparse addition to individual values or slices in a Variable.
    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.
    `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```
    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:
    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        add = ref.scatter_nd_add(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(add)
    ```
    The resulting update to ref would look like this:
        [1, 13, 3, 14, 14, 6, 7, 20]
    See `tf.scatter_nd` for more details about how to make updates to
    slices.
    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_add(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))
  def scatter_nd_update(self, indices, updates, name=None):
    """Applies sparse assignment to individual values or slices in a Variable.
    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.
    `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```
    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:
    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_update(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```
    The resulting update to ref would look like this:
        [1, 11, 3, 10, 9, 6, 7, 12]
    See `tf.scatter_nd` for more details about how to make updates to
    slices.
    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_update(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))
  def scatter_nd_max(self, indices, updates, name=None):
    """Updates this variable with the max of `tf.IndexedSlices` and itself.
    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.
    `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```
    See `tf.scatter_nd` for more details about how to make updates to
    slices.
    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_max(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))
  def scatter_nd_min(self, indices, updates, name=None):
    """Updates this variable with the min of `tf.IndexedSlices` and itself.
    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.
    `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```
    See `tf.scatter_nd` for more details about how to make updates to
    slices.
    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_min(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))
  def _write_object_proto(self, proto, options):
    write_object_proto_for_resource_variable(self, proto, options)
  def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask,
                            end_mask, ellipsis_mask, new_axis_mask,
                            shrink_axis_mask):
    with _handle_graph(self.handle), self._assign_dependencies():
      return self._lazy_read(
          gen_array_ops.resource_strided_slice_assign(
              ref=self.handle,
              begin=begin,
              end=end,
              strides=strides,
              value=ops.convert_to_tensor(value, dtype=self.dtype),
              name=name,
              begin_mask=begin_mask,
              end_mask=end_mask,
              ellipsis_mask=ellipsis_mask,
              new_axis_mask=new_axis_mask,
              shrink_axis_mask=shrink_axis_mask))
  def __complex__(self):
    return complex(self.value().numpy())
  def __int__(self):
    return int(self.value().numpy())
  def __long__(self):
    return long(self.value().numpy())
  def __float__(self):
    return float(self.value().numpy())
  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    del name
    if dtype is not None and not dtype.is_compatible_with(self.dtype):
      raise ValueError(
          f"Incompatible type conversion requested to type {dtype.name} for "
          f"`tf.Variable of type {self.dtype.name}. (Variable: {self})")
    if as_ref:
      return self.read_value().op.inputs[0]
    else:
      return self.value()
  def __iadd__(self, unused_other):
    raise RuntimeError("`variable += value` with `tf.Variable`s is not "
                       "supported. Use `variable.assign_add(value)` to modify "
                       "the variable, or `out = variable + value` if you "
                       "need to get a new output Tensor.")
  def __isub__(self, unused_other):
    raise RuntimeError("`variable -= value` with `tf.Variable`s is not "
                       "supported. Use `variable.assign_sub(value)` to modify "
                       "the variable, or `out = variable * value` if you "
                       "need to get a new output Tensor.")
  def __imul__(self, unused_other):
    raise RuntimeError("`var *= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var * value)` to modify "
                       "the variable, or `out = var * value` if you "
                       "need to get a new output Tensor.")
  def __idiv__(self, unused_other):
    raise RuntimeError("`var /= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var / value)` to modify "
                       "the variable, or `out = var / value` if you "
                       "need to get a new output Tensor.")
  def __itruediv__(self, unused_other):
    raise RuntimeError("`var /= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var / value)` to modify "
                       "the variable, or `out = var / value` if you "
                       "need to get a new output Tensor.")
  def __irealdiv__(self, unused_other):
    raise RuntimeError("`var /= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var / value)` to modify "
                       "the variable, or `out = var / value` if you "
                       "need to get a new output Tensor.")
  def __ipow__(self, unused_other):
    raise RuntimeError("`var **= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var ** value)` to modify "
                       "the variable, or `out = var ** value` if you "
                       "need to get a new output Tensor.")
class ResourceVariable(BaseResourceVariable):
  """Variable based on resource handles.
  See the [Variables How To](https://tensorflow.org/guide/variables)
  for a high level overview.
  A `ResourceVariable` allows you to maintain state across subsequent calls to
  session.run.
  The `ResourceVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.
  Just like any `Tensor`, variables created with
  `tf.Variable(use_resource=True)` can be used as inputs for other Ops in the
  graph. Additionally, all the operators overloaded for the `Tensor` class are
  carried over to variables, so you can also add nodes to the graph by just
  doing arithmetic on variables.
  Unlike ref-based variable, a ResourceVariable has well-defined semantics. Each
  usage of a ResourceVariable in a TensorFlow graph adds a read_value operation
  to the graph. The Tensors returned by a read_value operation are guaranteed to
  see all modifications to the value of the variable which happen in any
  operation on which the read_value depends on (either directly, indirectly, or
  via a control dependency) and guaranteed to not see any modification to the
  value of the variable from operations that depend on the read_value operation.
  Updates from operations that have no dependency relationship to the read_value
  operation might or might not be visible to read_value.
  For example, if there is more than one assignment to a ResourceVariable in
  a single session.run call there is a well-defined value for each operation
  which uses the variable's value if the assignments and the read are connected
  by edges in the graph. Consider the following example, in which two writes
  can cause tf.Variable and tf.ResourceVariable to behave differently:
  ```python
  a = tf.Variable(1.0, use_resource=True)
  a.initializer.run()
  assign = a.assign(2.0)
  with tf.control_dependencies([assign]):
    b = a.read_value()
  with tf.control_dependencies([b]):
    other_assign = a.assign(3.0)
  with tf.control_dependencies([other_assign]):
    tf.compat.v1.Print(b, [b]).eval()
  ```
  """
  def __init__(
      initial_value=None,
      trainable=None,
      collections=None,
      caching_device=None,
      name=None,
      dtype=None,
      variable_def=None,
      import_scope=None,
      constraint=None,
      distribute_strategy=None,
      synchronization=None,
      aggregation=None,
      shape=None):
    """Creates a variable.
    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. Can also be a callable with
        no argument that returns the initial value when called. (Note that
        initializer functions from init_ops.py must first be bound to a shape
        before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `ResourceVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        ResourceVariable. Only used when `variable_def` is provided.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      distribute_strategy: The tf.distribute.Strategy this variable is being
        created inside of.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is `None`, which signifies that this `Variable` will not be added to any
    collections.
    @end_compatibility
    """
    if variable_def:
      if initial_value is not None:
        raise ValueError(f"The variable_def and initial_value args to "
                         f"`tf.Variable` are mutually exclusive, but got both: "
                         f"variable_def={variable_def},\n"
                         f"initial_value={initial_value}")
      if context.executing_eagerly():
        raise ValueError(f"Creating a `tf.Variable` with a `variable_def` arg "
                         f"is not supported when eager execution is enabled. "
                         f"Got: variable_def={variable_def}")
      self._init_from_proto(variable_def, import_scope=import_scope,
                            validate_shape=validate_shape)
    else:
      self._init_from_args(
          initial_value=initial_value,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          name=name,
          dtype=dtype,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation,
          shape=shape,
          distribute_strategy=distribute_strategy,
          validate_shape=validate_shape,
      )
  def _init_from_args(self,
                      initial_value=None,
                      trainable=None,
                      collections=None,
                      caching_device=None,
                      name=None,
                      dtype=None,
                      constraint=None,
                      synchronization=None,
                      aggregation=None,
                      distribute_strategy=None,
                      shape=None,
                      validate_shape=True,
                      ):
    """Creates a variable.
    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound to
        a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: DistributionStrategy under which this variable was
        created.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the `GLOBAL_VARIABLES` or
    `TRAINABLE_VARIABLES` collections, and the `collections` argument is
    ignored.
    @end_compatibility
    """
    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            synchronization, aggregation, trainable, name))
    if initial_value is None:
      raise ValueError("The `initial_value` arg to `tf.Variable` must "
                       "be specified except when you are not providing a "
                       "`variable_def`. You provided neither.")
    init_from_fn = callable(initial_value)
    if isinstance(initial_value, ops.Tensor) and hasattr(
        initial_value, "graph") and initial_value.graph.building_function:
      raise ValueError(f"Argument `initial_value` ({initial_value}) could not "
                       "be lifted out of a `tf.function`. "
                       "(Tried to create variable with name='{name}'). "
                       "To avoid this error, when constructing `tf.Variable`s "
                       "inside of `tf.function` you can create the "
                       "`initial_value` tensor in a "
                       "`tf.init_scope` or pass a callable `initial_value` "
                       "(e.g., `tf.Variable(lambda : "
                       "tf.truncated_normal([10, 40]))`). "
                       "Please file a feature request if this "
                       "restriction inconveniences you.")
    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          f"collections argument to Variable constructor must be a list, "
          f"tuple, or set. Got {collections} of type {type(collections)}")
    if constraint is not None and not callable(constraint):
      raise ValueError(f"Argument `constraint` must be None or a callable. "
                       f"a callable. Got a {type(constraint)}:  {constraint}")
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
      with ops.name_scope(
          name,
          "Variable", [] if init_from_fn else [initial_value],
          skip_on_eager=False) as name:
        handle_name = ops.name_from_scope_name(name)
        if self._in_graph_mode:
          shared_name = handle_name
          unique_id = shared_name
        else:
          unique_id = "%s_%d" % (handle_name, ops.uid())
        device_context_manager = (
            ops.device if self._in_graph_mode else ops.NullContextmanager)
        attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                s=[compat.as_bytes("loc:@%s" % handle_name)]))
        with ops.get_default_graph()._attr_scope({"_class": attr}):
          with ops.name_scope("Initializer"), device_context_manager(None):
            if init_from_fn:
              initial_value = initial_value()
            if isinstance(initial_value, trackable.CheckpointInitialValue):
              self._maybe_initialize_trackable()
              self._update_uid = initial_value.checkpoint_position.restore_uid
              initial_value = initial_value.wrapped_value
            initial_value = ops.convert_to_tensor(initial_value,
                                                  name="initial_value",
                                                  dtype=dtype)
          if shape is not None:
            if not initial_value.shape.is_compatible_with(shape):
              raise ValueError(
                  f"In this `tf.Variable` creation, the initial value's shape "
                  f"({initial_value.shape}) is not compatible with "
                  f"the explicitly supplied `shape` argument ({shape}).")
          else:
            shape = initial_value.shape
          handle = eager_safe_variable_handle(
              initial_value=initial_value,
              shape=shape,
              shared_name=shared_name,
              name=name,
              graph_mode=self._in_graph_mode)
          handle._parent_trackable = weakref.ref(self)
        if (self._in_graph_mode and initial_value is not None and
            initial_value.op._get_control_flow_context() is not None):
          raise ValueError(
              f"The `initial_value` passed to `tf.Variable` {name} is from "
              f"inside a control-flow  construct, such as a loop or "
              f"conditional. When creating a "
              f"`tf.Variable` inside a loop or conditional, use a lambda as "
              f"the `initial_value`. Got: initial_value=({initial_value})")
        dtype = initial_value.dtype.base_dtype
        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            is_initialized_op = (
                gen_resource_variable_ops.var_is_initialized_op(handle))
          if initial_value is not None:
            with ops.name_scope("Assign") as n, \
                 ops.colocate_with(None, ignore_existing=True), \
                 ops.device(handle.device):
              initializer_op = (
                  gen_resource_variable_ops.assign_variable_op(
                      handle,
                      variables._try_guard_against_uninitialized_dependencies(
                          name, initial_value),
                      name=n))
          with ops.name_scope("Read"):
            with ops.device(handle.device):
              value = gen_resource_variable_ops.read_variable_op(handle, dtype)
              _maybe_set_handle_data(dtype, handle, value)
            graph_element = value
            if caching_device is not None:
              with ops.colocate_with(None, ignore_existing=True):
                with ops.device(caching_device):
                  cached_value = array_ops.identity(value)
            else:
              cached_value = None
        else:
          gen_resource_variable_ops.assign_variable_op(handle, initial_value)
          is_initialized_op = None
          initializer_op = None
          graph_element = None
          if caching_device:
            with ops.device(caching_device):
              cached_value = gen_resource_variable_ops.read_variable_op(
                  handle, dtype)
              _maybe_set_handle_data(dtype, handle, cached_value)
          else:
            cached_value = None
        if cached_value is not None:
        if not context.executing_eagerly():
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)
      initial_value = initial_value if self._in_graph_mode else None
      super(ResourceVariable, self).__init__(
          trainable=trainable,
          shape=shape,
          dtype=dtype,
          handle=handle,
          synchronization=synchronization,
          constraint=constraint,
          aggregation=aggregation,
          distribute_strategy=distribute_strategy,
          name=name,
          unique_id=unique_id,
          handle_name=handle_name,
          graph_element=graph_element,
          initial_value=initial_value,
          initializer_op=initializer_op,
          is_initialized_op=is_initialized_op,
          cached_value=cached_value,
          caching_device=caching_device,
          validate_shape=validate_shape,
      )
  def _init_from_proto(self, variable_def, import_scope=None,
                       validate_shape=True):
    assert not context.executing_eagerly()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError(f"The `variable_def` you passed to `tf.Variable` is "
                       f"Trying to restore a TF 1.x Reference Variable "
                       f"as a TF 2.x ResourceVariable. This is unsupported. "
                       f"Got variable_def={variable_def}")
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.variable_name, import_scope=import_scope))
    self._shape = tensor_shape.TensorShape(self._handle.op.get_attr("shape"))
    self._handle_name = self._handle.name
    self._unique_id = self._handle_name
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.initializer_name, import_scope=import_scope))
    if (hasattr(variable_def, "initial_value_name") and
        variable_def.initial_value_name):
      self._initial_value = g.as_graph_element(
          ops.prepend_name_scope(
              variable_def.initial_value_name, import_scope=import_scope))
    else:
      self._initial_value = None
    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            variable_def.synchronization, variable_def.aggregation,
            variable_def.trainable, variable_def.variable_name))
    self._synchronization = synchronization
    self._aggregation = aggregation
    self._trainable = trainable
    if variable_def.snapshot_name:
      snapshot = g.as_graph_element(
          ops.prepend_name_scope(
              variable_def.snapshot_name, import_scope=import_scope))
      if snapshot.op.type != "ReadVariableOp":
        self._cached_value = snapshot
      else:
        self._cached_value = None
      while snapshot.op.type != "ReadVariableOp":
        snapshot = snapshot.op.inputs[0]
      self._graph_element = snapshot
    else:
      self._cached_value = None
      self._graph_element = g.get_tensor_by_name(self._handle.op.name +
                                                 "/Read/ReadVariableOp:0")
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def,
          import_scope=import_scope)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._dtype = dtypes.as_dtype(self._handle.op.get_attr("dtype"))
    self._constraint = None
    self._validate_shape = validate_shape
class UninitializedVariable(BaseResourceVariable):
      self,
      trainable=None,
      caching_device=None,
      name=None,
      shape=None,
      dtype=None,
      constraint=None,
      synchronization=None,
      aggregation=None,
      extra_handle_data=None,
      distribute_strategy=None,
      **unused_kwargs):
    """Creates the variable handle.
    Args:
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      shape: The variable's shape.
      dtype: The variable's dtype.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      extra_handle_data: Optional, another resource handle or Tensor with handle
        data to merge with `shape` and `dtype`.
      distribute_strategy: The tf.distribute.Strategy this variable is being
        created inside of.
    """
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
      with ops.name_scope(name, "Variable", skip_on_eager=False) as name:
        handle_name = ops.name_from_scope_name(name)
        if self._in_graph_mode:
          shared_name = handle_name
          unique_id = shared_name
        else:
          unique_id = "%s_%d" % (handle_name, ops.uid())
        handle = _variable_handle_from_shape_and_dtype(
            shape=shape,
            dtype=dtype,
            shared_name=shared_name,
            name=name,
            graph_mode=self._in_graph_mode,
            initial_value=extra_handle_data)
        handle._parent_trackable = weakref.ref(self)
        if self._in_graph_mode:
          with ops.name_scope("Read"):
            with ops.device(handle.device):
              value = gen_resource_variable_ops.read_variable_op(handle, dtype)
              _maybe_set_handle_data(dtype, handle, value)
            graph_element = value
          ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, self)
        else:
          graph_element = None
    super(UninitializedVariable, self).__init__(
        distribute_strategy=distribute_strategy,
        shape=shape,
        dtype=dtype,
        unique_id=unique_id,
        handle_name=handle_name,
        constraint=constraint,
        handle=handle,
        graph_element=graph_element,
        trainable=trainable,
        synchronization=synchronization,
        aggregation=aggregation,
        in_graph_mode=self._in_graph_mode)
_pywrap_utils.RegisterType("ResourceVariable", ResourceVariable)
def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
ops.register_tensor_conversion_function(BaseResourceVariable,
                                        _dense_var_to_tensor)
class _UnreadVariable(BaseResourceVariable):
  def __init__(self, handle, dtype, shape, in_graph_mode, parent_op,
               unique_id):
    if isinstance(handle, ops.EagerTensor):
      handle_name = ""
    else:
      handle_name = handle.name
    if context.executing_eagerly() or ops.inside_function():
      graph_element = None
    else:
      with ops.control_dependencies([parent_op]):
        graph_element = gen_resource_variable_ops.read_variable_op(
            handle, dtype)
        _maybe_set_handle_data(dtype, handle, graph_element)
    super(_UnreadVariable, self).__init__(
        handle=handle,
        shape=shape,
        handle_name=handle_name,
        unique_id=unique_id,
        dtype=dtype,
        graph_element=graph_element)
    self._parent_op = parent_op
  @property
  def name(self):
    if self._in_graph_mode:
      return self._parent_op.name
    else:
      return "UnreadVariable"
  def value(self):
    return self._read_variable_op()
  def read_value(self):
    return self._read_variable_op()
  def _read_variable_op(self):
    with ops.control_dependencies([self._parent_op]):
      result = gen_resource_variable_ops.read_variable_op(
          self._handle, self._dtype)
      _maybe_set_handle_data(self._dtype, self._handle, result)
      return result
  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).assign_sub(delta, use_locking, name,
                                                     read_value)
  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).assign_add(delta, use_locking, name,
                                                     read_value)
  def assign(self, value, use_locking=None, name=None, read_value=True):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).assign(value, use_locking, name,
                                                 read_value)
  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_sub(sparse_delta, use_locking,
                                                      name)
  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_add(sparse_delta, use_locking,
                                                      name)
  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_max(sparse_delta, use_locking,
                                                      name)
  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_min(sparse_delta, use_locking,
                                                      name)
  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_mul(sparse_delta, use_locking,
                                                      name)
  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_div(sparse_delta, use_locking,
                                                      name)
  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable,
                   self).scatter_update(sparse_delta, use_locking, name)
  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable,
                   self).batch_scatter_update(sparse_delta, use_locking, name)
  def scatter_nd_sub(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_nd_sub(indices, updates, name)
  def scatter_nd_add(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_nd_add(indices, updates, name)
  def scatter_nd_update(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable,
                   self).scatter_nd_update(indices, updates, name)
  def scatter_nd_max(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_nd_max(indices, updates, name)
  def scatter_nd_min(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_nd_min(indices, updates, name)
  @property
  def op(self):
    return self._parent_op
@ops.RegisterGradient("ReadVariableOp")
def _ReadGrad(_, grad):
  return grad
def variable_shape(handle, out_type=dtypes.int32):
  handle_data = get_eager_safe_handle_data(handle)
  if handle_data is None or not handle_data.is_set:
    return gen_resource_variable_ops.variable_shape(handle, out_type=out_type)
  shape_proto = handle_data.shape_and_type[0].shape
  if shape_proto.unknown_rank or any(x.size == -1 for x in shape_proto.dim):
    return gen_resource_variable_ops.variable_shape(handle, out_type=out_type)
  return constant_op.constant([x.size for x in shape_proto.dim], dtype=out_type)
@ops.RegisterGradient("ResourceGather")
def _GatherGrad(op, grad):
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = variable_shape(handle)
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return (indexed_slices.IndexedSlices(values, indices, params_shape), None)
def _to_proto_fn(v, export_scope=None):
  return v.to_proto(export_scope=export_scope)
def _from_proto_fn(v, import_scope=None):
  if v.is_resource:
    return ResourceVariable.from_proto(v, import_scope=import_scope)
  return variables.Variable.from_proto(v, import_scope=import_scope)
ops.register_proto_function(
    ops.GraphKeys.GLOBAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.TRAINABLE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.LOCAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.MODEL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.GLOBAL_STEP,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.METRIC_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
@tf_export("__internal__.ops.is_resource_variable", v1=[])
def is_resource_variable(var):
  return isinstance(var, BaseResourceVariable) or hasattr(
      var, "_should_act_as_resource_variable")
def copy_to_graph_uninitialized(var):
  new_variable = UninitializedVariable(
      trainable=var.trainable,
      constraint=var._constraint,
      shape=var.shape,
      dtype=var.dtype,
      name=var._shared_name,
      synchronization=var.synchronization,
      aggregation=var.aggregation,
      extra_handle_data=var.handle)
  new_variable._maybe_initialize_trackable()
  return new_variable
ops.NotDifferentiable("Assert")
ops.NotDifferentiable("VarIsInitializedOp")
ops.NotDifferentiable("VariableShape")
class VariableSpec(tensor_spec.DenseSpec):
  __slots__ = ["trainable"]
  value_type = property(lambda self: BaseResourceVariable)
  def __init__(self,
               shape,
               dtype=dtypes.float32,
               trainable=True):
    super(VariableSpec, self).__init__(shape, dtype=dtype)
    self.trainable = trainable
  def is_compatible_with(self, spec_or_value):
    return (isinstance(spec_or_value, (type(self), self.value_type)) and
            self.shape.is_compatible_with(spec_or_value.shape) and
            self.dtype == spec_or_value.dtype and
            self.trainable == spec_or_value.trainable)
  @classmethod
  def from_value(cls, value):
    return cls(
        value.shape,
        dtype=value.dtype,
        trainable=value.trainable)
  def _to_components(self, value):
    return value.handle
  def _from_components(self, components):
    return BaseResourceVariable(
        trainable=self.trainable,
        shape=self.shape,
        dtype=self.dtype,
        handle=components)
  @property
  def _component_specs(self):
    return tensor_spec.TensorSpec(self.shape, dtypes.resource)
  def _from_compatible_tensor_list(self, tensor_list):
    assert len(tensor_list) == 1
    return tensor_list[0]
  def _serialize(self):
    return self.shape, self.dtype, self.trainable
  def __tf_tracing_type__(self, signature_context):
    return signature_context.make_reference_type(self, id(self))
  def __repr__(self):
    return (f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, "
            f"trainable={self.trainable})")
  def __hash__(self):
    return hash((self.shape, self.dtype, self.trainable))
  def __eq__(self, other):
    return (type(self) is type(other) and
            self.shape == other.shape and
            self.dtype == other.dtype and
            self.trainable == other.trainable)
_pywrap_utils.RegisterType("VariableSpec", VariableSpec)
def write_object_proto_for_resource_variable(resource_variable, proto, options):
  proto.variable.SetInParent()
  if not resource_variable.name.endswith(":0"):
    raise ValueError(f"Cowardly refusing to save variable "
                     f"{resource_variable.name} because of "
                     f"unexpected suffix in the name (':0') "
                     f"which won't be restored.")
  proto.variable.trainable = resource_variable.trainable
  proto.variable.dtype = resource_variable.dtype.as_datatype_enum
  proto.variable.synchronization = resource_variable.synchronization.value
  proto.variable.aggregation = resource_variable.aggregation.value
  proto.variable.shape.CopyFrom(resource_variable.shape.as_proto())
  ):
    if hasattr(resource_variable, "device"):
      proto.variable.device = resource_variable.device
