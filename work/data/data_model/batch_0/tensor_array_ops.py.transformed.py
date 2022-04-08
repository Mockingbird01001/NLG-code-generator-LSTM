
import contextlib
import traceback
import weakref
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
class _GraphTensorArray:
  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               name=None):
    """Constructs a graph mode TensorArray.
    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: Boolean (optional, default: True).  If True, clear
        TensorArray values after reading them.  This disables read-many
        semantics, but allows early release of memory.
      tensor_array_name: (optional) Python string: the name of the TensorArray.
        This is used when creating the TensorArray handle.  If this value is
        set, handle should be None.
      handle: (optional) A `Tensor` handle to an existing TensorArray.  If this
        is set, tensor_array_name should be None. Only supported in graph mode.
      flow: (optional) A float `Tensor` scalar coming from an existing
        `TensorArray.flow`. Only supported in graph mode.
      infer_shape: (optional, default: True) If True, shape inference is
        enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray. Need
        not be fully defined.
      colocate_with_first_write_call: If `True`, the TensorArray will be
        colocated on the same device as the Tensor used on its first write
        (write operations include `write`, `unstack`, and `split`).  If `False`,
        the TensorArray will be placed on the device determined by the device
        context available during its initialization.
      name: A name for the operation (optional).
    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
    if handle is not None and tensor_array_name:
      raise ValueError(
          "Cannot provide both `handle` and `tensor_array_name` arguments at "
          "the same time.")
    if handle is not None and not isinstance(handle, ops.Tensor):
      raise TypeError(
          f"Expected `handle` to be a Tensor, but got `{handle}` of type "
          f"`{type(handle)}` instead.")
    if handle is None and size is None:
      raise ValueError(
          "Argument `size` must be provided if handle is not provided.")
    if handle is not None and size is not None:
      raise ValueError("Cannot provide both a `handle` and `size` arguments "
                       "at the same time.")
    if handle is not None and element_shape is not None:
      raise ValueError(
          "Cannot provide both `handle` and `element_shape` arguments "
          "at the same time.")
    if handle is not None and dynamic_size is not None:
      raise ValueError(
          "Cannot provide both `handle` and `dynamic_size` arguments "
          "at the same time.")
    if handle is not None and clear_after_read is not None:
      raise ValueError(
          "Cannot provide both `handle` and `clear_after_read` arguments "
          "at the same time.")
    if clear_after_read is None:
      clear_after_read = True
    self._dynamic_size = dynamic_size or False
    self._dtype = dtypes.as_dtype(dtype).base_dtype
    self._colocate_with_first_write_call = colocate_with_first_write_call
    if colocate_with_first_write_call:
      self._colocate_with = []
    else:
      self._colocate_with = None
    self._element_shape = [tensor_shape.as_shape(element_shape)]
    self._infer_shape = infer_shape
    self._size = size
    with ops.name_scope(name, "TensorArray", [handle, size, flow]) as scope:
      if handle is not None:
        self._handle = handle
        if flow is None:
          raise ValueError("flow must not be None if handle is not None.")
        self._flow = flow
      else:
        def create():
          return gen_data_flow_ops.tensor_array_v3(
              dtype=dtype,
              size=size,
              element_shape=element_shape,
              identical_element_shapes=infer_shape,
              dynamic_size=self._dynamic_size,
              clear_after_read=clear_after_read,
              tensor_array_name=tensor_array_name,
              name=scope)
        if colocate_with_first_write_call:
          with ops.device(None), ops.colocate_with(None, ignore_existing=True):
            self._handle, self._flow = create()
        else:
          self._handle, self._flow = create()
  @property
  def flow(self):
    return self._flow
  @property
  def dtype(self):
    return self._dtype
  @property
  def handle(self):
    return self._handle
  @property
  def element_shape(self):
    return self._element_shape[0]
  def _check_element_shape(self, shape):
    if not shape.is_compatible_with(self.element_shape):
      raise ValueError("Inconsistent shapes: saw %s but expected %s " %
                       (shape, self.element_shape))
    if self._infer_shape:
      self._element_shape[0] = self.element_shape.merge_with(shape)
  @contextlib.contextmanager
  def _maybe_colocate_with(self, value):
    if not self._colocate_with_first_write_call:
      yield
    else:
      if not self._colocate_with:
        self._colocate_with.append(value)
      with ops.colocate_with(self._colocate_with[0]):
        yield
  def identity(self):
    flow = array_ops.identity(self._flow)
    return build_ta_with_new_flow(self, flow)
  def grad(self, source, flow=None, name=None):
    if flow is None:
      flow = self.flow
    with ops.name_scope(name, "TensorArrayGrad", [self._handle]):
      with ops.colocate_with(self._handle):
        g_handle, unused_flow = gen_data_flow_ops.tensor_array_grad_v3(
            handle=self._handle, source=source, flow_in=flow, name=name)
        with ops.control_dependencies([g_handle]):
          flow = array_ops.identity(flow, name="gradient_flow")
        g = TensorArray(
            dtype=self._dtype,
            handle=g_handle,
            flow=flow,
            infer_shape=self._infer_shape,
            colocate_with_first_write_call=False)
        g._implementation._element_shape = self._element_shape
        return g
  def read(self, index, name=None):
    value = gen_data_flow_ops.tensor_array_read_v3(
        handle=self._handle,
        index=index,
        flow_in=self._flow,
        dtype=self._dtype,
        name=name)
    if self._element_shape:
      value.set_shape(self._element_shape[0].dims)
    return value
  def write(self, index, value, name=None):
    with ops.name_scope(name, "TensorArrayWrite", [self._handle, index, value]):
      value = ops.convert_to_tensor(
          value, preferred_dtype=self._dtype, name="value")
      _check_dtypes(value, self._dtype)
      self._check_element_shape(value.shape)
      with self._maybe_colocate_with(value):
        flow_out = gen_data_flow_ops.tensor_array_write_v3(
            handle=self._handle,
            index=index,
            value=value,
            flow_in=self._flow,
            name=name)
      return build_ta_with_new_flow(self, flow_out)
  def stack(self, name=None):
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "TensorArrayStack", [self._handle]):
        value = self.gather(math_ops.range(0, self.size()), name=name)
        if (self.element_shape and not self._dynamic_size and
            self._size is not None):
          value.set_shape([tensor_util.constant_value(self._size)] +
                          self.element_shape.dims)
        return value
  def gather(self, indices, name=None):
    if self._element_shape:
      element_shape = self._element_shape[0]
    else:
      element_shape = tensor_shape.unknown_shape(None)
    value = gen_data_flow_ops.tensor_array_gather_v3(
        handle=self._handle,
        indices=indices,
        flow_in=self._flow,
        dtype=self._dtype,
        name=name,
        element_shape=element_shape)
    if self.element_shape:
      value.set_shape([None] + self.element_shape.dims)
    return value
  def concat(self, name=None):
    value, _ = gen_data_flow_ops.tensor_array_concat_v3(
        handle=self._handle,
        flow_in=self._flow,
        dtype=self._dtype,
        name=name,
        element_shape_except0=self.element_shape[1:])
    if self.element_shape:
      value.set_shape([None] + self.element_shape.dims[1:])
    return value
  @tf_should_use.should_use_result
  def unstack(self, value, name=None):
    with ops.name_scope(name, "TensorArrayUnstack", [self._handle, value]):
      num_elements = array_ops.shape(value)[0]
      return self.scatter(
          indices=math_ops.range(0, num_elements), value=value, name=name)
  @tf_should_use.should_use_result
  def scatter(self, indices, value, name=None):
    with ops.name_scope(name, "TensorArrayScatter",
                        [self._handle, value, indices]):
      value = ops.convert_to_tensor(
          value, preferred_dtype=self._dtype, name="value")
      _check_dtypes(value, self._dtype)
      if not context.executing_eagerly():
        self._check_element_shape(value.shape[1:])
      with self._maybe_colocate_with(value):
        flow_out = gen_data_flow_ops.tensor_array_scatter_v3(
            handle=self._handle,
            indices=indices,
            value=value,
            flow_in=self._flow,
            name=name)
      return build_ta_with_new_flow(self, flow_out)
  @tf_should_use.should_use_result
  def split(self, value, lengths, name=None):
    with ops.name_scope(name, "TensorArraySplit",
                        [self._handle, value, lengths]):
      value = ops.convert_to_tensor(value, dtype=self._dtype, name="value")
      with self._maybe_colocate_with(value):
        lengths_64 = math_ops.cast(lengths, dtypes.int64)
        if not context.executing_eagerly():
          clengths = tensor_util.constant_value(lengths_64)
          if value.shape.dims is not None and clengths is not None:
            if clengths.shape and clengths.max() == clengths.min():
              self._check_element_shape(
                  tensor_shape.TensorShape([clengths[0]
                                           ]).concatenate(value.shape[1:]))
        flow_out = gen_data_flow_ops.tensor_array_split_v3(
            handle=self._handle,
            value=value,
            lengths=lengths_64,
            flow_in=self._flow,
            name=name)
      return build_ta_with_new_flow(self, flow_out)
  def size(self, name=None):
    if not self._dynamic_size and self._size is not None:
      return ops.convert_to_tensor(self._size, dtype=dtypes.int32)
    else:
      return gen_data_flow_ops.tensor_array_size_v3(
          handle=self._handle, flow_in=self.flow, name=name)
  @tf_should_use.should_use_result
  def close(self, name=None):
    return gen_data_flow_ops.tensor_array_close_v3(
        handle=self._handle, name=name)
class _GraphTensorArrayV2:
  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               name=None):
    """Constructs a graph mode TensorArray.
    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if flow is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: (optional) unused. Not supported in TensorLists.
      tensor_array_name: (optional) unused.
      handle: (optional) Must always be None.
      flow: (optional) A variant `Tensor` scalar for a TensorList.
      infer_shape: (optional, default: True) If True, shape inference is
        enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray. Need
        not be fully defined.
      colocate_with_first_write_call: (optional). unused.
      name: (optional) A name for the operation.
    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
    assert handle is None
    del handle
    del clear_after_read
    del tensor_array_name
    del colocate_with_first_write_call
    self._dynamic_size = dynamic_size
    self._size = size
    if (flow is not None and
        (not isinstance(flow, ops.Tensor) or flow.dtype != dtypes.variant)):
      raise TypeError(
          f"Expected `flow` to be a variant tensor, but received `{flow.dtype}` "
          f"instead.")
    if flow is None and size is None:
      raise ValueError("Argument `size` must be provided if argument `flow` "
                       "is not provided.")
    if flow is not None and size is not None:
      raise ValueError("Cannot provide both `flow` and `size` arguments "
                       "at the same time.")
    if flow is not None and element_shape is not None:
      raise ValueError(
          "Cannot provide both `flow` and `element_shape` arguments"
          "at the same time.")
    self._dtype = dtypes.as_dtype(dtype).base_dtype
    self._element_shape = [tensor_shape.as_shape(element_shape)]
    self._infer_shape = infer_shape
    with ops.name_scope(name, "TensorArrayV2", [size, flow]) as scope:
      if flow is None:
        self._flow = list_ops.tensor_list_reserve(
            element_shape=element_shape,
            num_elements=size,
            element_dtype=dtype,
            name=scope)
      else:
        self._flow = flow
    self._colocate_with_first_write_call = None
    self._colocate_with = None
  @property
  def flow(self):
    return self._flow
  @property
  def dtype(self):
    return self._dtype
  @property
  def element_shape(self):
    return self._element_shape[0]
  @property
  def handle(self):
    return None
  def _check_element_shape(self, shape):
    if not shape.is_compatible_with(self.element_shape):
      raise ValueError("Inconsistent shapes: saw %s but expected %s " %
                       (shape, self.element_shape))
    if self._infer_shape:
      self._element_shape[0] = self.element_shape.merge_with(shape)
  def identity(self):
    flow = array_ops.identity(self._flow)
    return build_ta_with_new_flow(self, flow)
  def grad(self, source, flow=None, name=None):
    raise NotImplementedError()
  def read(self, index, name=None):
    with ops.name_scope(name, "TensorArrayV2Read", [self._flow, index]):
      value = list_ops.tensor_list_get_item(
          input_handle=self._flow,
          index=index,
          element_dtype=self._dtype,
          element_shape=self.element_shape,
          name=name)
      return value
  def write(self, index, value, name=None):
    with ops.name_scope(name, "TensorArrayV2Write", [self._flow, index, value]):
      value = ops.convert_to_tensor(
          value, preferred_dtype=self._dtype, name="value")
      _check_dtypes(value, self._dtype)
      self._check_element_shape(value.shape)
      flow_out = list_ops.tensor_list_set_item(
          input_handle=self._flow,
          index=index,
          item=value,
          resize_if_index_out_of_bounds=self._dynamic_size,
          name=name)
      return build_ta_with_new_flow(self, flow_out)
  def stack(self, name=None):
    with ops.name_scope(name, "TensorArrayV2Stack", [self._flow]):
      if not self._dynamic_size and self._size is not None:
        ta_size = tensor_util.constant_value(self._size)
      else:
        ta_size = -1
      value = list_ops.tensor_list_stack(
          input_handle=self._flow,
          element_dtype=self._dtype,
          num_elements=ta_size,
          element_shape=self.element_shape)
      return value
  def gather(self, indices, name=None):
    value = list_ops.tensor_list_gather(
        input_handle=self._flow,
        indices=indices,
        element_dtype=self._dtype,
        element_shape=self.element_shape,
        name=name)
    return value
  def concat(self, name=None):
    if self.element_shape:
      element_shape = [None] + self.element_shape.dims[1:]
    else:
      element_shape = None
    value = list_ops.tensor_list_concat(
        input_handle=self._flow,
        element_dtype=self._dtype,
        element_shape=element_shape,
        name=name)
    return value
  @tf_should_use.should_use_result
  def unstack(self, value, name=None):
    with ops.name_scope(name, "TensorArrayUnstack", [self._flow, value]):
      value = ops.convert_to_tensor(
          value, preferred_dtype=self._dtype, name="value")
      _check_dtypes(value, self._dtype)
      self._check_element_shape(value.shape[1:])
      flow_out = list_ops.tensor_list_from_tensor(
          tensor=value, element_shape=value.shape[1:])
      return build_ta_with_new_flow(self, flow_out)
  @tf_should_use.should_use_result
  def scatter(self, indices, value, name=None):
    with ops.name_scope(name, "TensorArrayScatter",
                        [self._flow, value, indices]):
      value = ops.convert_to_tensor(
          value, preferred_dtype=self._dtype, name="value")
      _check_dtypes(value, self._dtype)
      self._check_element_shape(value.shape[1:])
      flow_out = list_ops.tensor_list_scatter(
          tensor=value,
          indices=indices,
          element_shape=self.element_shape,
          input_handle=self._flow)
      return build_ta_with_new_flow(self, flow_out)
  @tf_should_use.should_use_result
  def split(self, value, lengths, name=None):
    with ops.name_scope(name, "TensorArraySplit", [self._flow, value, lengths]):
      value = ops.convert_to_tensor(
          value, preferred_dtype=self._dtype, name="value")
      _check_dtypes(value, self._dtype)
      lengths_64 = math_ops.cast(lengths, dtypes.int64)
      if not context.executing_eagerly():
        clengths = tensor_util.constant_value(lengths_64)
        if value.shape.dims is not None and clengths is not None:
          if clengths.shape and clengths.max() == clengths.min():
            self._check_element_shape(
                tensor_shape.TensorShape([clengths[0]
                                         ]).concatenate(value.shape[1:]))
      flow_out = list_ops.tensor_list_split(
          tensor=value,
          lengths=lengths_64,
          element_shape=self.element_shape,
          name=name)
      return build_ta_with_new_flow(self, flow_out)
  def size(self, name=None):
    if not self._dynamic_size and self._size is not None:
      return ops.convert_to_tensor(self._size, dtype=dtypes.int32)
    else:
      return list_ops.tensor_list_length(input_handle=self._flow, name=name)
  def close(self, name=None):
    return gen_control_flow_ops.no_op(name=name)
class _EagerTensorArray:
  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               name=None):
    """Constructs a TensorArray compatible with eager execution.
    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: Boolean (optional, default: True).  If True, clear
        TensorArray values after reading them.  This disables read-many
        semantics, but allows early release of memory.
      tensor_array_name: unused.
      handle: unsupported.
      flow: unsupported.
      infer_shape: used for error checking, same semantics as TensorArray.
      element_shape: used for error checking, same semantics as TensorArray.
      colocate_with_first_write_call: unsupported.
      name: unsupported.
    Raises:
      ValueError: handle or flow are supplied, or if size is not supplied.
    """
    if handle is not None:
      raise ValueError("TensorArray handles are not supported when eager "
                       "execution is enabled.")
    if size is None:
      raise ValueError("Size must be declared for TensorArrays when eager "
                       "execution is enabled.")
    self._handle = None
    self._flow = constant_op.constant(0, dtype=dtypes.int32)
    self._infer_shape = infer_shape
    self._element_shape = tensor_shape.as_shape(element_shape)
    self._colocate_with_first_write_call = colocate_with_first_write_call
    self._dtype = dtypes.as_dtype(dtype).base_dtype
    self._dynamic_size = dynamic_size or False
    self._clear_after_read = (True
                              if clear_after_read is None else clear_after_read)
    self._previously_read_indices = []
    if isinstance(size, ops.EagerTensor):
      size = size.numpy()
    self._tensor_array = [None for _ in range(size)]
  @property
  def flow(self):
    return self._flow
  @property
  def dtype(self):
    return self._dtype
  @property
  def handle(self):
    return self._handle
  @property
  def element_shape(self):
    return self._element_shape
  def identity(self):
    return self.parent()
  def grad(self, source, flow=None, name=None):
    raise NotImplementedError(
        "TensorArray.grad is not supported when executing eagerly; eager's "
        "gradient implementation does not use/need this function to compute "
        "gradients of operations that use TensorArrays.")
  def read(self, index, name=None):
    if isinstance(index, ops.EagerTensor):
      index = index.numpy()
    if index < 0:
      raise errors_impl.OutOfRangeError(
          None, None,
          "Reading from negative indices (index %d) is not allowed." % index)
    if index >= len(self._tensor_array):
      raise errors_impl.OutOfRangeError(
          None, None, "Tried to read from index %d but array size is: %d " %
          (index, len(self._tensor_array)))
    tensor = self._tensor_array[index]
    if tensor is None:
      if index in self._previously_read_indices:
        raise errors_impl.InvalidArgumentError(
            None, None,
            "Could not read index %d twice because it was cleared after "
            "a previous read (perhaps try setting clear_after_read = false?)" %
            index)
      else:
        tensor = self._maybe_zero(index)
    if self._clear_after_read:
      self._tensor_array[index] = None
      self._previously_read_indices.append(index)
    return tensor
  def _write(self, index, value):
    if isinstance(index, ops.EagerTensor):
      index = index.numpy()
    if index < 0:
      raise errors_impl.OutOfRangeError(
          None, None,
          "Writing to negative indices (index %d) is not allowed." % index)
    size = len(self._tensor_array)
    if index >= size:
      if not self._dynamic_size:
        raise errors_impl.OutOfRangeError(
            None, None,
            "Tried to write to index %d but array is not resizeable and size "
            "is: %d " % (index, size))
      self._tensor_array.extend(None for _ in range(index - size + 1))
    if not isinstance(value, ops.EagerTensor):
      value = ops.convert_to_tensor(
          value, preferred_dtype=self._dtype, name="value")
    if self._dtype != value.dtype:
      raise errors_impl.InvalidArgumentError(
          None, None,
          "TensorArray dtype is %s but Op is trying to write dtype %s " %
          (self._dtype.name, value.dtype.name))
    if not self._element_shape.is_compatible_with(value.shape):
      raise ValueError("Incompatible shape for value (%s), expected (%s)" %
                       (value.shape, self._element_shape))
    if self._infer_shape:
      self._element_shape = self._element_shape.merge_with(value.shape)
    self._tensor_array[index] = value
  def write(self, index, value, name=None):
    self._write(index, value)
    return self.parent()
  def _maybe_zero(self, ix):
    val = self._tensor_array[ix]
    if val is None:
      val = self._tensor_array[ix] = array_ops.zeros(
          shape=self._element_shape, dtype=self._dtype)
    return val
  def stack(self, name=None):
    if self._tensor_array:
      for ix in range(len(self._tensor_array)):
        self._maybe_zero(ix)
    if not self._tensor_array and self._element_shape.is_fully_defined():
      return ops.convert_to_tensor(
          np.ndarray([0] + self._element_shape), name=name, dtype=self._dtype)
    else:
      return ops.convert_to_tensor(
          self._tensor_array, name=name, dtype=self._dtype)
  def gather(self, indices, name=None):
    if isinstance(indices, ops.EagerTensor):
      indices = indices.numpy()
    return array_ops.stack([self._maybe_zero(i) for i in indices])
  def concat(self, name=None):
    try:
      return array_ops.concat(
          [self._maybe_zero(ix) for ix in range(len(self._tensor_array))],
          0,
          name=name)
    except errors_impl.OpError:
      shapes = [t.shape for t in self._tensor_array]
      ndims = [s.ndims for s in shapes]
      if 0 in ndims:
        idx = ndims.index(0)
        raise errors_impl.InvalidArgumentError(
            None, None, "Concat saw a scalar shape at index %d but requires "
            "at least vectors." % idx)
      else:
        raise
  def unstack(self, value, name=None):
    tensors = array_ops.unstack(value, name=name)
    if len(tensors) > len(self._tensor_array) and not self._dynamic_size:
      raise ValueError(
          "Cannot unstack %d tensors into a TensorArray of static size %d " %
          (len(tensors), len(self._tensor_array)))
    self._tensor_array = tensors
    return self.parent()
  def scatter(self, indices, value, name=None):
    if isinstance(indices, ops.EagerTensor):
      indices = indices.numpy()
    for index, val in zip(indices, array_ops.unstack(value)):
    return self.parent()
  def split(self, value, lengths, name=None):
    value = ops.convert_to_tensor(
        value, preferred_dtype=self._dtype, name="value")
    _check_dtypes(value, self._dtype)
    lengths = ops.convert_to_tensor(lengths)
    sum_lengths = math_ops.reduce_sum(lengths)
    if lengths.shape.ndims != 1:
      raise errors_impl.InvalidArgumentError(
          None, None, "Expected lengths to be a vector, received shape: %s " %
          lengths.shape.as_list())
    elif value.shape.ndims == 0:
      raise errors_impl.InvalidArgumentError(
          None, None, "Expected value to be at least a vector, "
          "but received shape: %s " % value.shape.as_list())
    elif sum_lengths.numpy() != value.shape.as_list()[0]:
      raise errors_impl.InvalidArgumentError(
          None, None, "Expected sum of lengths to be equal to "
          "values.shape[0], but sum of lengths is %d and "
          "value's shape is: %s " % (sum_lengths.numpy(),
                                     value.shape.as_list()))
    elif not self._dynamic_size and lengths.shape[0] != len(self._tensor_array):
      raise errors_impl.InvalidArgumentError(
          None, None, "TensorArray's size is not equal to the size of "
          "lengths (%d vs. %d), and the TensorArray is not marked as "
          "dynamically resizeable." %
          (len(self._tensor_array), lengths.shape[0]))
    else:
      self._tensor_array = array_ops.split(value, lengths, name=name)
      return self.parent()
  def size(self, name=None):
    return constant_op.constant(len(self._tensor_array))
  def close(self, name=None):
    del self._tensor_array[:]
@tf_export("TensorArray")
class TensorArray:
  """Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.
  This class is meant to be used with dynamic iteration primitives such as
  `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  "flow" control flow dependencies.
  Example 1: Plain reading and writing.
  >>> ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
  >>> ta = ta.write(0, 10)
  >>> ta = ta.write(1, 20)
  >>> ta = ta.write(2, 30)
  >>>
  >>> ta.read(0)
  <tf.Tensor: shape=(), dtype=float32, numpy=10.0>
  >>> ta.read(1)
  <tf.Tensor: shape=(), dtype=float32, numpy=20.0>
  >>> ta.read(2)
  <tf.Tensor: shape=(), dtype=float32, numpy=30.0>
  >>> ta.stack()
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([10., 20., 30.],
  dtype=float32)>
  Example 2: Fibonacci sequence algorithm that writes in a loop then returns.
  >>> @tf.function
  ... def fibonacci(n):
  ...   ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  ...   ta = ta.unstack([0., 1.])
  ...
  ...   for i in range(2, n):
  ...     ta = ta.write(i, ta.read(i - 1) + ta.read(i - 2))
  ...
  ...   return ta.stack()
  >>>
  >>> fibonacci(7)
  <tf.Tensor: shape=(7,), dtype=float32,
  numpy=array([0., 1., 1., 2., 3., 5., 8.], dtype=float32)>
  Example 3: A simple loop interacting with a `tf.Variable`.
  >>> v = tf.Variable(1)
  >>> @tf.function
  ... def f(x):
  ...   ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  ...   for i in tf.range(x):
  ...     v.assign_add(i)
  ...     ta = ta.write(i, v)
  ...   return ta.stack()
  >>> f(5)
  <tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 1,  2,  4,  7, 11],
  dtype=int32)>
  """
  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               name=None):
    """Construct a new TensorArray or wrap an existing TensorArray handle.
    A note about the parameter `name`:
    The name of the `TensorArray` (even if passed in) is uniquified: each time
    a new `TensorArray` is created at runtime it is assigned its own name for
    the duration of the run.  This avoids name collisions if a `TensorArray`
    is created within a `while_loop`.
    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: Boolean (optional, default: True).  If True, clear
        TensorArray values after reading them.  This disables read-many
        semantics, but allows early release of memory.
      tensor_array_name: (optional) Python string: the name of the TensorArray.
        This is used when creating the TensorArray handle.  If this value is
        set, handle should be None.
      handle: (optional) A `Tensor` handle to an existing TensorArray.  If this
        is set, tensor_array_name should be None. Only supported in graph mode.
      flow: (optional) A float `Tensor` scalar coming from an existing
        `TensorArray.flow`. Only supported in graph mode.
      infer_shape: (optional, default: True) If True, shape inference is
        enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray. Need
        not be fully defined.
      colocate_with_first_write_call: If `True`, the TensorArray will be
        colocated on the same device as the Tensor used on its first write
        (write operations include `write`, `unstack`, and `split`).  If `False`,
        the TensorArray will be placed on the device determined by the device
        context available during its initialization.
      name: A name for the operation (optional).
    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
    if (context.executing_eagerly() and
        (flow is None or flow.dtype != dtypes.variant)):
      implementation = _EagerTensorArray
    elif (flow is not None and flow.dtype == dtypes.variant or
          control_flow_util.EnableControlFlowV2(ops.get_default_graph())):
      implementation = _GraphTensorArrayV2
    else:
      implementation = _GraphTensorArray
    self._implementation = implementation(
        dtype,
        size=size,
        dynamic_size=dynamic_size,
        clear_after_read=clear_after_read,
        tensor_array_name=tensor_array_name,
        handle=handle,
        flow=flow,
        infer_shape=infer_shape,
        element_shape=element_shape,
        colocate_with_first_write_call=colocate_with_first_write_call,
        name=name)
    self._implementation.parent = weakref.ref(self)
  @property
  def flow(self):
    return self._implementation._flow
  @property
  def dtype(self):
    return self._implementation._dtype
  @property
  def handle(self):
    return self._implementation.handle
  @property
  def element_shape(self):
    return self._implementation.element_shape
  @property
  def dynamic_size(self):
    return self._implementation._dynamic_size
  @property
  def _infer_shape(self):
    return self._implementation._infer_shape
  def identity(self):
    return self._implementation.identity()
  def grad(self, source, flow=None, name=None):
    return self._implementation.grad(source, flow=flow, name=name)
  def read(self, index, name=None):
    """Read the value at location `index` in the TensorArray.
    Args:
      index: 0-D.  int32 tensor with the index to read from.
      name: A name for the operation (optional).
    Returns:
      The tensor at index `index`.
    """
    return self._implementation.read(index, name=name)
  @tf_should_use.should_use_result(warn_in_eager=True)
  def write(self, index, value, name=None):
    """Write `value` into index `index` of the TensorArray.
    Args:
      index: 0-D.  int32 scalar with the index to write to.
      value: N-D.  Tensor of type `dtype`.  The Tensor to write to this index.
      name: A name for the operation (optional).
    Returns:
      A new TensorArray object with flow that ensures the write occurs.
      Use this object for all subsequent operations.
    Raises:
      ValueError: if there are more writers than specified.
    """
    return self._implementation.write(index, value, name=name)
  def stack(self, name=None):
    """Return the values in the TensorArray as a stacked `Tensor`.
    All of the values must have been written and their shapes must all match.
    If input shapes have rank-`R`, then output shape will have rank-`(R+1)`.
    For example:
    >>> ta = tf.TensorArray(tf.int32, size=3)
    >>> ta.write(0, tf.constant([1, 2]))
    >>> ta.write(1, tf.constant([3, 4]))
    >>> ta.write(2, tf.constant([5, 6]))
    >>> ta.stack()
    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)>
    Args:
      name: A name for the operation (optional).
    Returns:
      All the tensors in the TensorArray stacked into one tensor.
    """
    return self._implementation.stack(name=name)
  def gather(self, indices, name=None):
    """Return selected values in the TensorArray as a packed `Tensor`.
    All of selected values must have been written and their shapes
    must all match.
    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If the
        `TensorArray` is not dynamic, `max_value=size()`.
      name: A name for the operation (optional).
    Returns:
      The tensors in the `TensorArray` selected by `indices`, packed into one
      tensor.
    """
    return self._implementation.gather(indices, name=name)
  def concat(self, name=None):
    """Return the values in the TensorArray as a concatenated `Tensor`.
    All of the values must have been written, their ranks must match, and
    and their shapes must all match for all dimensions except the first.
    Args:
      name: A name for the operation (optional).
    Returns:
      All the tensors in the TensorArray concatenated into one tensor.
    """
    return self._implementation.concat(name=name)
  @tf_should_use.should_use_result
  def unstack(self, value, name=None):
    """Unstack the values of a `Tensor` in the TensorArray.
    If input value shapes have rank-`R`, then the output TensorArray will
    contain elements whose shapes are rank-`(R-1)`.
    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unstack.
      name: A name for the operation (optional).
    Returns:
      A new TensorArray object with flow that ensures the unstack occurs.
      Use this object for all subsequent operations.
    Raises:
      ValueError: if the shape inference fails.
    """
    return self._implementation.unstack(value, name=name)
  @tf_should_use.should_use_result
  def scatter(self, indices, value, name=None):
    """Scatter the values of a `Tensor` in specific indices of a `TensorArray`.
    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If the
        `TensorArray` is not dynamic, `max_value=size()`.
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unpack.
      name: A name for the operation (optional).
    Returns:
      A new TensorArray object with flow that ensures the scatter occurs.
      Use this object for all subsequent operations.
    Raises:
      ValueError: if the shape inference fails.
    """
    return self._implementation.scatter(indices, value, name=name)
  @tf_should_use.should_use_result
  def split(self, value, lengths, name=None):
    """Split the values of a `Tensor` into the TensorArray.
    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to split.
      lengths: 1-D.  int32 vector with the lengths to use when splitting `value`
        along its first dimension.
      name: A name for the operation (optional).
    Returns:
      A new TensorArray object with flow that ensures the split occurs.
      Use this object for all subsequent operations.
    Raises:
      ValueError: if the shape inference fails.
    """
    return self._implementation.split(value, lengths, name=name)
  def size(self, name=None):
    return self._implementation.size(name=name)
  @tf_should_use.should_use_result
  def close(self, name=None):
    return self._implementation.close(name=name)
def build_ta_with_new_flow(old_ta, flow):
  impl = (old_ta._implementation if isinstance(old_ta, TensorArray) else old_ta)
  if not context.executing_eagerly():
    if (not isinstance(impl, _GraphTensorArrayV2) and
        control_flow_util.EnableControlFlowV2(ops.get_default_graph())):
      raise NotImplementedError("Attempting to build a graph-mode TF2-style "
                                "TensorArray from either an eager-mode "
                                "TensorArray or a TF1-style TensorArray.  "
                                "This is not currently supported.  You may be "
                                "attempting to capture a TensorArray "
                                "inside a tf.function or tf.data map function. "
                                "Instead, construct a new TensorArray inside "
                                "the function.")
  new_ta = TensorArray(
      dtype=impl.dtype,
      handle=impl.handle,
      flow=flow,
      infer_shape=impl._infer_shape,
      colocate_with_first_write_call=impl._colocate_with_first_write_call)
  new_impl = new_ta._implementation
  new_impl._dynamic_size = impl._dynamic_size
  new_impl._size = impl._size
  new_impl._colocate_with = impl._colocate_with
  return new_ta
def _check_dtypes(value, dtype):
  if value.dtype != dtype:
    logging.error("Error: Input value {} has dtype {}, but expected dtype {}.  "
                  "This leads to undefined behavior and will be an error "
                  "in future versions of TensorFlow.  Traceback:\n{}".format(
                      value, str(value.dtype), str(dtype),
                      "".join(traceback.format_stack())))
@tf_export("TensorArraySpec")
@type_spec.register("tf.TensorArraySpec")
class TensorArraySpec(type_spec.TypeSpec):
  __slots__ = ["_element_shape", "_dtype", "_dynamic_size", "_infer_shape"]
  value_type = property(lambda self: TensorArray)
  def __init__(self,
               element_shape=None,
               dtype=dtypes.float32,
               dynamic_size=False,
               infer_shape=True):
    self._element_shape = tensor_shape.as_shape(element_shape)
    self._dtype = dtypes.as_dtype(dtype)
    self._dynamic_size = dynamic_size
    self._infer_shape = infer_shape
  def is_subtype_of(self, other):
    return (isinstance(other, TensorArraySpec) and
            self._dtype == other._dtype and
            self._dynamic_size == other._dynamic_size)
  def most_specific_common_supertype(self, others):
    if not all(isinstance(other, TensorArraySpec) for other in others):
      return False
    common_shape = self._element_shape.most_specific_common_supertype(
        other._element_shape for other in others)
    if common_shape is None:
      return None
    if not all(self._dtype == other._dtype for other in others):
      return None
    if not all(self._dynamic_size == other._dynamic_size for other in others):
      return None
    infer_shape = self._infer_shape and all(
        other._infer_shape for other in others)
    return TensorArraySpec(common_shape, self._dtype, self._dynamic_size,
                           infer_shape)
  def is_compatible_with(self, other):
    if not isinstance(other, type_spec.TypeSpec):
      other = type_spec.type_spec_from_value(other)
    return (isinstance(other, TensorArraySpec) and
            self._dtype.is_compatible_with(other._dtype) and
            self._element_shape.is_compatible_with(other._element_shape) and
            self._dynamic_size == other._dynamic_size)
  def _serialize(self):
    return (self._element_shape, self._dtype, self._dynamic_size,
            self._infer_shape)
  @property
  def _component_specs(self):
    return [tensor_spec.TensorSpec([], dtypes.variant)]
  def _to_components(self, value):
    if not isinstance(value, TensorArray):
      raise TypeError("Expected value to be a TensorArray, but got: `{}`".format(
          type(value)))
    if value.flow is not None and value.flow.dtype == dtypes.variant:
      return [value.flow]
    else:
      with ops.name_scope("convert_tensor_array"):
        flow = list_ops.tensor_list_from_tensor(
            tensor=value.stack(), element_shape=value.element_shape)
      return [flow]
  def _from_components(self, tensor_list):
    ret = TensorArray(
        dtype=self._dtype,
        flow=tensor_list[0],
        dynamic_size=self._dynamic_size,
        infer_shape=self._infer_shape)
    return ret
  @staticmethod
  def from_value(value):
    if not isinstance(value, TensorArray):
      raise TypeError("Expected value to be a TensorArray, but got: `{}`".format(
          type(value)))
    return TensorArraySpec(
        dtype=value.dtype,
        element_shape=value.element_shape,
        dynamic_size=value.dynamic_size,
  def _to_legacy_output_types(self):
    return self._dtype
  def _to_legacy_output_shapes(self):
    return (tensor_shape.TensorShape([self._dynamic_size, self._infer_shape
                                     ]).concatenate(self._element_shape))
  def _to_legacy_output_classes(self):
    return TensorArray
type_spec.register_type_spec_from_value_converter(
    TensorArray, TensorArraySpec.from_value, allow_subclass=True)
