
import numpy as np
import six
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
try:
  from tensorflow.python.framework import fast_tensor_util
  _FAST_TENSOR_UTIL_AVAILABLE = True
except ImportError:
  _FAST_TENSOR_UTIL_AVAILABLE = False
def ExtractBitsFromFloat16(x):
  return np.asarray(x, dtype=np.float16).view(np.uint16).item()
def SlowAppendFloat16ArrayToTensorProto(tensor_proto, proto_values):
  tensor_proto.half_val.extend(
      [ExtractBitsFromFloat16(x) for x in proto_values])
def _MediumAppendFloat16ArrayToTensorProto(tensor_proto, proto_values):
  fast_tensor_util.AppendFloat16ArrayToTensorProto(
      tensor_proto,
      np.asarray(proto_values, dtype=np.float16).view(np.uint16))
def ExtractBitsFromBFloat16(x):
  return np.asarray(
      x, dtype=dtypes.bfloat16.as_numpy_dtype).view(np.uint16).item()
def SlowAppendBFloat16ArrayToTensorProto(tensor_proto, proto_values):
  tensor_proto.half_val.extend(
      [ExtractBitsFromBFloat16(x) for x in proto_values])
def FastAppendBFloat16ArrayToTensorProto(tensor_proto, proto_values):
  fast_tensor_util.AppendBFloat16ArrayToTensorProto(
      tensor_proto, np.asarray(
          proto_values, dtype=dtypes.bfloat16.as_numpy_dtype).view(np.uint16))
if _FAST_TENSOR_UTIL_AVAILABLE:
  _NP_TO_APPEND_FN = {
      dtypes.bfloat16.as_numpy_dtype:
          FastAppendBFloat16ArrayToTensorProto,
      np.float16:
          _MediumAppendFloat16ArrayToTensorProto,
      np.float32:
          fast_tensor_util.AppendFloat32ArrayToTensorProto,
      np.float64:
          fast_tensor_util.AppendFloat64ArrayToTensorProto,
      np.int32:
          fast_tensor_util.AppendInt32ArrayToTensorProto,
      np.int64:
          fast_tensor_util.AppendInt64ArrayToTensorProto,
      np.uint8:
          fast_tensor_util.AppendUInt8ArrayToTensorProto,
      np.uint16:
          fast_tensor_util.AppendUInt16ArrayToTensorProto,
      np.uint32:
          fast_tensor_util.AppendUInt32ArrayToTensorProto,
      np.uint64:
          fast_tensor_util.AppendUInt64ArrayToTensorProto,
      np.int8:
          fast_tensor_util.AppendInt8ArrayToTensorProto,
      np.int16:
          fast_tensor_util.AppendInt16ArrayToTensorProto,
      np.complex64:
          fast_tensor_util.AppendComplex64ArrayToTensorProto,
      np.complex128:
          fast_tensor_util.AppendComplex128ArrayToTensorProto,
      np.object_:
          fast_tensor_util.AppendObjectArrayToTensorProto,
      np.bool_:
          fast_tensor_util.AppendBoolArrayToTensorProto,
      dtypes.qint8.as_numpy_dtype:
          fast_tensor_util.AppendInt8ArrayToTensorProto,
      dtypes.quint8.as_numpy_dtype:
          fast_tensor_util.AppendUInt8ArrayToTensorProto,
      dtypes.qint16.as_numpy_dtype:
          fast_tensor_util.AppendInt16ArrayToTensorProto,
      dtypes.quint16.as_numpy_dtype:
          fast_tensor_util.AppendUInt16ArrayToTensorProto,
      dtypes.qint32.as_numpy_dtype:
          fast_tensor_util.AppendInt32ArrayToTensorProto,
  }
else:
  def SlowAppendFloat32ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.float_val.extend([x.item() for x in proto_values])
  def SlowAppendFloat64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.double_val.extend([x.item() for x in proto_values])
  def SlowAppendIntArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int_val.extend([x.item() for x in proto_values])
  def SlowAppendInt64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int64_val.extend([x.item() for x in proto_values])
  def SlowAppendQIntArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int_val.extend([x.item()[0] for x in proto_values])
  def SlowAppendUInt32ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.uint32_val.extend([x.item() for x in proto_values])
  def SlowAppendUInt64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.uint64_val.extend([x.item() for x in proto_values])
  def SlowAppendComplex64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.scomplex_val.extend(
        [v.item() for x in proto_values for v in [x.real, x.imag]])
  def SlowAppendComplex128ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.dcomplex_val.extend(
        [v.item() for x in proto_values for v in [x.real, x.imag]])
  def SlowAppendObjectArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.string_val.extend([compat.as_bytes(x) for x in proto_values])
  def SlowAppendBoolArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.bool_val.extend([x.item() for x in proto_values])
  _NP_TO_APPEND_FN = {
      dtypes.bfloat16.as_numpy_dtype: SlowAppendBFloat16ArrayToTensorProto,
      np.float16: SlowAppendFloat16ArrayToTensorProto,
      np.float32: SlowAppendFloat32ArrayToTensorProto,
      np.float64: SlowAppendFloat64ArrayToTensorProto,
      np.int32: SlowAppendIntArrayToTensorProto,
      np.int64: SlowAppendInt64ArrayToTensorProto,
      np.uint8: SlowAppendIntArrayToTensorProto,
      np.uint16: SlowAppendIntArrayToTensorProto,
      np.uint32: SlowAppendUInt32ArrayToTensorProto,
      np.uint64: SlowAppendUInt64ArrayToTensorProto,
      np.int8: SlowAppendIntArrayToTensorProto,
      np.int16: SlowAppendIntArrayToTensorProto,
      np.complex64: SlowAppendComplex64ArrayToTensorProto,
      np.complex128: SlowAppendComplex128ArrayToTensorProto,
      np.object_: SlowAppendObjectArrayToTensorProto,
      np.bool_: SlowAppendBoolArrayToTensorProto,
      dtypes.qint8.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      dtypes.quint8.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      dtypes.qint16.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      dtypes.quint16.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      dtypes.qint32.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
  }
def GetFromNumpyDTypeDict(dtype_dict, dtype):
  for key, val in six.iteritems(dtype_dict):
    if key == dtype:
      return val
  return None
def GetNumpyAppendFn(dtype):
  if dtype.type == np.bytes_ or dtype.type == np.str_:
    if _FAST_TENSOR_UTIL_AVAILABLE:
      return fast_tensor_util.AppendObjectArrayToTensorProto
    else:
      return SlowAppendObjectArrayToTensorProto
  return GetFromNumpyDTypeDict(_NP_TO_APPEND_FN, dtype)
def TensorShapeProtoToList(shape):
  return [dim.size for dim in shape.dim]
def _GetDenseDimensions(list_of_lists):
  if not isinstance(list_of_lists, (list, tuple)):
    return []
  elif not list_of_lists:
    return [0]
  else:
    return [len(list_of_lists)] + _GetDenseDimensions(list_of_lists[0])
def _FlattenToStrings(nested_strings):
  if isinstance(nested_strings, (list, tuple)):
    for inner in nested_strings:
      for flattened_string in _FlattenToStrings(inner):
        yield flattened_string
  else:
    yield nested_strings
_TENSOR_CONTENT_TYPES = frozenset([
    dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32, dtypes.uint8,
    dtypes.int16, dtypes.int8, dtypes.int64, dtypes.qint8, dtypes.quint8,
    dtypes.qint16, dtypes.quint16, dtypes.qint32, dtypes.uint32, dtypes.uint64
])
def _check_failed(v):
  raise ValueError(v)
def _check_quantized(values):
  if not isinstance(values, (list, tuple)):
    _check_failed(values)
  if isinstance(values, tuple):
    _ = [_check_int(v) for v in values]
  else:
    _ = [_check_quantized(v) for v in values]
def _generate_isinstance_check(expected_types):
  def inner(values):
    for v in nest.flatten(values):
      if not (isinstance(v, expected_types) or
              (isinstance(v, np.ndarray) and
               issubclass(v.dtype.type, expected_types))):
        _check_failed(v)
  return inner
_check_int = _generate_isinstance_check(
    (compat.integral_types, tensor_shape.Dimension))
_check_float = _generate_isinstance_check(compat.real_types)
_check_complex = _generate_isinstance_check(compat.complex_types)
_check_str = _generate_isinstance_check(compat.bytes_or_text_types)
_check_bool = _generate_isinstance_check(bool)
def _check_not_tensor(values):
  _ = [_check_failed(v) for v in nest.flatten(values)
       if isinstance(v, ops.Tensor)]
_TF_TO_IS_OK = {
    dtypes.bool: _check_bool,
    dtypes.complex128: _check_complex,
    dtypes.complex64: _check_complex,
    dtypes.float16: _check_float,
    dtypes.float32: _check_float,
    dtypes.float64: _check_float,
    dtypes.int16: _check_int,
    dtypes.int32: _check_int,
    dtypes.int64: _check_int,
    dtypes.int8: _check_int,
    dtypes.qint16: _check_quantized,
    dtypes.qint32: _check_quantized,
    dtypes.qint8: _check_quantized,
    dtypes.quint16: _check_quantized,
    dtypes.quint8: _check_quantized,
    dtypes.string: _check_str,
    dtypes.uint16: _check_int,
    dtypes.uint8: _check_int,
    dtypes.uint32: _check_int,
    dtypes.uint64: _check_int,
}
def _AssertCompatible(values, dtype):
  if dtype is None:
    fn = _check_not_tensor
  else:
    try:
      fn = _TF_TO_IS_OK[dtype]
    except KeyError:
      if dtype.is_integer:
        fn = _check_int
      elif dtype.is_floating:
        fn = _check_float
      elif dtype.is_complex:
        fn = _check_complex
      elif dtype.is_quantized:
        fn = _check_quantized
      else:
        fn = _check_not_tensor
  try:
    fn(values)
  except ValueError as e:
    [mismatch] = e.args
    if dtype is None:
      raise TypeError("Expected any non-tensor type, but got a tensor instead.")
    else:
      raise TypeError(f"Expected {dtype.name}, but got {mismatch} of type "
                      f"'{type(mismatch).__name__}'.")
    return False
  if (callable(getattr(obj, "__array__", None)) or
      isinstance(getattr(obj, "__array_interface__", None), dict)):
    return True
  try:
    memoryview(obj)
  except TypeError:
    return False
  else:
    return not isinstance(obj, bytes)
@tf_export("make_tensor_proto")
def make_tensor_proto(values, dtype=None, shape=None, verify_shape=False,
                      allow_broadcast=False):
  """Create a TensorProto.
  In TensorFlow 2.0, representing tensors as protos should no longer be a
  common workflow. That said, this utility function is still useful for
  generating TF Serving request protos:
  ```python
    request = tensorflow_serving.apis.predict_pb2.PredictRequest()
    request.model_spec.name = "my_model"
    request.model_spec.signature_name = "serving_default"
    request.inputs["images"].CopyFrom(tf.make_tensor_proto(X_new))
  ```
  `make_tensor_proto` accepts "values" of a python scalar, a python list, a
  numpy ndarray, or a numpy scalar.
  If "values" is a python scalar or a python list, make_tensor_proto
  first convert it to numpy ndarray. If dtype is None, the
  conversion tries its best to infer the right numpy data
  type. Otherwise, the resulting numpy array has a compatible data
  type with the given dtype.
  In either case above, the numpy ndarray (either the caller provided
  or the auto-converted) must have the compatible type with dtype.
  `make_tensor_proto` then converts the numpy array to a tensor proto.
  If "shape" is None, the resulting tensor proto represents the numpy
  array precisely.
  Otherwise, "shape" specifies the tensor's shape and the numpy array
  can not have more elements than what "shape" specifies.
  Args:
    values:         Values to put in the TensorProto.
    dtype:          Optional tensor_pb2 DataType value.
    shape:          List of integers representing the dimensions of tensor.
    verify_shape:   Boolean that enables verification of a shape of values.
    allow_broadcast:  Boolean that enables allowing scalars and 1 length vector
        broadcasting. Cannot be true when verify_shape is true.
  Returns:
    A `TensorProto`. Depending on the type, it may contain data in the
    "tensor_content" attribute, which is not directly useful to Python programs.
    To access the values you should convert the proto back to a numpy ndarray
    with `tf.make_ndarray(proto)`.
    If `values` is a `TensorProto`, it is immediately returned; `dtype` and
    `shape` are ignored.
  Raises:
    TypeError:  if unsupported types are provided.
    ValueError: if arguments have inappropriate values or if verify_shape is
     True and shape of values is not equals to a shape from the argument.
  """
  if allow_broadcast and verify_shape:
    raise ValueError("allow_broadcast and verify_shape are not both allowed.")
  if isinstance(values, tensor_pb2.TensorProto):
    return values
  if dtype:
    dtype = dtypes.as_dtype(dtype)
  is_quantized = (
      dtype in [
          dtypes.qint8, dtypes.quint8, dtypes.qint16, dtypes.quint16,
          dtypes.qint32
      ])
  if _is_array_like(values):
    values = np.asarray(values)
  if isinstance(values, (np.ndarray, np.generic)):
    if dtype and dtype.is_numpy_compatible:
      nparray = values.astype(dtype.as_numpy_dtype)
    else:
      nparray = values
  else:
    if values is None:
      raise ValueError("None values not supported.")
    if dtype and dtype.is_numpy_compatible:
      np_dt = dtype.as_numpy_dtype
    else:
      np_dt = None
    if shape is not None and np.prod(shape, dtype=np.int64) == 0:
      nparray = np.empty(shape, dtype=np_dt)
    else:
      _AssertCompatible(values, dtype)
      nparray = np.array(values, dtype=np_dt)
      if (list(nparray.shape) != _GetDenseDimensions(values) and
          not is_quantized):
        raise ValueError(f"Expected values {values} to be a dense tensor with "
                         f"shape {_GetDenseDimensions(values)}, but got shape "
                         f"{list(nparray.shape)}.")
    if (nparray.dtype == np.float64) and dtype is None:
      nparray = nparray.astype(np.float32)
    elif (nparray.dtype == np.int64) and dtype is None:
      downcasted_array = nparray.astype(np.int32)
      if np.array_equal(downcasted_array, nparray):
        nparray = downcasted_array
  numpy_dtype = dtypes.as_dtype(nparray.dtype)
  if numpy_dtype is None:
    raise TypeError(f"Unrecognized data type: {nparray.dtype}.")
  if is_quantized:
    numpy_dtype = dtype
  if dtype is not None and (not hasattr(dtype, "base_dtype") or
                            dtype.base_dtype != numpy_dtype.base_dtype):
    raise TypeError(f"`dtype` {dtype} is not compatible with {values} of "
                    f"dtype {nparray.dtype}.")
  if shape is None:
    shape = nparray.shape
    is_same_size = True
    shape_size = nparray.size
  else:
    shape = [int(dim) for dim in shape]
    shape_size = np.prod(shape, dtype=np.int64)
    is_same_size = shape_size == nparray.size
    if allow_broadcast:
      if nparray.shape == (1,) or nparray.shape == tuple():
        pass
      elif nparray.size != shape_size:
        raise TypeError(f"Expected Tensor's shape: {tuple(shape)}, but got "
                        f"{nparray.shape}.")
    else:
      if verify_shape and nparray.shape != tuple(shape):
        raise TypeError(f"Expected Tensor's shape: {tuple(shape)}, but got "
                        f"{nparray.shape}.")
      if nparray.size > shape_size:
        raise ValueError("Too many elements provided. Takes at most "
                         f"{shape_size:d}, but got {nparray.size:d}.")
  tensor_proto = tensor_pb2.TensorProto(
      dtype=numpy_dtype.as_datatype_enum,
      tensor_shape=tensor_shape.as_shape(shape).as_proto())
  if is_same_size and numpy_dtype in _TENSOR_CONTENT_TYPES and shape_size > 1:
    if nparray.size * nparray.itemsize >= (1 << 31):
      raise ValueError(
          "Cannot create a tensor proto whose content is larger than 2GB.")
    tensor_proto.tensor_content = nparray.tobytes()
    return tensor_proto
  if numpy_dtype == dtypes.string and not isinstance(values, np.ndarray):
    proto_values = _FlattenToStrings(values)
    try:
      str_values = [compat.as_bytes(x) for x in proto_values]
    except TypeError:
      raise TypeError(f"Failed to convert elements of {values} to Tensor. "
                      "Consider casting elements to a supported type. See "
                      "https://www.tensorflow.org/api_docs/python/tf/dtypes "
                      "for supported TF dtypes.")
    tensor_proto.string_val.extend(str_values)
    return tensor_proto
  proto_values = nparray.ravel()
  append_fn = GetNumpyAppendFn(proto_values.dtype)
  if append_fn is None:
    raise TypeError(
        f"Element type not supported in TensorProto: {numpy_dtype.name}.")
  append_fn(tensor_proto, proto_values)
  return tensor_proto
@tf_export("make_ndarray")
def MakeNdarray(tensor):
  """Create a numpy ndarray from a tensor.
  Create a numpy ndarray with the same shape and data as the tensor.
  For example:
  ```python
  a = tf.constant([[1,2,3],[4,5,6]])
  ```
  Args:
    tensor: A TensorProto.
  Returns:
    A numpy array with the tensor contents.
  Raises:
    TypeError: if tensor has unsupported type.
  """
  shape = [d.size for d in tensor.tensor_shape.dim]
  num_elements = np.prod(shape, dtype=np.int64)
  tensor_dtype = dtypes.as_dtype(tensor.dtype)
  dtype = tensor_dtype.as_numpy_dtype
  if tensor.tensor_content:
    return (np.frombuffer(tensor.tensor_content,
                          dtype=dtype).copy().reshape(shape))
  if tensor_dtype == dtypes.string:
    values = list(tensor.string_val)
    padding = num_elements - len(values)
    if padding > 0:
      last = values[-1] if values else ""
      values.extend([last] * padding)
    return np.array(values, dtype=dtype).reshape(shape)
  if tensor_dtype == dtypes.float16 or tensor_dtype == dtypes.bfloat16:
    values = np.fromiter(tensor.half_val, dtype=np.uint16)
    values.dtype = tensor_dtype.as_numpy_dtype
  elif tensor_dtype == dtypes.float32:
    values = np.fromiter(tensor.float_val, dtype=dtype)
  elif tensor_dtype == dtypes.float64:
    values = np.fromiter(tensor.double_val, dtype=dtype)
  elif tensor_dtype in [
      dtypes.int32, dtypes.uint8, dtypes.uint16, dtypes.int16, dtypes.int8,
      dtypes.qint32, dtypes.quint8, dtypes.qint8, dtypes.qint16, dtypes.quint16
  ]:
    values = np.fromiter(tensor.int_val, dtype=dtype)
  elif tensor_dtype == dtypes.int64:
    values = np.fromiter(tensor.int64_val, dtype=dtype)
  elif tensor_dtype == dtypes.uint32:
    values = np.fromiter(tensor.uint32_val, dtype=dtype)
  elif tensor_dtype == dtypes.uint64:
    values = np.fromiter(tensor.uint64_val, dtype=dtype)
  elif tensor_dtype == dtypes.complex64:
    it = iter(tensor.scomplex_val)
    values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
  elif tensor_dtype == dtypes.complex128:
    it = iter(tensor.dcomplex_val)
    values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
  elif tensor_dtype == dtypes.bool:
    values = np.fromiter(tensor.bool_val, dtype=dtype)
  else:
    raise TypeError(f"Unsupported tensor type: {tensor.dtype}. See "
                    "https://www.tensorflow.org/api_docs/python/tf/dtypes "
                    "for supported TF dtypes.")
  if values.size == 0:
    return np.zeros(shape, dtype)
  if values.size != num_elements:
    values = np.pad(values, (0, num_elements - values.size), "edge")
  return values.reshape(shape)
def ShapeEquals(tensor_proto, shape):
  if not isinstance(tensor_proto, tensor_pb2.TensorProto):
    raise TypeError("`tensor_proto` must be a tensor_pb2.TensorProto object, "
                    f"but got type {type(tensor_proto)}.")
  if isinstance(shape, tensor_shape_pb2.TensorShapeProto):
    shape = [d.size for d in shape.dim]
  elif not isinstance(shape, (list, tuple)):
    raise TypeError("`shape` must be a list or tuple, but got type "
                    f"{type(shape)}.")
  tensor_shape_list = [d.size for d in tensor_proto.tensor_shape.dim]
  return all(x == y for x, y in zip(tensor_shape_list, shape))
def _ConstantValue(tensor, partial):
  if not isinstance(tensor, ops.Tensor):
    raise TypeError(f"{tensor!r} must be a Tensor, but got {type(tensor)}.")
  if tensor.op.type == "Const":
    return MakeNdarray(tensor.op.get_attr("value"))
  elif tensor.op.type == "Shape":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.is_fully_defined():
      return np.array(
          [dim.value for dim in input_shape.dims],
          dtype=tensor.dtype.as_numpy_dtype)
    else:
      return None
  elif tensor.op.type == "Size":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.is_fully_defined():
      return np.prod([dim.value for dim in input_shape.dims], dtype=np.int32)
    else:
      return None
  elif tensor.op.type == "Rank":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.ndims is not None:
      return np.ndarray(
          shape=(),
          buffer=np.array([input_shape.ndims], dtype=np.int32),
          dtype=np.int32)
    else:
      return None
  elif tensor.op.type == "Range":
    start = constant_value(tensor.op.inputs[0])
    if start is None:
      return None
    limit = constant_value(tensor.op.inputs[1])
    if limit is None:
      return None
    delta = constant_value(tensor.op.inputs[2])
    if delta is None:
      return None
    return np.arange(start, limit, delta, dtype=tensor.dtype.as_numpy_dtype)
  elif tensor.op.type == "Cast":
    pre_cast = constant_value(tensor.op.inputs[0])
    if pre_cast is None:
      return None
    cast_dtype = dtypes.as_dtype(tensor.op.get_attr("DstT"))
    return pre_cast.astype(cast_dtype.as_numpy_dtype)
  elif tensor.op.type == "Concat":
    dim = constant_value(tensor.op.inputs[0])
    if dim is None:
      return None
    values = []
    for x in tensor.op.inputs[1:]:
      value = constant_value(x)
      if value is None:
        return None
      values.append(value)
    return np.concatenate(values, axis=dim)
  elif tensor.op.type == "ConcatV2":
    dim = constant_value(tensor.op.inputs[-1])
    if dim is None:
      return None
    values = []
    for x in tensor.op.inputs[:-1]:
      value = constant_value(x)
      if value is None:
        return None
      values.append(value)
    return np.concatenate(values, axis=dim)
  elif tensor.op.type == "Pack":
    values = []
    if not tensor.op.inputs:
      return None
    if tensor.op.get_attr("axis") != 0:
      return None
    for x in tensor.op.inputs:
      value = constant_value(x, partial)
      if value is None and not partial:
        return None
      values.append(value)
    return np.array(values)
  elif tensor.op.type == "Unpack":
    if tensor.op.get_attr("axis") != 0:
      return None
    value = constant_value(tensor.op.inputs[0], partial)
    if value is None:
      return None
    return value[tensor.value_index]
  elif tensor.op.type == "Split":
    dim = constant_value(tensor.op.inputs[0])
    value = constant_value(tensor.op.inputs[1], partial)
    if value is None or dim is None:
      return None
    split = np.split(value, tensor.op.get_attr("num_split"), dim)
    return split[tensor.value_index]
  elif tensor.op.type == "Fill":
    fill_shape = tensor.shape
    fill_value = constant_value(tensor.op.inputs[1])
    if fill_shape.is_fully_defined() and fill_value is not None:
      return np.full(fill_shape.as_list(), fill_value, dtype=fill_value.dtype)
    else:
      return None
  elif tensor.op.type == "Equal":
    value1 = constant_value(tensor.op.inputs[0])
    if value1 is None:
      return None
    value2 = constant_value(tensor.op.inputs[1])
    if value2 is None:
      return None
    return np.equal(value1, value2)
  elif tensor.op.type == "NotEqual":
    value1 = constant_value(tensor.op.inputs[0])
    if value1 is None:
      return None
    value2 = constant_value(tensor.op.inputs[1])
    if value2 is None:
      return None
    return np.not_equal(value1, value2)
  elif tensor.op.type == "StopGradient":
    return constant_value(tensor.op.inputs[0], partial)
  elif tensor.op.type in ("CheckNumericsV2", "DebugIdentityV2", "Identity"):
    return constant_value(tensor.op.inputs[0], partial)
  else:
    return None
@tf_export("get_static_value")
  """Returns the constant value of the given tensor, if efficiently calculable.
  This function attempts to partially evaluate the given tensor, and
  returns its value as a numpy ndarray if this succeeds.
  Example usage:
  >>> a = tf.constant(10)
  >>> tf.get_static_value(a)
  10
  >>> b = tf.constant(20)
  >>> tf.get_static_value(tf.add(a, b))
  30
  >>> c = tf.Variable(30)
  >>> print(tf.get_static_value(c))
  None
  Using `partial` option is most relevant when calling `get_static_value` inside
  a `tf.function`. Setting it to `True` will return the results but for the
  values that cannot be evaluated will be `None`. For example:
  ```python
  class Foo(object):
    def __init__(self):
      self.a = tf.Variable(1)
      self.b = tf.constant(2)
    @tf.function
    def bar(self, partial):
      packed = tf.raw_ops.Pack(values=[self.a, self.b])
      static_val = tf.get_static_value(packed, partial=partial)
      tf.print(static_val)
  f = Foo()
  ```
  Compatibility(V1): If `constant_value(tensor)` returns a non-`None` result, it
  will no longer be possible to feed a different value for `tensor`. This allows
  the result of this function to influence the graph that is constructed, and
  permits static shape optimizations.
  Args:
    tensor: The Tensor to be evaluated.
    partial: If True, the returned numpy array is allowed to have partially
      evaluated values. Values that can't be evaluated will be None.
  Returns:
    A numpy ndarray containing the constant value of the given `tensor`,
    or None if it cannot be calculated.
  Raises:
    TypeError: if tensor is not an ops.Tensor.
  """
  if isinstance(tensor, ops.EagerTensor):
    try:
      return tensor.numpy()
    except errors_impl.UnimplementedError:
      return None
  if not is_tensor(tensor):
    return tensor
  if not isinstance(tensor, ops.Tensor):
    return None
  ret = _ConstantValue(tensor, partial)
  if ret is not None:
    tensor.graph.prevent_feeding(tensor)
  return ret
  """A version of `constant_value()` that returns a `TensorShape`.
  This version should be used when a constant tensor value is
  interpreted as a (possibly partial) shape, e.g. in the shape
  function for `tf.reshape()`. By explicitly requesting a
  `TensorShape` as the return value, it is possible to represent
  unknown dimensions; by contrast, `constant_value()` is
  all-or-nothing.
  Args:
    tensor: The rank-0 or rank-1 Tensor to be evaluated.
  Returns:
    A `TensorShape` based on the constant value of the given `tensor`.
  Raises:
    ValueError: If the shape is rank-0 and is not statically known to be -1.
  """
  if isinstance(tensor, ops.EagerTensor):
    return tensor_shape.TensorShape(
        [dim if dim != -1 else None for dim in tensor.numpy()])
  if tensor.get_shape().ndims == 0:
    value = constant_value(tensor)
    if value is None:
      raise ValueError(
          "Received a scalar with unknown value as shape; require a statically "
          "known scalar with value '-1' to describe an unknown shape.")
    if value != -1:
      raise ValueError(
          f"Received a scalar value '{value}' as shape; require a statically "
          "known scalar with value '-1' to describe an unknown shape.")
    return tensor_shape.unknown_shape()
  shape = tensor.get_shape().with_rank(1)
  if shape == [0]:
    return tensor_shape.TensorShape([])
  elif tensor.op.type == "Cast":
    pre_cast = constant_value_as_shape(tensor.op.inputs[0])
    if pre_cast.dims is None:
      return pre_cast
    cast_dtype = dtypes.as_dtype(tensor.op.get_attr("DstT"))
    if cast_dtype not in (dtypes.int32, dtypes.int64):
      return tensor_shape.unknown_shape(shape.dims[0].value)
    dest_dtype_shape_array = np.array(
        [x if x is not None else -1 for x in pre_cast.as_list()]).astype(
            cast_dtype.as_numpy_dtype)
    return tensor_shape.TensorShape([
        x if x >= 0 else None
        for x in dest_dtype_shape_array])
  elif tensor.op.type == "Shape":
    return tensor.op.inputs[0].get_shape()
  elif tensor.op.type == "Pack":
    assert tensor.op.get_attr("axis") == 0
    for pack_input in tensor.op.inputs:
      pack_input_val = constant_value(pack_input)
      if pack_input_val is None or pack_input_val < 0:
        new_dim = tensor_shape.Dimension(None)
      else:
        new_dim = tensor_shape.Dimension(pack_input_val)
      ret = ret.concatenate([new_dim])
    return ret
  elif tensor.op.type == "Concat":
    for concat_input in tensor.op.inputs[1:]:
      ret = ret.concatenate(constant_value_as_shape(concat_input))
    return ret
  elif tensor.op.type == "ConcatV2":
    for concat_input in tensor.op.inputs[:-1]:
      ret = ret.concatenate(constant_value_as_shape(concat_input))
    return ret
  elif tensor.op.type == "StridedSlice":
    try:
      begin = constant_value(tensor.op.inputs[1])
      end = constant_value(tensor.op.inputs[2])
      strides = constant_value(tensor.op.inputs[3])
      if begin is not None and end is not None and strides is not None:
        begin = begin[0]
        end = end[0]
        strides = strides[0]
        begin_mask = tensor.op.get_attr("begin_mask")
        if begin_mask == 1:
          begin = None
        end_mask = tensor.op.get_attr("end_mask")
        if end_mask == 1:
          end = None
        ellipsis_mask = tensor.op.get_attr("ellipsis_mask")
        new_axis_mask = tensor.op.get_attr("new_axis_mask")
        shrink_axis_mask = tensor.op.get_attr("shrink_axis_mask")
        valid_attributes = (not ellipsis_mask and not new_axis_mask and
                            not shrink_axis_mask and (not begin_mask or
                                                      (begin_mask == 1)) and
                            (not end_mask or (end_mask == 1)))
          prev = constant_value_as_shape(tensor.op.inputs[0])
          prev = prev[begin:end:strides]
          ret = tensor_shape.TensorShape(prev)
          return ret
      pass
      pass
  elif (tensor.op.type == "Placeholder" and
        tensor.op.graph.building_function and
        hasattr(tensor.op.graph, "internal_captures")):
    for i, capture in enumerate(tensor.op.graph.internal_captures):
      if capture is tensor:
        external_capture = tensor.op.graph.external_captures[i]
        return constant_value_as_shape(external_capture)
  ret = tensor_shape.unknown_shape(shape.dims[0].value)
  value = constant_value(tensor)
  if value is not None:
    ret = ret.merge_with(
        tensor_shape.TensorShape([d if d >= 0 else None for d in value]))
  return ret
@tf_export("is_tensor")
  """Checks whether `x` is a TF-native type that can be passed to many TF ops.
  Use `is_tensor` to differentiate types that can ingested by TensorFlow ops
  without any conversion (e.g., `tf.Tensor`, `tf.SparseTensor`, and
  `tf.RaggedTensor`) from types that need to be converted into tensors before
  they are ingested (e.g., numpy `ndarray` and Python scalars).
  For example, in the following code block:
  ```python
  if not tf.is_tensor(t):
    t = tf.convert_to_tensor(t)
  return t.shape, t.dtype
  ```
  we check to make sure that `t` is a tensor (and convert it if not) before
  accessing its `shape` and `dtype`.  (But note that not all TensorFlow native
  types have shapes or dtypes; `tf.data.Dataset` is an example of a TensorFlow
  native type that has neither shape nor dtype.)
  Args:
    x: A python object to check.
  Returns:
    `True` if `x` is a TensorFlow-native type.
  """
  return (isinstance(x, internal.NativeObject) or
          isinstance(x, core.Tensor) or
          getattr(x, "is_tensor_like", False))
is_tensor = is_tf_type
  dtype = None
  if isinstance(shape, (tuple, list)):
    if not shape:
      dtype = dtypes.int32
    else:
      shape = tuple(map(tensor_shape.dimension_value, shape))
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")
_ENABLE_MAYBE_SET_STATIC_SHAPE = True
  """Sets the shape of `tensor` to the `shape`'s constant value, if inferrable.
  This is a temporary workaround to fix shape inference across functional op
  boundaries. E.g.
  ```python
  shape = tf.constant([3])
  @tf.function
  def f():
    u = tf.random_uniform(shape)
    return u
  ```
  If we were to rely solely on C++ shape inference, the shape of `u` inside
  `f` would be unknown because C++ shape inference is not aware of the outer
  graph and all it sees is a Placeholder node when backtracing the captured
  tensor for `shape`. `maybe_set_static_shape` computes the static shape value
  of `shape` by traversing the `FuncGraph` boundaries and sets the correct
  shape.
  A longer term solution would be to fix C++ shape inference.
  Args:
    tensor: A tensor.
    shape: A shape tensor.
  """
  if (_ENABLE_MAYBE_SET_STATIC_SHAPE and not context.executing_eagerly() and
      ops.get_default_graph().building_function and
      not tensor.shape.is_fully_defined() and is_tensor(shape)):
    shape = shape_tensor(shape)
    const_shape = constant_value_as_shape(shape)
    tensor.set_shape(const_shape)
