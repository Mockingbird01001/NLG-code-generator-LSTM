
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export("ragged.constant")
@dispatch.add_dispatch_support
def constant(pylist, dtype=None, ragged_rank=None, inner_shape=None,
             name=None, row_splits_dtype=dtypes.int64):
  """Constructs a constant RaggedTensor from a nested Python list.
  Example:
  >>> tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
  <tf.RaggedTensor [[1, 2], [3], [4, 5, 6]]>
  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensor` will have rank `K`.  If `pylist` contains no scalar
  values, then `K` is one greater than the maximum depth of empty lists in
  `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.
  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list`, `tuple` or `np.ndarray` must be a scalar value
      compatible with `dtype`.
    dtype: The type of elements for the returned `RaggedTensor`.  If not
      specified, then a default is chosen based on the scalar values in
      `pylist`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensor`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to
      `max(0, K - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensor`.  Defaults to `()` if `ragged_rank`
      is not specified.  If `ragged_rank` is specified, then a default is chosen
      based on the contents of `pylist`.
    name: A name prefix for the returned tensor (optional).
    row_splits_dtype: data type for the constructed `RaggedTensor`'s row_splits.
      One of `tf.int32` or `tf.int64`.
  Returns:
    A potentially ragged tensor with rank `K` and the specified `ragged_rank`,
    containing the values from `pylist`.
  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  def ragged_factory(values, row_splits):
    row_splits = constant_op.constant(row_splits, dtype=row_splits_dtype)
    return ragged_tensor.RaggedTensor.from_row_splits(values, row_splits,
                                                      validate=False)
  with ops.name_scope(name, "RaggedConstant"):
    return _constant_value(ragged_factory, constant_op.constant, pylist, dtype,
                           ragged_rank, inner_shape)
@tf_export(v1=["ragged.constant_value"])
@dispatch.add_dispatch_support
def constant_value(pylist, dtype=None, ragged_rank=None, inner_shape=None,
                   row_splits_dtype="int64"):
  """Constructs a RaggedTensorValue from a nested Python list.
  Warning: This function returns a `RaggedTensorValue`, not a `RaggedTensor`.
  If you wish to construct a constant `RaggedTensor`, use
  [`ragged.constant(...)`](constant.md) instead.
  Example:
  >>> tf.compat.v1.ragged.constant_value([[1, 2], [3], [4, 5, 6]])
  tf.RaggedTensorValue(values=array([1, 2, 3, 4, 5, 6]),
                       row_splits=array([0, 2, 3, 6]))
  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensorValue` will have rank `K`.  If `pylist` contains no
  scalar values, then `K` is one greater than the maximum depth of empty lists
  in `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.
  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list` or `tuple` must be a scalar value compatible with `dtype`.
    dtype: `numpy.dtype`.  The type of elements for the returned `RaggedTensor`.
      If not specified, then a default is chosen based on the scalar values in
      `pylist`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensorValue`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K
      - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensorValue`.  Defaults to `()` if
      `ragged_rank` is not specified.  If `ragged_rank` is specified, then a
      default is chosen based on the contents of `pylist`.
    row_splits_dtype: data type for the constructed `RaggedTensorValue`'s
      row_splits.  One of `numpy.int32` or `numpy.int64`.
  Returns:
    A `tf.RaggedTensorValue` or `numpy.array` with rank `K` and the specified
    `ragged_rank`, containing the values from `pylist`.
  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  if dtype is not None and isinstance(dtype, dtypes.DType):
    dtype = dtype.as_numpy_dtype
  row_splits_dtype = dtypes.as_dtype(row_splits_dtype).as_numpy_dtype
  def _ragged_factory(values, row_splits):
    row_splits = np.array(row_splits, dtype=row_splits_dtype)
    return ragged_tensor_value.RaggedTensorValue(values, row_splits)
    return np.reshape(np.array(pylist, dtype=dtype), shape)
  return _constant_value(_ragged_factory, _inner_factory, pylist, dtype,
                         ragged_rank, inner_shape)
def _constant_value(ragged_factory, inner_factory, pylist, dtype, ragged_rank,
                    inner_shape):
  """Constructs a constant RaggedTensor or RaggedTensorValue.
  Args:
    ragged_factory: A factory function with the signature:
      `ragged_factory(values, row_splits)`
    inner_factory: A factory function with the signature: `inner_factory(pylist,
      dtype, shape, name)`
    pylist: A nested `list`, `tuple` or `np.ndarray`.
    dtype: Data type for returned value.
    ragged_rank: Ragged rank for returned value.
    inner_shape: Inner value shape for returned value.
  Returns:
    A value returned by `ragged_factory` or `inner_factory`.
  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  if ragged_tensor.is_ragged(pylist):
    raise TypeError("pylist may not be a RaggedTensor or RaggedTensorValue.")
  if not isinstance(pylist, (list, tuple)) and np.ndim(pylist) == 0:
    if ragged_rank is not None and ragged_rank != 0:
      raise ValueError("Invalid pylist=%r: incompatible with ragged_rank=%d" %
                       (pylist, ragged_rank))
    if inner_shape is not None and inner_shape:
      raise ValueError(
          "Invalid pylist=%r: incompatible with dim(inner_shape)=%d" %
          (pylist, len(inner_shape)))
    return inner_factory(pylist, dtype, ())
  if ragged_rank is not None and ragged_rank < 0:
    raise ValueError(
        "Invalid ragged_rank=%r: must be nonnegative" % ragged_rank)
  scalar_depth, max_depth = _find_scalar_and_max_depth(pylist)
  if scalar_depth is not None:
    if max_depth > scalar_depth:
      raise ValueError("Invalid pylist=%r: empty list nesting is greater "
                       "than scalar value nesting" % pylist)
  if inner_shape is not None and ragged_rank is not None:
    expected_depth = ragged_rank + len(inner_shape) + 1
    if ((scalar_depth is not None and expected_depth != scalar_depth) or
        (scalar_depth is None and expected_depth < max_depth)):
      raise ValueError(
          "Invalid pylist=%r: incompatible with ragged_rank=%d "
          "and dim(inner_shape)=%d" % (pylist, ragged_rank, len(inner_shape)))
  if (ragged_rank == 0 or
      (ragged_rank is None and
       ((max_depth < 2) or
        (inner_shape is not None and max_depth - len(inner_shape) < 2)))):
    return inner_factory(pylist, dtype, inner_shape)
  if inner_shape is None:
    if ragged_rank is None:
      inner_shape = ()
    else:
      inner_shape = _default_inner_shape_for_pylist(pylist, ragged_rank)
  if ragged_rank is None:
    if scalar_depth is None:
      ragged_rank = max(1, max_depth - 1)
    else:
      ragged_rank = max(1, scalar_depth - 1 - len(inner_shape))
  nested_splits = []
  values = pylist
  for dim in range(ragged_rank):
    nested_splits.append([0])
    concatenated_values = []
    for row in values:
      nested_splits[dim].append(nested_splits[dim][-1] + len(row))
      concatenated_values.extend(row)
    values = concatenated_values
  values = inner_factory(
      values, dtype=dtype, shape=(len(values),) + inner_shape, name="values")
  for row_splits in reversed(nested_splits):
    values = ragged_factory(values, row_splits)
  return values
def _find_scalar_and_max_depth(pylist):
  """Finds nesting depth of scalar values in pylist.
  Args:
    pylist: A nested python `list` or `tuple`.
  Returns:
    A tuple `(scalar_depth, max_depth)`.  `scalar_depth` is the nesting
    depth of scalar values in `pylist`, or `None` if `pylist` contains no
    scalars.  `max_depth` is the maximum depth of `pylist` (including
    empty lists).
  Raises:
    ValueError: If pylist has inconsistent nesting depths for scalars.
  """
  if isinstance(pylist, (list, tuple)) or np.ndim(pylist) != 0:
    scalar_depth = None
    max_depth = 1
    for child in pylist:
      child_scalar_depth, child_max_depth = _find_scalar_and_max_depth(child)
      if child_scalar_depth is not None:
        if scalar_depth is not None and scalar_depth != child_scalar_depth + 1:
          raise ValueError("all scalar values must have the same nesting depth")
        scalar_depth = child_scalar_depth + 1
      max_depth = max(max_depth, child_max_depth + 1)
    return (scalar_depth, max_depth)
  return (0, 0)
def _default_inner_shape_for_pylist(pylist, ragged_rank):
  def get_inner_shape(item):
    if not isinstance(item, (list, tuple)) and np.ndim(item) == 0:
      return ()
      return (len(item),) + get_inner_shape(item[0])
    return (0,)
  def check_inner_shape(item, shape):
    is_nested = isinstance(item, (list, tuple)) or np.ndim(item) != 0
    if is_nested != bool(shape):
      raise ValueError("inner values have inconsistent shape")
    if is_nested:
      if shape[0] != len(item):
        raise ValueError("inner values have inconsistent shape")
      for child in item:
        check_inner_shape(child, shape[1:])
  flat_values = pylist
  for dim in range(ragged_rank):
    if not all(
        isinstance(v, (list, tuple)) or np.ndim(v) != 0 for v in flat_values):
      raise ValueError("pylist has scalar values depth %d, but ragged_rank=%d "
                       "requires scalar value depth greater than %d" %
                       (dim + 1, ragged_rank, ragged_rank))
    flat_values = sum((list(v) for v in flat_values), [])
  inner_shape = get_inner_shape(flat_values)
  check_inner_shape(flat_values, inner_shape)
  return inner_shape[1:]
@tf_export(v1=["ragged.placeholder"])
@dispatch.add_dispatch_support
def placeholder(dtype, ragged_rank, value_shape=None, name=None):
  """Creates a placeholder for a `tf.RaggedTensor` that will always be fed.
  **Important**: This ragged tensor will produce an error if evaluated.
  Its value must be fed using the `feed_dict` optional argument to
  `Session.run()`, `Tensor.eval()`, or `Operation.run()`.
  @compatibility{eager} Placeholders are not compatible with eager execution.
  Args:
    dtype: The data type for the `RaggedTensor`.
    ragged_rank: The ragged rank for the `RaggedTensor`
    value_shape: The shape for individual flat values in the `RaggedTensor`.
    name: A name for the operation (optional).
  Returns:
    A `RaggedTensor` that may be used as a handle for feeding a value, but
    not evaluated directly.
  Raises:
    RuntimeError: if eager execution is enabled
  """
  if ragged_rank == 0:
    return array_ops.placeholder(dtype, value_shape, name)
  with ops.name_scope(name, "RaggedPlaceholder", []):
    flat_shape = tensor_shape.TensorShape([None]).concatenate(value_shape)
    result = array_ops.placeholder(dtype, flat_shape, "flat_values")
    for i in reversed(range(ragged_rank)):
      row_splits = array_ops.placeholder(dtypes.int64, [None],
                                         "row_splits_%d" % i)
      result = ragged_tensor.RaggedTensor.from_row_splits(result, row_splits,
                                                          validate=False)
    return result
