
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export("__operators__.ragged_getitem", v1=[])
@dispatch.add_dispatch_support
def ragged_tensor_getitem(rt_input, key):
  """Returns the specified piece of this RaggedTensor.
  Supports multidimensional indexing and slicing, with one restriction:
  indexing into a ragged inner dimension is not allowed.  This case is
  problematic because the indicated value may exist in some rows but not
  others.  In such cases, it's not obvious whether we should (1) report an
  IndexError; (2) use a default value; or (3) skip that value and return a
  tensor with fewer rows than we started with.  Following the guiding
  principles of Python ("In the face of ambiguity, refuse the temptation to
  guess"), we simply disallow this operation.
  Args:
    rt_input: The RaggedTensor to slice.
    key: Indicates which piece of the RaggedTensor to return, using standard
      Python semantics (e.g., negative values index from the end).  `key`
      may have any of the following types:
      * `int` constant
      * Scalar integer `Tensor`
      * `slice` containing integer constants and/or scalar integer
        `Tensor`s
      * `Ellipsis`
      * `tf.newaxis`
      * `tuple` containing any of the above (for multidimensional indexing)
  Returns:
    A `Tensor` or `RaggedTensor` object.  Values that include at least one
    ragged dimension are returned as `RaggedTensor`.  Values that include no
    ragged dimensions are returned as `Tensor`.  See above for examples of
    expressions that return `Tensor`s vs `RaggedTensor`s.
  Raises:
    ValueError: If `key` is out of bounds.
    ValueError: If `key` is not supported.
    TypeError: If the indices in `key` have an unsupported type.
  Examples:
  >>> rt = tf.ragged.constant([['a', 'b', 'c'], ['d', 'e'], ['f'], ['g']])
  array([b'a', b'b', b'c'], dtype=object)
  [[b'a', b'b', b'c'], [b'd', b'e'], [b'f']]
  b'g'
  >>> rt = tf.ragged.constant([[[1, 2, 3], [4]],
  ...                          [[5], [], [6]],
  ...                          [[7]],
  ...                          [[8, 9], [10]]])
  [[5], [], [6]]
  array([8, 9], dtype=int32)
  [[[4]], [[], [6]], [], [[10]]]
  [[[4]], [[6]], [[7]], [[10]]]
  """
  if not isinstance(rt_input, ragged_tensor.RaggedTensor):
    raise TypeError("Ragged __getitem__ expects a ragged_tensor.")
  scope_tensors = [rt_input] + list(_tensors_in_key_list(key))
  if isinstance(key, (list, tuple)):
    key = list(key)
  else:
    key = [key]
  with ops.name_scope(None, "RaggedGetItem", scope_tensors):
    return _ragged_getitem(rt_input, key)
def _ragged_getitem(rt_input, key_list):
  """Helper for indexing and slicing ragged tensors with __getitem__().
  Extracts the specified piece of the `rt_input`.  See
  `RaggedTensor.__getitem__` for examples and restrictions.
  Args:
    rt_input: The `RaggedTensor` from which a piece should be returned.
    key_list: The list of keys specifying which piece to return. Each key
      corresponds with a separate dimension.
  Returns:
    The indicated piece of rt_input.
  Raises:
    ValueError: If `key_list` is not supported.
    TypeError: If any keys in `key_list` have an unsupported type.
  """
  if not key_list:
    return rt_input
  row_key = key_list[0]
  inner_keys = key_list[1:]
  if row_key is Ellipsis:
    expanded_key_list = _expand_ellipsis(key_list, rt_input.shape.ndims)
    return _ragged_getitem(rt_input, expanded_key_list)
  if row_key is array_ops.newaxis:
    inner_rt = _ragged_getitem(rt_input, inner_keys)
    nsplits = tensor_shape.dimension_at_index(inner_rt.row_splits.shape, 0)
    if nsplits.value is not None:
      nsplits = nsplits.value
    else:
      nsplits = array_ops.shape(inner_rt.row_splits,
                                out_type=inner_rt.row_splits.dtype)[0]
    return ragged_tensor.RaggedTensor.from_uniform_row_length(
        inner_rt, nsplits - 1, nrows=1, validate=False)
  if isinstance(row_key, slice):
    sliced_rt_input = _slice_ragged_row_dimension(rt_input, row_key)
    if rt_input.uniform_row_length is not None:
      sliced_rt_input = ragged_tensor.RaggedTensor.from_uniform_row_length(
          sliced_rt_input.values, rt_input.uniform_row_length,
          nrows=sliced_rt_input.nrows())
    return _ragged_getitem_inner_dimensions(sliced_rt_input, inner_keys)
  else:
    starts = rt_input.row_splits[:-1]
    limits = rt_input.row_splits[1:]
    if context.executing_eagerly():
      try:
        if int(row_key) >= len(starts):
          raise IndexError("Row key {} out of bounds".format(row_key))
      except (TypeError, ValueError):
        pass
    row = rt_input.values[starts[row_key]:limits[row_key]]
    return row.__getitem__(inner_keys)
def _slice_ragged_row_dimension(rt_input, row_key):
  if row_key.start is None and row_key.stop is None and row_key.step is None:
    return rt_input
  new_starts = rt_input.row_splits[:-1][row_key]
  new_limits = rt_input.row_splits[1:][row_key]
  zero_pad = array_ops.zeros([1], rt_input.row_splits.dtype)
  if row_key.step is None or row_key.step == 1:
    new_splits = array_ops.concat(
        [zero_pad[array_ops.size(new_starts):], new_starts[:1], new_limits],
        axis=0)
    values_start = new_splits[0]
    values_limit = new_splits[-1]
    return ragged_tensor.RaggedTensor.from_row_splits(
        rt_input.values[values_start:values_limit], new_splits - values_start,
        validate=False)
  else:
    return _build_ragged_tensor_from_value_ranges(new_starts, new_limits, 1,
                                                  rt_input.values)
def _ragged_getitem_inner_dimensions(rt_input, key_list):
  if not key_list:
    return rt_input
  if isinstance(rt_input, ops.Tensor):
    return rt_input.__getitem__([slice(None, None, None)] + key_list)
  column_key = key_list[0]
  if column_key is Ellipsis:
    expanded_key_list = _expand_ellipsis(key_list, rt_input.values.shape.ndims)
    return _ragged_getitem_inner_dimensions(rt_input, expanded_key_list)
  if column_key is array_ops.newaxis:
    inner_rt = _ragged_getitem_inner_dimensions(rt_input, key_list[1:])
    nsplits = tensor_shape.dimension_at_index(inner_rt.row_splits.shape, 0)
    if nsplits.value is not None:
      nsplits = nsplits.value
    else:
      nsplits = array_ops.shape(inner_rt.row_splits,
                                out_type=inner_rt.row_splits.dtype)[0]
    return ragged_tensor.RaggedTensor.from_uniform_row_length(
        inner_rt, 1, nrows=nsplits - 1, validate=False)
  if isinstance(column_key, slice):
    if (column_key.start is None and column_key.stop is None and
        column_key.step is None):
      return rt_input.with_values(
          _ragged_getitem_inner_dimensions(rt_input.values, key_list[1:]))
    else:
      if not (isinstance(column_key.start, (ops.Tensor, int, type(None))) and
              isinstance(column_key.stop, (ops.Tensor, int, type(None)))):
        raise TypeError("slice offsets must be integers or None")
      starts = rt_input.row_splits[:-1]
      limits = rt_input.row_splits[1:]
      step = 1 if column_key.step is None else column_key.step
      lower_bound = _if_ge_zero(step, lambda: starts, lambda: starts - 1)
      upper_bound = _if_ge_zero(step, lambda: limits, lambda: limits - 1)
      if column_key.start is None:
        inner_rt_starts = _if_ge_zero(step, lambda: starts, lambda: limits - 1)
      else:
        start_offset = math_ops.cast(column_key.start, starts.dtype)
        inner_rt_starts = _if_ge_zero(
            column_key.start,
            lambda: math_ops.minimum(starts + start_offset, upper_bound),
            lambda: math_ops.maximum(limits + start_offset, lower_bound))
      if column_key.stop is None:
        inner_rt_limits = _if_ge_zero(step, lambda: limits, lambda: starts - 1)
      else:
        stop_offset = math_ops.cast(column_key.stop, starts.dtype)
        inner_rt_limits = _if_ge_zero(
            column_key.stop,
            lambda: math_ops.minimum(starts + stop_offset, upper_bound),
            lambda: math_ops.maximum(limits + stop_offset, lower_bound))
      inner_rt = _build_ragged_tensor_from_value_ranges(
          inner_rt_starts, inner_rt_limits, column_key.step, rt_input.values)
      if rt_input.uniform_row_length is not None:
        new_row_length = _slice_length(rt_input.uniform_row_length, column_key)
        inner_rt = ragged_tensor.RaggedTensor.from_uniform_row_length(
            inner_rt.values, new_row_length, rt_input.nrows())
      return inner_rt.with_values(
          _ragged_getitem_inner_dimensions(inner_rt.values, key_list[1:]))
  if rt_input.uniform_row_length is None:
    raise ValueError("Cannot index into an inner ragged dimension.")
  row_length = rt_input.uniform_row_length
  column_key = math_ops.cast(column_key, row_length.dtype)
  oob_err_msg = "Index out of bounds when indexing into a ragged tensor"
  oob_checks = [
      check_ops.assert_greater_equal(
          column_key, -row_length, message=oob_err_msg),
      check_ops.assert_less(column_key, row_length, message=oob_err_msg),
  ]
  with ops.control_dependencies(oob_checks):
    offset = _if_ge_zero(column_key, lambda: column_key,
                         lambda: row_length + column_key)
    sliced_rt = rt_input.values[offset::row_length]
    return _ragged_getitem_inner_dimensions(sliced_rt, key_list[1:])
def _slice_length(value_length, slice_key):
  """Computes the number of elements in a slice of a value with a given length.
  Returns the equivalent of: `len(range(value_length)[slice_key])`
  Args:
    value_length: Scalar int `Tensor`: the length of the value being sliced.
    slice_key: A `slice` object used to slice elements from the value.
  Returns:
    The number of elements in the sliced value.
  """
  zeros = array_ops.zeros(value_length, dtype=dtypes.bool)
  return array_ops.size(zeros[slice_key], out_type=value_length.dtype)
def _expand_ellipsis(key_list, num_remaining_dims):
  """Expands the ellipsis at the start of `key_list`.
  Assumes that the first element of `key_list` is Ellipsis.  This will either
  remove the Ellipsis (if it corresponds to zero indices) or prepend a new
  `slice(None, None, None)` (if it corresponds to more than zero indices).
  Args:
    key_list: The arguments to `__getitem__()`.
    num_remaining_dims: The number of dimensions remaining.
  Returns:
    A copy of `key_list` with he ellipsis expanded.
  Raises:
    ValueError: If ragged_rank.shape.ndims is None
    IndexError: If there are too many elements in `key_list`.
  """
  if num_remaining_dims is None:
    raise ValueError("Ellipsis not supported for unknown shape RaggedTensors")
  num_indices = sum(1 for idx in key_list if idx is not array_ops.newaxis)
  if num_indices > num_remaining_dims + 1:
    raise IndexError("Too many indices for RaggedTensor")
  elif num_indices == num_remaining_dims + 1:
    return key_list[1:]
  else:
    return [slice(None, None, None)] + key_list
def _tensors_in_key_list(key_list):
  if isinstance(key_list, ops.Tensor):
    yield key_list
  if isinstance(key_list, (list, tuple)):
    for v in key_list:
      for tensor in _tensors_in_key_list(v):
        yield tensor
  if isinstance(key_list, slice):
    for tensor in _tensors_in_key_list(key_list.start):
      yield tensor
    for tensor in _tensors_in_key_list(key_list.stop):
      yield tensor
    for tensor in _tensors_in_key_list(key_list.step):
      yield tensor
def _build_ragged_tensor_from_value_ranges(starts, limits, step, values):
  if step is None:
    step = 1
  step = ops.convert_to_tensor(step, name="step")
  if step.dtype.is_integer:
    step = math_ops.cast(step, starts.dtype)
  else:
    raise TypeError("slice strides must be integers or None")
  value_indices = ragged_math_ops.range(starts, limits, step,
                                        row_splits_dtype=starts.dtype)
  if isinstance(values, ragged_tensor.RaggedTensor):
    gathered_values = ragged_gather_ops.gather(
        params=values, indices=value_indices.values)
  else:
    gathered_values = array_ops.gather(
        params=values, indices=value_indices.values)
  return value_indices.with_values(gathered_values)
def _if_ge_zero(value, true_fn, false_fn):
  if isinstance(value, ops.Tensor):
    const_value = tensor_util.constant_value(value)
    if const_value is None:
      return control_flow_ops.cond(value >= 0, true_fn, false_fn)
    else:
      value = const_value
  if value >= 0:
    return true_fn()
  else:
    return false_fn()
