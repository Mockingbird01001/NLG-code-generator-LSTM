
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
def assert_splits_match(nested_splits_lists):
  error_msg = "Inputs must have identical ragged splits"
  for splits_list in nested_splits_lists:
    if len(splits_list) != len(nested_splits_lists[0]):
      raise ValueError(error_msg)
  return [
      check_ops.assert_equal(s1, s2, message=error_msg)
      for splits_list in nested_splits_lists[1:]
      for (s1, s2) in zip(nested_splits_lists[0], splits_list)
  ]
get_positive_axis = array_ops.get_positive_axis
convert_to_int_tensor = array_ops.convert_to_int_tensor
repeat = array_ops.repeat_with_axis
def lengths_to_splits(lengths):
  return array_ops.concat([[0], math_ops.cumsum(lengths)], axis=-1)
def repeat_ranges(params, splits, repeats):
  """Repeats each range of `params` (as specified by `splits`) `repeats` times.
  Let the `i`th range of `params` be defined as
  `params[splits[i]:splits[i + 1]]`.  Then this function returns a tensor
  containing range 0 repeated `repeats[0]` times, followed by range 1 repeated
  `repeats[1]`, ..., followed by the last range repeated `repeats[-1]` times.
  Args:
    params: The `Tensor` whose values should be repeated.
    splits: A splits tensor indicating the ranges of `params` that should be
      repeated.
    repeats: The number of times each range should be repeated.  Supports
      broadcasting from a scalar value.
  Returns:
    A `Tensor` with the same rank and type as `params`.
  >>> print(repeat_ranges(
  ...     params=tf.constant(['a', 'b', 'c']),
  ...     splits=tf.constant([0, 2, 3]),
  ...     repeats=tf.constant(3)))
  tf.Tensor([b'a' b'b' b'a' b'b' b'a' b'b' b'c' b'c' b'c'],
      shape=(9,), dtype=string)
  """
  if repeats.shape.ndims != 0:
    repeated_starts = repeat(splits[:-1], repeats, axis=0)
    repeated_limits = repeat(splits[1:], repeats, axis=0)
  else:
    repeated_splits = repeat(splits, repeats, axis=0)
    n_splits = array_ops.shape(repeated_splits, out_type=repeats.dtype)[0]
    repeated_starts = repeated_splits[:n_splits - repeats]
    repeated_limits = repeated_splits[repeats:]
  one = array_ops.ones((), repeated_starts.dtype)
  offsets = gen_ragged_math_ops.ragged_range(
      repeated_starts, repeated_limits, one)
  return array_ops.gather(params, offsets.rt_dense_values)
