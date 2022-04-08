
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
def alias_tensors(*args):
  def alias_if_tensor(a):
    return array_ops.identity(a) if isinstance(a, ops.Tensor) else a
  if len(args) > 1:
    return (alias_if_tensor(a) for a in args)
  elif len(args) == 1:
    return alias_if_tensor(args[0])
  raise ValueError('at least one argument required')
def get_range_len(start, limit, delta):
  dist = ops.convert_to_tensor(limit - start)
  unadjusted_len = dist // delta
  adjustment = math_ops.cast(
      gen_math_ops.not_equal(dist % delta,
                             array_ops.zeros_like(unadjusted_len)), dist.dtype)
  final_len = unadjusted_len + adjustment
  return gen_math_ops.maximum(final_len, array_ops.zeros_like(final_len))
