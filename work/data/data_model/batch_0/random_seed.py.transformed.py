
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def get_seed(seed):
  seed, seed2 = random_seed.get_seed(seed)
  if seed is None:
    seed = constant_op.constant(0, dtype=dtypes.int64, name="seed")
  else:
    seed = ops.convert_to_tensor(seed, dtype=dtypes.int64, name="seed")
  if seed2 is None:
    seed2 = constant_op.constant(0, dtype=dtypes.int64, name="seed2")
  else:
    with ops.name_scope("seed2") as scope:
      seed2 = ops.convert_to_tensor(seed2, dtype=dtypes.int64)
      seed2 = array_ops.where_v2(
          math_ops.logical_and(
              math_ops.equal(seed, 0), math_ops.equal(seed2, 0)),
          constant_op.constant(2**31 - 1, dtype=dtypes.int64),
          seed2,
          name=scope)
  return seed, seed2
