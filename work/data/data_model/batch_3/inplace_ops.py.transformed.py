
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
def _inplace_helper(x, i, v, op):
  """Applies an inplace op on (x, i, v).
  op is one of gen_array_ops.alias_inplace_update,
  gen_array_ops.alias_inplace_add, or gen_array_ops.alias_inplace_sub.
  If i is None, x and v must be the same shape. Computes
    x op v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    x[i, :] op v;
  Otherwise, x and v must have the same rank. Computes
    x[i, :] op v;
  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.
    op: alias_inplace_update, alias_inplace_add, or alias_inplace_sub.
  Returns:
    Returns x.
  """
  x = ops.convert_to_tensor(x)
  v = ops.convert_to_tensor(v, x.dtype)
  if i is None:
    return array_ops.reshape(
        op(array_ops.reshape(x, [1, -1]), [0], array_ops.reshape(v, [1, -1])),
        array_ops.shape(x))
  i = math_ops.cast(i, dtypes.int32)
  if i.get_shape().ndims == 0:
    return op(x, array_ops.reshape(i, [1]), array_ops.expand_dims(v, 0))
  return op(x, i, v)
@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_update, which offers the same functionality '
     'with well-defined read-write semantics.'))
def alias_inplace_update(x, i, v):
  return _inplace_helper(x, i, v, gen_array_ops.inplace_update)
@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_add, which offers the same functionality '
     'with well-defined read-write semantics.'))
def alias_inplace_add(x, i, v):
  return _inplace_helper(x, i, v, gen_array_ops.inplace_add)
@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_sub, which offers the same functionality '
     'with well-defined read-write semantics.'))
def alias_inplace_sub(x, i, v):
  return _inplace_helper(x, i, v, gen_array_ops.inplace_sub)
def empty_like(x, init=None):
  """Returns a non-initialized tensor with the same shape and dtype as x.
  Args:
    x: A Tensor.
    init: Initialize the returned tensor with the default value of
      x.dtype(), if True. Otherwise, do not initialize. Defaults to
      None.
  Returns:
    A tensor y, whose dtype and shape are the same as those of x.
    y is guaranteed not to be an alias of x. Upon return, y may contain
    arbitrary data.
  """
  x = ops.convert_to_tensor(x)
  return gen_array_ops.empty(array_ops.shape(x), x.dtype, init=init)
@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_update, which offers the same functionality '
     'with well-defined read-write semantics.'))
def inplace_update(x, i, v):
  return alias_inplace_update(gen_array_ops.deep_copy(x), i, v)
@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_add, which offers the same functionality '
     'with well-defined read-write semantics.'))
def inplace_add(x, i, v):
  return alias_inplace_add(gen_array_ops.deep_copy(x), i, v)
@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_sub, which offers the same functionality '
     'with well-defined read-write semantics.'))
def inplace_sub(x, i, v):
  return alias_inplace_sub(gen_array_ops.deep_copy(x), i, v)
empty = gen_array_ops.empty
