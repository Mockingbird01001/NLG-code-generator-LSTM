
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
__all__ = ["LinearOperatorPermutation",]
@tf_export("linalg.LinearOperatorPermutation")
@linear_operator.make_composite_tensor
class LinearOperatorPermutation(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] of permutation matrices.
  This operator acts like a [batch] of permutations with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.
  `LinearOperatorPermutation` is initialized with a (batch) vector.
  A permutation, is defined by an integer vector `v` whose values are unique
  and are in the range `[0, ... n]`. Applying the permutation on an input
  matrix has the folllowing meaning: the value of `v` at index `i`
  says to move the `v[i]`-th row of the input matrix to the `i`-th row.
  Because all values are unique, this will result in a permutation of the
  rows the input matrix. Note, that the permutation vector `v` has the same
  semantics as `tf.transpose`.
  ```python
  vec = [0, 2, 1]
  operator = LinearOperatorPermutation(vec)
  operator.to_dense()
  ==> [[1., 0., 0.]
       [0., 0., 1.]
       [0., 1., 0.]]
  operator.shape
  ==> [3, 3]
  operator.log_abs_determinant()
  ==> scalar Tensor
  x = ... Shape [3, 4] Tensor
  operator.matmul(x)
  ==> Shape [3, 4] Tensor
  ```
  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if
  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```
  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """
  def __init__(self,
               perm,
               dtype=dtypes.float32,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorPermutation"):
    parameters = dict(
        perm=perm,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    with ops.name_scope(name, values=[perm]):
      self._perm = linear_operator_util.convert_nonref_to_tensor(
          perm, name="perm")
      self._check_perm(self._perm)
        raise ValueError(f"A Permutation operator is always non-singular. "
                         f"Expected argument `is_non_singular` to be True. "
                         f"Received: {is_non_singular}.")
        raise ValueError(f"A Permutation operator is always square. "
                         f"Expected argument `is_square` to be True. "
                         f"Received: {is_square}.")
      is_square = True
      super(LinearOperatorPermutation, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)
  def _check_perm(self, perm):
    if (perm.shape.ndims is not None and perm.shape.ndims < 1):
      raise ValueError(f"Argument `perm` must have at least 1 dimension. "
                       f"Received: {perm}.")
    if not perm.dtype.is_integer:
      raise TypeError(f"Argument `perm` must be integer dtype. "
                      f"Received: {perm}.")
    static_perm = tensor_util.constant_value(perm)
    if static_perm is not None:
      sorted_perm = np.sort(static_perm, axis=-1)
      if np.any(sorted_perm != np.arange(0, static_perm.shape[-1])):
        raise ValueError(
            f"Argument `perm` must be a vector of unique integers from "
            f"0 to {static_perm.shape[-1] - 1}.")
  def _shape(self):
    perm_shape = self._perm.shape
    return perm_shape.concatenate(perm_shape[-1:])
  def _shape_tensor(self):
    perm_shape = array_ops.shape(self._perm)
    k = perm_shape[-1]
    return array_ops.concat((perm_shape, [k]), 0)
  def _assert_non_singular(self):
    return control_flow_ops.no_op("assert_non_singular")
  def _domain_dimension_tensor(self, perm=None):
    perm = perm if perm is not None else self.perm
    return array_ops.shape(perm)[-1]
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    perm = ops.convert_to_tensor_v2_with_dispatch(self.perm)
    if adjoint and not self.is_self_adjoint:
      perm = sort_ops.argsort(perm, axis=-1)
    x = linalg.adjoint(x) if adjoint_arg else x
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(x)[:-1], array_ops.shape(perm))
    k = array_ops.shape(x)[-1]
    broadcast_x_shape = array_ops.concat([broadcast_shape, [k]], axis=-1)
    x = array_ops.broadcast_to(x, broadcast_x_shape)
    perm = array_ops.broadcast_to(perm, broadcast_shape)
    m = array_ops.shape(x)[-2]
    x = array_ops.reshape(x, [-1, m, k])
    perm = array_ops.reshape(perm, [-1, m])
    y = array_ops.gather(x, perm, axis=-2, batch_dims=1)
    return array_ops.reshape(y, broadcast_x_shape)
  def _log_abs_determinant(self):
    return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)
  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    return self._matmul(rhs, adjoint=(not adjoint), adjoint_arg=adjoint_arg)
  def _to_dense(self):
    perm = ops.convert_to_tensor_v2_with_dispatch(self.perm)
    return math_ops.cast(math_ops.equal(
        math_ops.range(0, self._domain_dimension_tensor(perm)),
        perm[..., array_ops.newaxis]), self.dtype)
  def _diag_part(self):
    perm = ops.convert_to_tensor_v2_with_dispatch(self.perm)
    return math_ops.cast(math_ops.equal(
        math_ops.range(0, self._domain_dimension_tensor(perm)),
        perm), self.dtype)
  def _cond(self):
    return array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)
  @property
  def perm(self):
    return self._perm
  @property
  def _composite_tensor_fields(self):
    return ("perm", "dtype")
