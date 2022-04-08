
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
__all__ = ["LinearOperatorDiag",]
@tf_export("linalg.LinearOperatorDiag")
@linear_operator.make_composite_tensor
class LinearOperatorDiag(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] square diagonal matrix.
  This operator acts like a [batch] diagonal matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.
  `LinearOperatorDiag` is initialized with a (batch) vector.
  ```python
  diag = [1., -1.]
  operator = LinearOperatorDiag(diag)
  operator.to_dense()
  ==> [[1.,  0.]
       [0., -1.]]
  operator.shape
  ==> [2, 2]
  operator.log_abs_determinant()
  ==> scalar Tensor
  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor
  diag = tf.random.normal(shape=[2, 3, 4])
  operator = LinearOperatorDiag(diag)
  y = tf.random.normal(shape=[2, 1, 4, 2])
  x = operator.solve(y)
  ==> operator.matmul(x) = y
  ```
  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if
  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```
  Suppose `operator` is a `LinearOperatorDiag` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then
  * `operator.matmul(x)` involves `N * R` multiplications.
  * `operator.solve(x)` involves `N` divisions and `N * R` multiplications.
  * `operator.determinant()` involves a size `N` `reduce_prod`.
  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.
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
               diag,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorDiag"):
    parameters = dict(
        diag=diag,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    with ops.name_scope(name, values=[diag]):
      self._diag = linear_operator_util.convert_nonref_to_tensor(
          diag, name="diag")
      self._check_diag(self._diag)
      if not self._diag.dtype.is_complex:
        if is_self_adjoint is False:
          raise ValueError("A real diagonal operator is always self adjoint.")
        else:
          is_self_adjoint = True
      if is_square is False:
        raise ValueError("Only square diagonal operators currently supported.")
      is_square = True
      super(LinearOperatorDiag, self).__init__(
          dtype=self._diag.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)
  def _check_diag(self, diag):
    if diag.shape.ndims is not None and diag.shape.ndims < 1:
      raise ValueError("Argument diag must have at least 1 dimension.  "
                       "Found: %s" % diag)
  def _shape(self):
    d_shape = self._diag.shape
    return d_shape.concatenate(d_shape[-1:])
  def _shape_tensor(self):
    d_shape = array_ops.shape(self._diag)
    k = d_shape[-1]
    return array_ops.concat((d_shape, [k]), 0)
  @property
  def diag(self):
    return self._diag
  def _assert_non_singular(self):
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self._diag,
        message="Singular operator:  Diagonal contained zero values.")
  def _assert_positive_definite(self):
    if self.dtype.is_complex:
      message = (
          "Diagonal operator had diagonal entries with non-positive real part, "
          "thus was not positive definite.")
    else:
      message = (
          "Real diagonal operator had non-positive diagonal entries, "
          "thus was not positive definite.")
    return check_ops.assert_positive(
        math_ops.real(self._diag),
        message=message)
  def _assert_self_adjoint(self):
    return linear_operator_util.assert_zero_imag_part(
        self._diag,
        message=(
            "This diagonal operator contained non-zero imaginary values.  "
            " Thus it was not self-adjoint."))
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    diag_term = math_ops.conj(self._diag) if adjoint else self._diag
    x = linalg.adjoint(x) if adjoint_arg else x
    diag_mat = array_ops.expand_dims(diag_term, -1)
    return diag_mat * x
  def _matvec(self, x, adjoint=False):
    diag_term = math_ops.conj(self._diag) if adjoint else self._diag
    return diag_term * x
  def _determinant(self):
    return math_ops.reduce_prod(self._diag, axis=[-1])
  def _log_abs_determinant(self):
    log_det = math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self._diag)), axis=[-1])
    if self.dtype.is_complex:
      log_det = math_ops.cast(log_det, dtype=self.dtype)
    return log_det
  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    diag_term = math_ops.conj(self._diag) if adjoint else self._diag
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    inv_diag_mat = array_ops.expand_dims(1. / diag_term, -1)
    return rhs * inv_diag_mat
  def _to_dense(self):
    return array_ops.matrix_diag(self._diag)
  def _diag_part(self):
    return self.diag
  def _add_to_tensor(self, x):
    x_diag = array_ops.matrix_diag_part(x)
    new_diag = self._diag + x_diag
    return array_ops.matrix_set_diag(x, new_diag)
  def _eigvals(self):
    return ops.convert_to_tensor_v2_with_dispatch(self.diag)
  def _cond(self):
    abs_diag = math_ops.abs(self.diag)
    return (math_ops.reduce_max(abs_diag, axis=-1) /
            math_ops.reduce_min(abs_diag, axis=-1))
  @property
  def _composite_tensor_fields(self):
    return ("diag",)
