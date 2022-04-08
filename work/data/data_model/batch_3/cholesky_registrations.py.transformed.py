
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_kronecker
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
@linear_operator_algebra.RegisterCholesky(linear_operator.LinearOperator)
def _cholesky_linear_operator(linop):
  return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
      linalg_ops.cholesky(linop.to_dense()),
      is_non_singular=True,
      is_self_adjoint=False,
      is_square=True)
@linear_operator_algebra.RegisterCholesky(
    linear_operator_diag.LinearOperatorDiag)
def _cholesky_diag(diag_operator):
  return linear_operator_diag.LinearOperatorDiag(
      math_ops.sqrt(diag_operator.diag),
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)
@linear_operator_algebra.RegisterCholesky(
    linear_operator_identity.LinearOperatorIdentity)
def _cholesky_identity(identity_operator):
  return linear_operator_identity.LinearOperatorIdentity(
      batch_shape=identity_operator.batch_shape,
      dtype=identity_operator.dtype,
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)
@linear_operator_algebra.RegisterCholesky(
    linear_operator_identity.LinearOperatorScaledIdentity)
def _cholesky_scaled_identity(identity_operator):
  return linear_operator_identity.LinearOperatorScaledIdentity(
      multiplier=math_ops.sqrt(identity_operator.multiplier),
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)
@linear_operator_algebra.RegisterCholesky(
    linear_operator_block_diag.LinearOperatorBlockDiag)
def _cholesky_block_diag(block_diag_operator):
  return linear_operator_block_diag.LinearOperatorBlockDiag(
      operators=[
          operator.cholesky() for operator in block_diag_operator.operators],
      is_non_singular=True,
      is_self_adjoint=False,
      is_square=True)
@linear_operator_algebra.RegisterCholesky(
    linear_operator_kronecker.LinearOperatorKronecker)
def _cholesky_kronecker(kronecker_operator):
  return linear_operator_kronecker.LinearOperatorKronecker(
      operators=[
          operator.cholesky() for operator in kronecker_operator.operators],
      is_non_singular=True,
      is_self_adjoint=False,
      is_square=True)
