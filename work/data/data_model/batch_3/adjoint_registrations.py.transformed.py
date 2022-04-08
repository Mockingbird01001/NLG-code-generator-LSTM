
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_adjoint
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_householder
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_kronecker
@linear_operator_algebra.RegisterAdjoint(linear_operator.LinearOperator)
def _adjoint_linear_operator(linop):
  return linear_operator_adjoint.LinearOperatorAdjoint(
      linop,
      is_non_singular=linop.is_non_singular,
      is_self_adjoint=linop.is_self_adjoint,
      is_positive_definite=linop.is_positive_definite,
      is_square=linop.is_square)
@linear_operator_algebra.RegisterAdjoint(
    linear_operator_adjoint.LinearOperatorAdjoint)
def _adjoint_adjoint_linear_operator(linop):
  return linop.operator
@linear_operator_algebra.RegisterAdjoint(
    linear_operator_identity.LinearOperatorIdentity)
def _adjoint_identity(identity_operator):
  return identity_operator
@linear_operator_algebra.RegisterAdjoint(
    linear_operator_identity.LinearOperatorScaledIdentity)
def _adjoint_scaled_identity(identity_operator):
  multiplier = identity_operator.multiplier
  if multiplier.dtype.is_complex:
    multiplier = math_ops.conj(multiplier)
  return linear_operator_identity.LinearOperatorScaledIdentity(
      multiplier=multiplier,
      is_non_singular=identity_operator.is_non_singular,
      is_self_adjoint=identity_operator.is_self_adjoint,
      is_positive_definite=identity_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterAdjoint(
    linear_operator_diag.LinearOperatorDiag)
def _adjoint_diag(diag_operator):
  diag = diag_operator.diag
  if diag.dtype.is_complex:
    diag = math_ops.conj(diag)
  return linear_operator_diag.LinearOperatorDiag(
      diag=diag,
      is_non_singular=diag_operator.is_non_singular,
      is_self_adjoint=diag_operator.is_self_adjoint,
      is_positive_definite=diag_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterAdjoint(
    linear_operator_block_diag.LinearOperatorBlockDiag)
def _adjoint_block_diag(block_diag_operator):
  return linear_operator_block_diag.LinearOperatorBlockDiag(
      operators=[
          operator.adjoint() for operator in block_diag_operator.operators],
      is_non_singular=block_diag_operator.is_non_singular,
      is_self_adjoint=block_diag_operator.is_self_adjoint,
      is_positive_definite=block_diag_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterAdjoint(
    linear_operator_kronecker.LinearOperatorKronecker)
def _adjoint_kronecker(kronecker_operator):
  return linear_operator_kronecker.LinearOperatorKronecker(
      operators=[
          operator.adjoint() for operator in kronecker_operator.operators],
      is_non_singular=kronecker_operator.is_non_singular,
      is_self_adjoint=kronecker_operator.is_self_adjoint,
      is_positive_definite=kronecker_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterAdjoint(
    linear_operator_circulant.LinearOperatorCirculant)
def _adjoint_circulant(circulant_operator):
  spectrum = circulant_operator.spectrum
  if spectrum.dtype.is_complex:
    spectrum = math_ops.conj(spectrum)
  return linear_operator_circulant.LinearOperatorCirculant(
      spectrum=spectrum,
      is_non_singular=circulant_operator.is_non_singular,
      is_self_adjoint=circulant_operator.is_self_adjoint,
      is_positive_definite=circulant_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterAdjoint(
    linear_operator_householder.LinearOperatorHouseholder)
def _adjoint_householder(householder_operator):
  return householder_operator
