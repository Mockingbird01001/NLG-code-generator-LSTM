
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_zeros
from tensorflow.python.ops.linalg import registrations_util
@linear_operator_algebra.RegisterMatmul(
    linear_operator.LinearOperator, linear_operator.LinearOperator)
def _matmul_linear_operator(linop_a, linop_b):
  is_square = registrations_util.is_square(linop_a, linop_b)
  is_non_singular = None
  is_self_adjoint = None
  is_positive_definite = None
  if is_square:
    is_non_singular = registrations_util.combined_non_singular_hint(
        linop_a, linop_b)
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
  return linear_operator_composition.LinearOperatorComposition(
      operators=[linop_a, linop_b],
      is_non_singular=is_non_singular,
      is_self_adjoint=is_self_adjoint,
      is_positive_definite=is_positive_definite,
      is_square=is_square,
  )
@linear_operator_algebra.RegisterMatmul(
    linear_operator_identity.LinearOperatorIdentity,
    linear_operator.LinearOperator)
def _matmul_linear_operator_identity_left(identity, linop):
  del identity
  return linop
@linear_operator_algebra.RegisterMatmul(
    linear_operator.LinearOperator,
    linear_operator_identity.LinearOperatorIdentity)
def _matmul_linear_operator_identity_right(linop, identity):
  del identity
  return linop
@linear_operator_algebra.RegisterMatmul(
    linear_operator_identity.LinearOperatorScaledIdentity,
    linear_operator_identity.LinearOperatorScaledIdentity)
def _matmul_linear_operator_scaled_identity(linop_a, linop_b):
  return linear_operator_identity.LinearOperatorScaledIdentity(
      num_rows=linop_a.domain_dimension_tensor(),
      multiplier=linop_a.multiplier * linop_b.multiplier,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_a, linop_b),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_a, linop_b)),
      is_square=True)
@linear_operator_algebra.RegisterMatmul(
    linear_operator.LinearOperator,
    linear_operator_zeros.LinearOperatorZeros)
def _matmul_linear_operator_zeros_right(linop, zeros):
  if not zeros.is_square or not linop.is_square:
    raise ValueError("Matmul with non-square `LinearOperator`s or non-square "
                     "`LinearOperatorZeros` not supported at this time.")
  return zeros
@linear_operator_algebra.RegisterMatmul(
    linear_operator_zeros.LinearOperatorZeros,
    linear_operator.LinearOperator)
def _matmul_linear_operator_zeros_left(zeros, linop):
  if not zeros.is_square or not linop.is_square:
    raise ValueError("Matmul with non-square `LinearOperator`s or non-square "
                     "`LinearOperatorZeros` not supported at this time.")
  return zeros
@linear_operator_algebra.RegisterMatmul(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_diag.LinearOperatorDiag)
def _matmul_linear_operator_diag(linop_a, linop_b):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_a.diag * linop_b.diag,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_a, linop_b),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_a, linop_b)),
      is_square=True)
@linear_operator_algebra.RegisterMatmul(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_identity.LinearOperatorScaledIdentity)
def _matmul_linear_operator_diag_scaled_identity_right(
    linop_diag, linop_scaled_identity):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_diag.diag * linop_scaled_identity.multiplier,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_diag, linop_scaled_identity),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_diag, linop_scaled_identity),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_diag, linop_scaled_identity)),
      is_square=True)
@linear_operator_algebra.RegisterMatmul(
    linear_operator_identity.LinearOperatorScaledIdentity,
    linear_operator_diag.LinearOperatorDiag)
def _matmul_linear_operator_diag_scaled_identity_left(
    linop_scaled_identity, linop_diag):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_diag.diag * linop_scaled_identity.multiplier,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_diag, linop_scaled_identity),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_diag, linop_scaled_identity),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_diag, linop_scaled_identity)),
      is_square=True)
@linear_operator_algebra.RegisterMatmul(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_lower_triangular.LinearOperatorLowerTriangular)
def _matmul_linear_operator_diag_tril(linop_diag, linop_triangular):
  return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
      tril=linop_diag.diag[..., None] * linop_triangular.to_dense(),
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_diag, linop_triangular),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_diag, linop_triangular),
      is_positive_definite=None,
      is_square=True)
@linear_operator_algebra.RegisterMatmul(
    linear_operator_lower_triangular.LinearOperatorLowerTriangular,
    linear_operator_diag.LinearOperatorDiag)
def _matmul_linear_operator_tril_diag(linop_triangular, linop_diag):
  return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
      tril=linop_triangular.to_dense() * linop_diag.diag,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_diag, linop_triangular),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_diag, linop_triangular),
      is_positive_definite=None,
      is_square=True)
@linear_operator_algebra.RegisterMatmul(
    linear_operator_circulant.LinearOperatorCirculant,
    linear_operator_circulant.LinearOperatorCirculant)
def _matmul_linear_operator_circulant_circulant(linop_a, linop_b):
  return linear_operator_circulant.LinearOperatorCirculant(
      spectrum=linop_a.spectrum * linop_b.spectrum,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_a, linop_b),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_a, linop_b)),
      is_square=True)
@linear_operator_algebra.RegisterMatmul(
    linear_operator_block_diag.LinearOperatorBlockDiag,
    linear_operator_block_diag.LinearOperatorBlockDiag)
def _matmul_linear_operator_block_diag_block_diag(linop_a, linop_b):
  return linear_operator_block_diag.LinearOperatorBlockDiag(
      operators=[
          o1.matmul(o2) for o1, o2 in zip(
              linop_a.operators, linop_b.operators)],
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=None,
      is_positive_definite=None,
      is_square=True)
