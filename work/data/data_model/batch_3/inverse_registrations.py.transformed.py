
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_addition
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_block_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_householder
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_kronecker
@linear_operator_algebra.RegisterInverse(linear_operator.LinearOperator)
def _inverse_linear_operator(linop):
  return linear_operator_inversion.LinearOperatorInversion(
      linop,
      is_non_singular=linop.is_non_singular,
      is_self_adjoint=linop.is_self_adjoint,
      is_positive_definite=linop.is_positive_definite,
      is_square=linop.is_square)
@linear_operator_algebra.RegisterInverse(
    linear_operator_inversion.LinearOperatorInversion)
def _inverse_inverse_linear_operator(linop_inversion):
  return linop_inversion.operator
@linear_operator_algebra.RegisterInverse(
    linear_operator_diag.LinearOperatorDiag)
def _inverse_diag(diag_operator):
  return linear_operator_diag.LinearOperatorDiag(
      1. / diag_operator.diag,
      is_non_singular=diag_operator.is_non_singular,
      is_self_adjoint=diag_operator.is_self_adjoint,
      is_positive_definite=diag_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterInverse(
    linear_operator_identity.LinearOperatorIdentity)
def _inverse_identity(identity_operator):
  return identity_operator
@linear_operator_algebra.RegisterInverse(
    linear_operator_identity.LinearOperatorScaledIdentity)
def _inverse_scaled_identity(identity_operator):
  return linear_operator_identity.LinearOperatorScaledIdentity(
      multiplier=1. / identity_operator.multiplier,
      is_non_singular=identity_operator.is_non_singular,
      is_self_adjoint=True,
      is_positive_definite=identity_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterInverse(
    linear_operator_block_diag.LinearOperatorBlockDiag)
def _inverse_block_diag(block_diag_operator):
  return linear_operator_block_diag.LinearOperatorBlockDiag(
      operators=[
          operator.inverse() for operator in block_diag_operator.operators],
      is_non_singular=block_diag_operator.is_non_singular,
      is_self_adjoint=block_diag_operator.is_self_adjoint,
      is_positive_definite=block_diag_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterInverse(
    linear_operator_block_lower_triangular.LinearOperatorBlockLowerTriangular)
def _inverse_block_lower_triangular(block_lower_triangular_operator):
  if len(block_lower_triangular_operator.operators) == 1:
    return (linear_operator_block_lower_triangular.
            LinearOperatorBlockLowerTriangular(
                [[block_lower_triangular_operator.operators[0][0].inverse()]],
                is_non_singular=block_lower_triangular_operator.is_non_singular,
                is_self_adjoint=block_lower_triangular_operator.is_self_adjoint,
                is_positive_definite=(block_lower_triangular_operator.
                                      is_positive_definite),
                is_square=True))
  blockwise_dim = len(block_lower_triangular_operator.operators)
  upper_left_inverse = (
      linear_operator_block_lower_triangular.LinearOperatorBlockLowerTriangular(
          block_lower_triangular_operator.operators[:-1]).inverse())
  bottom_row = block_lower_triangular_operator.operators[-1]
  bottom_right_inverse = bottom_row[-1].inverse()
  inverse_bottom_row = []
  for i in range(blockwise_dim - 1):
    blocks = []
    for j in range(i, blockwise_dim - 1):
      result = bottom_row[j].matmul(upper_left_inverse.operators[j][i])
      if not any(isinstance(result, op_type)
                 for op_type in linear_operator_addition.SUPPORTED_OPERATORS):
        result = linear_operator_full_matrix.LinearOperatorFullMatrix(
            result.to_dense())
      blocks.append(result)
    summed_blocks = linear_operator_addition.add_operators(blocks)
    assert len(summed_blocks) == 1
    block = summed_blocks[0]
    block = bottom_right_inverse.matmul(block)
    block = linear_operator_identity.LinearOperatorScaledIdentity(
        num_rows=bottom_right_inverse.domain_dimension_tensor(),
        multiplier=math_ops.cast(-1, dtype=block.dtype)).matmul(block)
    inverse_bottom_row.append(block)
  inverse_bottom_row.append(bottom_right_inverse)
  return (
      linear_operator_block_lower_triangular.LinearOperatorBlockLowerTriangular(
          upper_left_inverse.operators + [inverse_bottom_row],
          is_non_singular=block_lower_triangular_operator.is_non_singular,
          is_self_adjoint=block_lower_triangular_operator.is_self_adjoint,
          is_positive_definite=(block_lower_triangular_operator.
                                is_positive_definite),
          is_square=True))
@linear_operator_algebra.RegisterInverse(
    linear_operator_kronecker.LinearOperatorKronecker)
def _inverse_kronecker(kronecker_operator):
  return linear_operator_kronecker.LinearOperatorKronecker(
      operators=[
          operator.inverse() for operator in kronecker_operator.operators],
      is_non_singular=kronecker_operator.is_non_singular,
      is_self_adjoint=kronecker_operator.is_self_adjoint,
      is_positive_definite=kronecker_operator.is_positive_definite,
      is_square=True)
@linear_operator_algebra.RegisterInverse(
    linear_operator_circulant.LinearOperatorCirculant)
def _inverse_circulant(circulant_operator):
  return linear_operator_circulant.LinearOperatorCirculant(
      spectrum=1. / circulant_operator.spectrum,
      is_non_singular=circulant_operator.is_non_singular,
      is_self_adjoint=circulant_operator.is_self_adjoint,
      is_positive_definite=circulant_operator.is_positive_definite,
      is_square=True,
      input_output_dtype=circulant_operator.dtype)
@linear_operator_algebra.RegisterInverse(
    linear_operator_householder.LinearOperatorHouseholder)
def _inverse_householder(householder_operator):
  return householder_operator
