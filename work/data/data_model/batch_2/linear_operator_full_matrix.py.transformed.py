
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
__all__ = ["LinearOperatorFullMatrix"]
@tf_export("linalg.LinearOperatorFullMatrix")
@linear_operator.make_composite_tensor
class LinearOperatorFullMatrix(linear_operator.LinearOperator):
  """`LinearOperator` that wraps a [batch] matrix.
  This operator wraps a [batch] matrix `A` (which is a `Tensor`) with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `M x N` matrix.
  ```python
  matrix = [[1., 2.], [3., 4.]]
  operator = LinearOperatorFullMatrix(matrix)
  operator.to_dense()
  ==> [[1., 2.]
       [3., 4.]]
  operator.shape
  ==> [2, 2]
  operator.log_abs_determinant()
  ==> scalar Tensor
  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor
  matrix = tf.random.normal(shape=[2, 3, 4, 4])
  operator = LinearOperatorFullMatrix(matrix)
  ```
  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if
  ```
  operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```
  `LinearOperatorFullMatrix` has exactly the same performance as would be
  achieved by using standard `TensorFlow` matrix ops.  Intelligent choices are
  made based on the following initialization hints.
  * If `dtype` is real, and `is_self_adjoint` and `is_positive_definite`, a
    Cholesky factorization is used for the determinant and solve.
  In all cases, suppose `operator` is a `LinearOperatorFullMatrix` of shape
  `[M, N]`, and `x.shape = [N, R]`.  Then
  * `operator.matmul(x)` is `O(M * N * R)`.
  * If `M=N`, `operator.solve(x)` is `O(N^3 * R)`.
  * If `M=N`, `operator.determinant()` is `O(N^3)`.
  If instead `operator` and `x` have shape `[B1,...,Bb, M, N]` and
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
               matrix,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorFullMatrix"):
    parameters = dict(
        matrix=matrix,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    with ops.name_scope(name, values=[matrix]):
      self._matrix = linear_operator_util.convert_nonref_to_tensor(
          matrix, name="matrix")
      self._check_matrix(self._matrix)
      super(LinearOperatorFullMatrix, self).__init__(
          dtype=self._matrix.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)
  def _check_matrix(self, matrix):
    allowed_dtypes = [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.complex64,
        dtypes.complex128,
    ]
    matrix = ops.convert_to_tensor_v2_with_dispatch(matrix, name="matrix")
    dtype = matrix.dtype
    if dtype not in allowed_dtypes:
      raise TypeError(f"Argument `matrix` must have dtype in {allowed_dtypes}. "
                      f"Received: {dtype}.")
    if matrix.shape.ndims is not None and matrix.shape.ndims < 2:
      raise ValueError(f"Argument `matrix` must have at least 2 dimensions. "
                       f"Received: {matrix}.")
  def _shape(self):
    return self._matrix.shape
  def _shape_tensor(self):
    return array_ops.shape(self._matrix)
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    return math_ops.matmul(
        self._matrix, x, adjoint_a=adjoint, adjoint_b=adjoint_arg)
  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    return self._dense_solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
  def _to_dense(self):
    return self._matrix
  @property
  def _composite_tensor_fields(self):
    return ("matrix",)