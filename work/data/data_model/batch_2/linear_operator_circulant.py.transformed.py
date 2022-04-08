
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
__all__ = [
    "LinearOperatorCirculant",
    "LinearOperatorCirculant2D",
    "LinearOperatorCirculant3D",
]
_FFT_OP = {1: fft_ops.fft, 2: fft_ops.fft2d, 3: fft_ops.fft3d}
_IFFT_OP = {1: fft_ops.ifft, 2: fft_ops.ifft2d, 3: fft_ops.ifft3d}
class _BaseLinearOperatorCirculant(linear_operator.LinearOperator):
  def __init__(self,
               spectrum,
               block_depth,
               input_output_dtype=dtypes.complex64,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               parameters=None,
               name="LinearOperatorCirculant"):
    allowed_block_depths = [1, 2, 3]
    self._name = name
    if block_depth not in allowed_block_depths:
      raise ValueError(
          f"Argument `block_depth` must be one of {allowed_block_depths}. "
          f"Received: {block_depth}.")
    self._block_depth = block_depth
    with ops.name_scope(name, values=[spectrum]):
      self._spectrum = self._check_spectrum_and_return_tensor(spectrum)
      if not self.spectrum.dtype.is_complex:
        if is_self_adjoint is False:
          raise ValueError(
              f"A real spectrum always corresponds to a self-adjoint operator. "
              f"Expected argument `is_self_adjoint` to be True when "
              f"`spectrum.dtype.is_complex` = True. "
              f"Received: {is_self_adjoint}.")
        is_self_adjoint = True
      if is_square is False:
        raise ValueError(
            f"A [[nested] block] circulant operator is always square. "
            f"Expected argument `is_square` to be True. Received: {is_square}.")
      is_square = True
      super(_BaseLinearOperatorCirculant, self).__init__(
          dtype=dtypes.as_dtype(input_output_dtype),
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)
  def _check_spectrum_and_return_tensor(self, spectrum):
    spectrum = linear_operator_util.convert_nonref_to_tensor(spectrum,
                                                             name="spectrum")
    if spectrum.shape.ndims is not None:
      if spectrum.shape.ndims < self.block_depth:
        raise ValueError(
            f"Argument `spectrum` must have at least {self.block_depth} "
            f"dimensions. Received: {spectrum}.")
    return spectrum
  @property
  def block_depth(self):
    return self._block_depth
  def block_shape_tensor(self):
    return self._block_shape_tensor()
  def _block_shape_tensor(self, spectrum_shape=None):
    if self.block_shape.is_fully_defined():
      return linear_operator_util.shape_tensor(
          self.block_shape.as_list(), name="block_shape")
    spectrum_shape = (
        array_ops.shape(self.spectrum)
        if spectrum_shape is None else spectrum_shape)
    return spectrum_shape[-self.block_depth:]
  @property
  def block_shape(self):
    return self.spectrum.shape[-self.block_depth:]
  @property
  def spectrum(self):
    return self._spectrum
  def _vectorize_then_blockify(self, matrix):
    vec = distribution_util.rotate_transpose(matrix, shift=1)
    if (vec.shape.is_fully_defined() and
        self.block_shape.is_fully_defined()):
      vec_leading_shape = vec.shape[:-1]
      final_shape = vec_leading_shape.concatenate(self.block_shape)
    else:
      vec_leading_shape = array_ops.shape(vec)[:-1]
      final_shape = array_ops.concat(
          (vec_leading_shape, self.block_shape_tensor()), 0)
    return array_ops.reshape(vec, final_shape)
  def _unblockify_then_matricize(self, vec):
    if vec.shape.is_fully_defined():
      vec_shape = vec.shape.as_list()
      vec_leading_shape = vec_shape[:-self.block_depth]
      vec_block_shape = vec_shape[-self.block_depth:]
      flat_shape = vec_leading_shape + [np.prod(vec_block_shape)]
    else:
      vec_shape = array_ops.shape(vec)
      vec_leading_shape = vec_shape[:-self.block_depth]
      vec_block_shape = vec_shape[-self.block_depth:]
      flat_shape = array_ops.concat(
          (vec_leading_shape, [math_ops.reduce_prod(vec_block_shape)]), 0)
    vec_flat = array_ops.reshape(vec, flat_shape)
    matrix = distribution_util.rotate_transpose(vec_flat, shift=-1)
    return matrix
  def _fft(self, x):
    x_complex = _to_complex(x)
    return _FFT_OP[self.block_depth](x_complex)
  def _ifft(self, x):
    x_complex = _to_complex(x)
    return _IFFT_OP[self.block_depth](x_complex)
  def convolution_kernel(self, name="convolution_kernel"):
      h = self._ifft(_to_complex(self.spectrum))
      return math_ops.cast(h, self.dtype)
  def _shape(self):
    s_shape = self._spectrum.shape
    batch_shape = s_shape[:-self.block_depth]
    trailing_dims = s_shape[-self.block_depth:]
    if trailing_dims.is_fully_defined():
      n = np.prod(trailing_dims.as_list())
    else:
      n = None
    n_x_n = tensor_shape.TensorShape([n, n])
    return batch_shape.concatenate(n_x_n)
  def _shape_tensor(self, spectrum=None):
    spectrum = self.spectrum if spectrum is None else spectrum
    s_shape = array_ops.shape(spectrum)
    batch_shape = s_shape[:-self.block_depth]
    trailing_dims = s_shape[-self.block_depth:]
    n = math_ops.reduce_prod(trailing_dims)
    n_x_n = [n, n]
    return array_ops.concat((batch_shape, n_x_n), 0)
  def assert_hermitian_spectrum(self, name="assert_hermitian_spectrum"):
    eps = np.finfo(self.dtype.real_dtype.as_numpy_dtype).eps
      max_err = eps * self.domain_dimension_tensor()
      imag_convolution_kernel = math_ops.imag(self.convolution_kernel())
      return check_ops.assert_less(
          math_ops.abs(imag_convolution_kernel),
          max_err,
          message="Spectrum was not Hermitian")
  def _assert_non_singular(self):
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self.spectrum,
        message="Singular operator:  Spectrum contained zero values.")
  def _assert_positive_definite(self):
    message = (
        "Not positive definite:  Real part of spectrum was not all positive.")
    return check_ops.assert_positive(
        math_ops.real(self.spectrum), message=message)
  def _assert_self_adjoint(self):
    return linear_operator_util.assert_zero_imag_part(
        self.spectrum,
        message=(
            "Not self-adjoint:  The spectrum contained non-zero imaginary part."
        ))
  def _broadcast_batch_dims(self, x, spectrum):
    spectrum = ops.convert_to_tensor_v2_with_dispatch(spectrum, name="spectrum")
    batch_shape = self._batch_shape_tensor(
        shape=self._shape_tensor(spectrum=spectrum))
    spec_mat = array_ops.reshape(
        spectrum, array_ops.concat((batch_shape, [-1, 1]), axis=0))
    x, spec_mat = linear_operator_util.broadcast_matrix_batch_dims((x,
                                                                    spec_mat))
    x_batch_shape = array_ops.shape(x)[:-2]
    spectrum_shape = array_ops.shape(spectrum)
    spectrum = array_ops.reshape(
        spec_mat,
        array_ops.concat(
            (x_batch_shape,
             self._block_shape_tensor(spectrum_shape=spectrum_shape)),
            axis=0))
    return x, spectrum
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x = linalg.adjoint(x) if adjoint_arg else x
    spectrum = _to_complex(self.spectrum)
    if adjoint:
      spectrum = math_ops.conj(spectrum)
    x = math_ops.cast(x, spectrum.dtype)
    x, spectrum = self._broadcast_batch_dims(x, spectrum)
    x_vb = self._vectorize_then_blockify(x)
    fft_x_vb = self._fft(x_vb)
    block_vector_result = self._ifft(spectrum * fft_x_vb)
    y = self._unblockify_then_matricize(block_vector_result)
    return math_ops.cast(y, self.dtype)
  def _determinant(self):
    axis = [-(i + 1) for i in range(self.block_depth)]
    det = math_ops.reduce_prod(self.spectrum, axis=axis)
    return math_ops.cast(det, self.dtype)
  def _log_abs_determinant(self):
    axis = [-(i + 1) for i in range(self.block_depth)]
    lad = math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self.spectrum)), axis=axis)
    return math_ops.cast(lad, self.dtype)
  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    spectrum = _to_complex(self.spectrum)
    if adjoint:
      spectrum = math_ops.conj(spectrum)
    rhs, spectrum = self._broadcast_batch_dims(rhs, spectrum)
    rhs_vb = self._vectorize_then_blockify(rhs)
    fft_rhs_vb = self._fft(rhs_vb)
    solution_vb = self._ifft(fft_rhs_vb / spectrum)
    x = self._unblockify_then_matricize(solution_vb)
    return math_ops.cast(x, self.dtype)
  def _diag_part(self):
    if self.shape.is_fully_defined():
      diag_shape = self.shape[:-1]
      diag_size = self.domain_dimension.value
    else:
      diag_shape = self.shape_tensor()[:-1]
      diag_size = self.domain_dimension_tensor()
    ones_diag = array_ops.ones(diag_shape, dtype=self.dtype)
    diag_value = self.trace() / math_ops.cast(diag_size, self.dtype)
    return diag_value[..., array_ops.newaxis] * ones_diag
  def _trace(self):
    if self.spectrum.shape.is_fully_defined():
      spec_rank = self.spectrum.shape.ndims
      axis = np.arange(spec_rank - self.block_depth, spec_rank, dtype=np.int32)
    else:
      spec_rank = array_ops.rank(self.spectrum)
      axis = math_ops.range(spec_rank - self.block_depth, spec_rank)
    re_d_value = math_ops.reduce_sum(math_ops.real(self.spectrum), axis=axis)
    if not self.dtype.is_complex:
      return math_ops.cast(re_d_value, self.dtype)
    if self.is_self_adjoint:
      im_d_value = array_ops.zeros_like(re_d_value)
    else:
      im_d_value = math_ops.reduce_sum(math_ops.imag(self.spectrum), axis=axis)
    return math_ops.cast(math_ops.complex(re_d_value, im_d_value), self.dtype)
  @property
  def _composite_tensor_fields(self):
    return ("spectrum", "input_output_dtype")
@tf_export("linalg.LinearOperatorCirculant")
@linear_operator.make_composite_tensor
class LinearOperatorCirculant(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a circulant matrix.
  This operator acts like a circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.
  Circulant means the entries of `A` are generated by a single vector, the
  convolution kernel `h`: `A_{mn} := h_{m-n mod N}`.  With `h = [w, x, y, z]`,
  ```
  A = |w z y x|
      |x w z y|
      |y x w z|
      |z y x w|
  ```
  This means that the result of matrix multiplication `v = Au` has `Lth` column
  given circular convolution between `h` with the `Lth` column of `u`.
  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.  Define the discrete Fourier transform (DFT) and its inverse by
  ```
  DFT[ h[n] ] = H[k] := sum_{n = 0}^{N - 1} h_n e^{-i 2pi k n / N}
  IDFT[ H[k] ] = h[n] = N^{-1} sum_{k = 0}^{N - 1} H_k e^{i 2pi k n / N}
  ```
  From these definitions, we see that
  ```
  H[0] = sum_{n = 0}^{N - 1} h_n
  H[1] = "the first positive frequency"
  H[N - 1] = "the first negative frequency"
  ```
  Loosely speaking, with `*` element-wise multiplication, matrix multiplication
  is equal to the action of a Fourier multiplier: `A u = IDFT[ H * DFT[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT[u]` be the `[N, R]`
  matrix with `rth` column equal to the DFT of the `rth` column of `u`.
  Define the `IDFT` similarly.
  Matrix multiplication may be expressed columnwise:
  ```(A u)_r = IDFT[ H * (DFT[u])_r ]```
  Letting `U` be the `kth` Euclidean basis vector, and `U = IDFT[u]`.
  The above formulas show that`A U = H_k * U`.  We conclude that the elements
  of `H` are the eigenvalues of this operator.   Therefore
  * This operator is positive definite if and only if `Real{H} > 0`.
  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.
  Suppose `H.shape = [B1,...,Bb, N]`.  We say that `H` is a Hermitian spectrum
  if, with `%` meaning modulus division,
  ```H[..., n % N] = ComplexConjugate[ H[..., (-n) % N] ]```
  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.
  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.
  ```python
  spectrum = [6., 4, 2]
  operator = LinearOperatorCirculant(spectrum)
  operator.convolution_kernel()
  ==> [4 + 0j, 1 + 0.58j, 1 - 0.58j]
  operator.to_dense()
  ==> [[4 + 0.0j, 1 - 0.6j, 1 + 0.6j],
       [1 + 0.6j, 4 + 0.0j, 1 - 0.6j],
       [1 - 0.6j, 1 + 0.6j, 4 + 0.0j]]
  ```
  ```python
  convolution_kernel = [1., 2., 1.]]
  spectrum = tf.signal.fft(tf.cast(convolution_kernel, tf.complex64))
  operator = LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)
  operator.to_dense()
  ==> [[ 1, 1, 2],
       [ 2, 1, 1],
       [ 1, 2, 1]]
  ```
  ```python
  spectrum = [1, 1j, -1j]
  operator = LinearOperatorCirculant(spectrum)
  operator.to_dense()
  ==> [[ 0.33 + 0j,  0.91 + 0j, -0.24 + 0j],
       [-0.24 + 0j,  0.33 + 0j,  0.91 + 0j],
       [ 0.91 + 0j, -0.24 + 0j,  0.33 + 0j]
  ```
  ```python
  spectrum = [6., 4, 2, 4]
  operator = LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)
  operator.shape
  ==> [4, 4]
  operator.to_dense()
  ==> [[4, 1, 0, 1],
       [1, 4, 1, 0],
       [0, 1, 4, 1],
       [1, 0, 1, 4]]
  operator.convolution_kernel()
  ==> [4, 1, 0, 1]
  ```
  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then
  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
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
  References:
    Toeplitz and Circulant Matrices - A Review:
      [Gray, 2006](https://www.nowpublishers.com/article/Details/CIT-006)
      ([pdf](https://ee.stanford.edu/~gray/toeplitz.pdf))
  """
  def __init__(self,
               spectrum,
               input_output_dtype=dtypes.complex64,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name="LinearOperatorCirculant"):
    parameters = dict(
        spectrum=spectrum,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    super(LinearOperatorCirculant, self).__init__(
        spectrum,
        block_depth=1,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters,
        name=name)
  def _eigvals(self):
    return ops.convert_to_tensor_v2_with_dispatch(self.spectrum)
@tf_export("linalg.LinearOperatorCirculant2D")
@linear_operator.make_composite_tensor
class LinearOperatorCirculant2D(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a block circulant matrix.
  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.
  If `A` is block circulant, with block sizes `N0, N1` (`N0 * N1 = N`):
  `A` has a block circulant structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` circulant matrix.
  For example, with `W`, `X`, `Y`, `Z` each circulant,
  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```
  Note that `A` itself will not in general be circulant.
  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.
  If `H.shape = [N0, N1]`, (`N0 * N1 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT2[ H DFT2[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT2[u]` be the
  `[N0, N1, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, R]` and taking
  a two dimensional DFT across the first two dimensions.  Let `IDFT2` be the
  inverse of `DFT2`.  Matrix multiplication may be expressed columnwise:
  ```(A u)_r = IDFT2[ H * (DFT2[u])_r ]```
  * This operator is positive definite if and only if `Real{H} > 0`.
  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.
  Suppose `H.shape = [B1,...,Bb, N0, N1]`, we say that `H` is a Hermitian
  spectrum if, with `%` indicating modulus division,
  ```
  H[..., n0 % N0, n1 % N1] = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1 ].
  ```
  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.
  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.
  ```python
  spectrum = [[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]]
  operator = LinearOperatorCirculant2D(spectrum)
  operator.convolution_kernel()
  ==> [[5.0+0.0j, -0.5-.3j, -0.5+.3j],
       [-1.5-.9j,        0,        0],
       [-1.5+.9j,        0,        0]]
  operator.to_dense()
  ==> Complex self adjoint 9 x 9 matrix.
  ```
  ```python
  convolution_kernel = [[1., 2., 1.], [5., -1., 1.]]
  spectrum = tf.signal.fft2d(tf.cast(convolution_kernel, tf.complex64))
  operator = LinearOperatorCirculant2D(spectrum, input_output_dtype=tf.float32)
  ```
  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then
  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.
  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.
  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """
  def __init__(self,
               spectrum,
               input_output_dtype=dtypes.complex64,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name="LinearOperatorCirculant2D"):
    parameters = dict(
        spectrum=spectrum,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    super(LinearOperatorCirculant2D, self).__init__(
        spectrum,
        block_depth=2,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters,
        name=name)
@tf_export("linalg.LinearOperatorCirculant3D")
@linear_operator.make_composite_tensor
class LinearOperatorCirculant3D(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a nested block circulant matrix.
  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.
  If `A` is nested block circulant, with block sizes `N0, N1, N2`
  (`N0 * N1 * N2 = N`):
  `A` has a block structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` block circulant matrix.
  For example, with `W`, `X`, `Y`, `Z` each block circulant,
  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```
  Note that `A` itself will not in general be circulant.
  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.
  If `H.shape = [N0, N1, N2]`, (`N0 * N1 * N2 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT3[ H DFT3[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT3[u]` be the
  `[N0, N1, N2, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, N2, R]` and
  taking a three dimensional DFT across the first three dimensions.  Let `IDFT3`
  be the inverse of `DFT3`.  Matrix multiplication may be expressed columnwise:
  ```(A u)_r = IDFT3[ H * (DFT3[u])_r ]```
  * This operator is positive definite if and only if `Real{H} > 0`.
  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.
  Suppose `H.shape = [B1,...,Bb, N0, N1, N2]`, we say that `H` is a Hermitian
  spectrum if, with `%` meaning modulus division,
  ```
  H[..., n0 % N0, n1 % N1, n2 % N2]
    = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1, (-n2) % N2] ].
  ```
  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.
  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.
  See `LinearOperatorCirculant` and `LinearOperatorCirculant2D` for examples.
  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then
  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.
  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.
  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """
  def __init__(self,
               spectrum,
               input_output_dtype=dtypes.complex64,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name="LinearOperatorCirculant3D"):
    parameters = dict(
        spectrum=spectrum,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    super(LinearOperatorCirculant3D, self).__init__(
        spectrum,
        block_depth=3,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters,
        name=name)
def _to_complex(x):
  if x.dtype.is_complex:
    return x
  dtype = dtypes.complex64
  if x.dtype == dtypes.float64:
    dtype = dtypes.complex128
  return math_ops.cast(x, dtype)
