
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test
@test_util.run_all_without_tensor_float_32
class SquareRootOpTest(test.TestCase):
  def _verifySquareRoot(self, matrix, np_type):
    matrix = matrix.astype(np_type)
    sqrt = gen_linalg_ops.matrix_square_root(matrix)
    square = test_util.matmul_without_tf32(sqrt, sqrt)
    self.assertShapeEqual(matrix, square)
    self.assertAllClose(matrix, square, rtol=1e-4, atol=1e-3)
  def _verifySquareRootReal(self, x):
    for np_type in [np.float32, np.float64]:
      self._verifySquareRoot(x, np_type)
  def _verifySquareRootComplex(self, x):
    for np_type in [np.complex64, np.complex128]:
      self._verifySquareRoot(x, np_type)
  def _makeBatch(self, matrix1, matrix2):
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch
  def _testMatrices(self, matrix1, matrix2):
    self._verifySquareRootReal(matrix1)
    self._verifySquareRootReal(matrix2)
    self._verifySquareRootReal(self._makeBatch(matrix1, matrix2))
    matrix1 = matrix1.astype(np.complex64)
    matrix2 = matrix2.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 += 1j * matrix2
    self._verifySquareRootComplex(matrix1)
    self._verifySquareRootComplex(matrix2)
    self._verifySquareRootComplex(self._makeBatch(matrix1, matrix2))
  def testSymmetricPositiveDefinite(self):
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._testMatrices(matrix1, matrix2)
  def testAsymmetric(self):
    matrix1 = np.array([[0., 4.], [-1., 5.]])
    matrix2 = np.array([[33., 24.], [48., 57.]])
    self._testMatrices(matrix1, matrix2)
  def testIdentityMatrix(self):
    identity = np.array([[1., 0], [0, 1.]])
    self._verifySquareRootReal(identity)
    identity = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    self._verifySquareRootReal(identity)
  def testEmpty(self):
    self._verifySquareRootReal(np.empty([0, 2, 2]))
    self._verifySquareRootReal(np.empty([2, 0, 0]))
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    tensor = constant_op.constant([1., 2.])
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      gen_linalg_ops.matrix_square_root(tensor)
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNotSquare(self):
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      tensor = constant_op.constant([[1., 0., -1.], [-1., 1., 0.]])
      self.evaluate(gen_linalg_ops.matrix_square_root(tensor))
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testConcurrentExecutesWithoutError(self):
    matrix_shape = [5, 5]
    seed = [42, 24]
    matrix1 = stateless_random_ops.stateless_random_normal(
        shape=matrix_shape, seed=seed)
    matrix2 = stateless_random_ops.stateless_random_normal(
        shape=matrix_shape, seed=seed)
    self.assertAllEqual(matrix1, matrix2)
    square1 = math_ops.matmul(matrix1, matrix1)
    square2 = math_ops.matmul(matrix2, matrix2)
    sqrt1 = gen_linalg_ops.matrix_square_root(square1)
    sqrt2 = gen_linalg_ops.matrix_square_root(square2)
    all_ops = [sqrt1, sqrt2]
    sqrt = self.evaluate(all_ops)
    self.assertAllClose(sqrt[0], sqrt[1])
if __name__ == "__main__":
  test.main()
