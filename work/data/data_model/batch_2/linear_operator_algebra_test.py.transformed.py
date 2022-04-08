
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.platform import test
_ADJOINTS = linear_operator_algebra._ADJOINTS
_registered_adjoint = linear_operator_algebra._registered_adjoint
_CHOLESKY_DECOMPS = linear_operator_algebra._CHOLESKY_DECOMPS
_registered_cholesky = linear_operator_algebra._registered_cholesky
_INVERSES = linear_operator_algebra._INVERSES
_registered_inverse = linear_operator_algebra._registered_inverse
_MATMUL = linear_operator_algebra._MATMUL
_registered_matmul = linear_operator_algebra._registered_matmul
_SOLVE = linear_operator_algebra._SOLVE
_registered_solve = linear_operator_algebra._registered_solve
class AdjointTest(test.TestCase):
  def testRegistration(self):
    class CustomLinOp(linear_operator.LinearOperator):
      def _matmul(self, a):
        pass
      def _shape(self):
        return tensor_shape.TensorShape([1, 1])
      def _shape_tensor(self):
        pass
    @linear_operator_algebra.RegisterAdjoint(CustomLinOp)
      return "OK"
    self.assertEqual("OK", CustomLinOp(dtype=None).adjoint())
  def testRegistrationFailures(self):
    class CustomLinOp(linear_operator.LinearOperator):
      pass
    with self.assertRaisesRegex(TypeError, "must be callable"):
      linear_operator_algebra.RegisterAdjoint(CustomLinOp)("blah")
    linear_operator_algebra.RegisterAdjoint(CustomLinOp)(lambda a: None)
    with self.assertRaisesRegex(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterAdjoint(CustomLinOp)(lambda a: None)
  def testExactAdjointRegistrationsAllMatch(self):
    for (k, v) in _ADJOINTS.items():
      self.assertEqual(v, _registered_adjoint(k[0]))
class CholeskyTest(test.TestCase):
  def testRegistration(self):
    class CustomLinOp(linear_operator.LinearOperator):
      def _matmul(self, a):
        pass
      def _shape(self):
        return tensor_shape.TensorShape([1, 1])
      def _shape_tensor(self):
        pass
    @linear_operator_algebra.RegisterCholesky(CustomLinOp)
      return "OK"
    with self.assertRaisesRegex(ValueError, "positive definite"):
      CustomLinOp(dtype=None, is_self_adjoint=True).cholesky()
    with self.assertRaisesRegex(ValueError, "self adjoint"):
      CustomLinOp(dtype=None, is_positive_definite=True).cholesky()
    custom_linop = CustomLinOp(
        dtype=None, is_self_adjoint=True, is_positive_definite=True)
    self.assertEqual("OK", custom_linop.cholesky())
  def testRegistrationFailures(self):
    class CustomLinOp(linear_operator.LinearOperator):
      pass
    with self.assertRaisesRegex(TypeError, "must be callable"):
      linear_operator_algebra.RegisterCholesky(CustomLinOp)("blah")
    linear_operator_algebra.RegisterCholesky(CustomLinOp)(lambda a: None)
    with self.assertRaisesRegex(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterCholesky(CustomLinOp)(lambda a: None)
  def testExactCholeskyRegistrationsAllMatch(self):
    for (k, v) in _CHOLESKY_DECOMPS.items():
      self.assertEqual(v, _registered_cholesky(k[0]))
class MatmulTest(test.TestCase):
  def testRegistration(self):
    class CustomLinOp(linear_operator.LinearOperator):
      def _matmul(self, a):
        pass
      def _shape(self):
        return tensor_shape.TensorShape([1, 1])
      def _shape_tensor(self):
        pass
    @linear_operator_algebra.RegisterMatmul(CustomLinOp, CustomLinOp)
      return "OK"
    custom_linop = CustomLinOp(
        dtype=None, is_self_adjoint=True, is_positive_definite=True)
    self.assertEqual("OK", custom_linop.matmul(custom_linop))
  def testRegistrationFailures(self):
    class CustomLinOp(linear_operator.LinearOperator):
      pass
    with self.assertRaisesRegex(TypeError, "must be callable"):
      linear_operator_algebra.RegisterMatmul(CustomLinOp, CustomLinOp)("blah")
    linear_operator_algebra.RegisterMatmul(
        CustomLinOp, CustomLinOp)(lambda a: None)
    with self.assertRaisesRegex(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterMatmul(
          CustomLinOp, CustomLinOp)(lambda a: None)
  def testExactMatmulRegistrationsAllMatch(self):
    for (k, v) in _MATMUL.items():
      self.assertEqual(v, _registered_matmul(k[0], k[1]))
class SolveTest(test.TestCase):
  def testRegistration(self):
    class CustomLinOp(linear_operator.LinearOperator):
      def _matmul(self, a):
        pass
      def _solve(self, a):
        pass
      def _shape(self):
        return tensor_shape.TensorShape([1, 1])
      def _shape_tensor(self):
        pass
    @linear_operator_algebra.RegisterSolve(CustomLinOp, CustomLinOp)
      return "OK"
    custom_linop = CustomLinOp(
        dtype=None, is_self_adjoint=True, is_positive_definite=True)
    self.assertEqual("OK", custom_linop.solve(custom_linop))
  def testRegistrationFailures(self):
    class CustomLinOp(linear_operator.LinearOperator):
      pass
    with self.assertRaisesRegex(TypeError, "must be callable"):
      linear_operator_algebra.RegisterSolve(CustomLinOp, CustomLinOp)("blah")
    linear_operator_algebra.RegisterSolve(
        CustomLinOp, CustomLinOp)(lambda a: None)
    with self.assertRaisesRegex(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterSolve(
          CustomLinOp, CustomLinOp)(lambda a: None)
  def testExactSolveRegistrationsAllMatch(self):
    for (k, v) in _SOLVE.items():
      self.assertEqual(v, _registered_solve(k[0], k[1]))
class InverseTest(test.TestCase):
  def testRegistration(self):
    class CustomLinOp(linear_operator.LinearOperator):
      def _matmul(self, a):
        pass
      def _shape(self):
        return tensor_shape.TensorShape([1, 1])
      def _shape_tensor(self):
        pass
    @linear_operator_algebra.RegisterInverse(CustomLinOp)
      return "OK"
    with self.assertRaisesRegex(ValueError, "singular"):
      CustomLinOp(dtype=None, is_non_singular=False).inverse()
    self.assertEqual("OK", CustomLinOp(
        dtype=None, is_non_singular=True).inverse())
  def testRegistrationFailures(self):
    class CustomLinOp(linear_operator.LinearOperator):
      pass
    with self.assertRaisesRegex(TypeError, "must be callable"):
      linear_operator_algebra.RegisterInverse(CustomLinOp)("blah")
    linear_operator_algebra.RegisterInverse(CustomLinOp)(lambda a: None)
    with self.assertRaisesRegex(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterInverse(CustomLinOp)(lambda a: None)
  def testExactRegistrationsAllMatch(self):
    for (k, v) in _INVERSES.items():
      self.assertEqual(v, _registered_inverse(k[0]))
if __name__ == "__main__":
  test.main()
