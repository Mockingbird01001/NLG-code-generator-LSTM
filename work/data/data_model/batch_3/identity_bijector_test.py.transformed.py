
from tensorflow.python.framework import test_util
from tensorflow.python.ops.distributions import bijector_test_util
from tensorflow.python.ops.distributions import identity_bijector
from tensorflow.python.platform import test
class IdentityBijectorTest(test.TestCase):
  def testBijector(self):
    bijector = identity_bijector.Identity(validate_args=True)
    self.assertEqual("identity", bijector.name)
    x = [[[0.], [1.]]]
    self.assertAllEqual(x, self.evaluate(bijector.forward(x)))
    self.assertAllEqual(x, self.evaluate(bijector.inverse(x)))
    self.assertAllEqual(
        0.,
        self.evaluate(
            bijector.inverse_log_det_jacobian(x, event_ndims=3)))
    self.assertAllEqual(
        0.,
        self.evaluate(
            bijector.forward_log_det_jacobian(x, event_ndims=3)))
  @test_util.run_deprecated_v1
  def testScalarCongruency(self):
    with self.cached_session():
      bijector = identity_bijector.Identity()
      bijector_test_util.assert_scalar_congruency(
          bijector, lower_x=-2., upper_x=2.)
if __name__ == "__main__":
  test.main()
