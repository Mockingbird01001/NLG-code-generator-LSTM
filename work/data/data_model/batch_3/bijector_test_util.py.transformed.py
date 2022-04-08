
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib
def assert_finite(array):
  if not np.isfinite(array).all():
    raise AssertionError("array was not all finite. %s" % array[:15])
def assert_strictly_increasing(array):
  np.testing.assert_array_less(0., np.diff(array))
def assert_strictly_decreasing(array):
  np.testing.assert_array_less(np.diff(array), 0.)
def assert_strictly_monotonic(array):
  if array[0] < array[-1]:
    assert_strictly_increasing(array)
  else:
    assert_strictly_decreasing(array)
def assert_scalar_congruency(bijector,
                             lower_x,
                             upper_x,
                             n=int(10e3),
                             rtol=0.01,
                             sess=None):
  """Assert `bijector`'s forward/inverse/inverse_log_det_jacobian are congruent.
  We draw samples `X ~ U(lower_x, upper_x)`, then feed these through the
  `bijector` in order to check that:
  1. the forward is strictly monotonic.
  2. the forward/inverse methods are inverses of each other.
  3. the jacobian is the correct change of measure.
  This can only be used for a Bijector mapping open subsets of the real line
  to themselves.  This is due to the fact that this test compares the `prob`
  before/after transformation with the Lebesgue measure on the line.
  Args:
    bijector:  Instance of Bijector
    lower_x:  Python scalar.
    upper_x:  Python scalar.  Must have `lower_x < upper_x`, and both must be in
      the domain of the `bijector`.  The `bijector` should probably not produce
      huge variation in values in the interval `(lower_x, upper_x)`, or else
      the variance based check of the Jacobian will require small `rtol` or
      huge `n`.
    n:  Number of samples to draw for the checks.
    rtol:  Positive number.  Used for the Jacobian check.
    sess:  `tf.compat.v1.Session`.  Defaults to the default session.
  Raises:
    AssertionError:  If tests fail.
  """
  if sess is None:
    sess = ops.get_default_session()
  ten_x_pts = np.linspace(lower_x, upper_x, num=10).astype(np.float32)
  if bijector.dtype is not None:
    ten_x_pts = ten_x_pts.astype(bijector.dtype.as_numpy_dtype)
  forward_on_10_pts = bijector.forward(ten_x_pts)
  lower_y, upper_y = sess.run(
      [bijector.forward(lower_x), bijector.forward(upper_x)])
    lower_y, upper_y = upper_y, lower_y
  uniform_x_samps = uniform_lib.Uniform(
      low=lower_x, high=upper_x).sample(n, seed=0)
  uniform_y_samps = uniform_lib.Uniform(
      low=lower_y, high=upper_y).sample(n, seed=1)
  inverse_forward_x = bijector.inverse(bijector.forward(uniform_x_samps))
  forward_inverse_y = bijector.forward(bijector.inverse(uniform_y_samps))
  dy_dx = math_ops.exp(bijector.inverse_log_det_jacobian(
      uniform_y_samps, event_ndims=0))
  expectation_of_dy_dx_under_uniform = math_ops.reduce_mean(dy_dx)
  change_measure_dy_dx = (
      (upper_y - lower_y) * expectation_of_dy_dx_under_uniform)
  dx_dy = math_ops.exp(
      bijector.forward_log_det_jacobian(
          bijector.inverse(uniform_y_samps), event_ndims=0))
  [
      forward_on_10_pts_v,
      dy_dx_v,
      dx_dy_v,
      change_measure_dy_dx_v,
      uniform_x_samps_v,
      uniform_y_samps_v,
      inverse_forward_x_v,
      forward_inverse_y_v,
  ] = sess.run([
      forward_on_10_pts,
      dy_dx,
      dx_dy,
      change_measure_dy_dx,
      uniform_x_samps,
      uniform_y_samps,
      inverse_forward_x,
      forward_inverse_y,
  ])
  assert_strictly_monotonic(forward_on_10_pts_v)
  np.testing.assert_allclose(
      inverse_forward_x_v, uniform_x_samps_v, atol=1e-5, rtol=1e-3)
  np.testing.assert_allclose(
      forward_inverse_y_v, uniform_y_samps_v, atol=1e-5, rtol=1e-3)
  np.testing.assert_allclose(
      upper_x - lower_x, change_measure_dy_dx_v, atol=0, rtol=rtol)
  np.testing.assert_allclose(
      dy_dx_v, np.divide(1., dx_dy_v), atol=1e-5, rtol=1e-3)
def assert_bijective_and_finite(
    bijector, x, y, event_ndims, atol=0, rtol=1e-5, sess=None):
  """Assert that forward/inverse (along with jacobians) are inverses and finite.
  It is recommended to use x and y values that are very very close to the edge
  of the Bijector's domain.
  Args:
    bijector:  A Bijector instance.
    x:  np.array of values in the domain of bijector.forward.
    y:  np.array of values in the domain of bijector.inverse.
    event_ndims: Integer describing the number of event dimensions this bijector
      operates on.
    atol:  Absolute tolerance.
    rtol:  Relative tolerance.
    sess:  TensorFlow session.  Defaults to the default session.
  Raises:
    AssertionError:  If tests fail.
  """
  sess = sess or ops.get_default_session()
  assert_finite(x)
  assert_finite(y)
  f_x = bijector.forward(x)
  g_y = bijector.inverse(y)
  [
      x_from_x,
      y_from_y,
      ildj_f_x,
      fldj_x,
      ildj_y,
      fldj_g_y,
      f_x_v,
      g_y_v,
  ] = sess.run([
      bijector.inverse(f_x),
      bijector.forward(g_y),
      bijector.inverse_log_det_jacobian(f_x, event_ndims=event_ndims),
      bijector.forward_log_det_jacobian(x, event_ndims=event_ndims),
      bijector.inverse_log_det_jacobian(y, event_ndims=event_ndims),
      bijector.forward_log_det_jacobian(g_y, event_ndims=event_ndims),
      f_x,
      g_y,
  ])
  assert_finite(x_from_x)
  assert_finite(y_from_y)
  assert_finite(ildj_f_x)
  assert_finite(fldj_x)
  assert_finite(ildj_y)
  assert_finite(fldj_g_y)
  assert_finite(f_x_v)
  assert_finite(g_y_v)
  np.testing.assert_allclose(x_from_x, x, atol=atol, rtol=rtol)
  np.testing.assert_allclose(y_from_y, y, atol=atol, rtol=rtol)
  np.testing.assert_allclose(-ildj_f_x, fldj_x, atol=atol, rtol=rtol)
  np.testing.assert_allclose(-ildj_y, fldj_g_y, atol=atol, rtol=rtol)
