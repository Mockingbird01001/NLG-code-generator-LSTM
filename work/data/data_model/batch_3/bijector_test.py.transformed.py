
import abc
import numpy as np
import six
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.platform import test
@test_util.run_all_in_graph_and_eager_modes
class BaseBijectorTest(test.TestCase):
  def testIsAbstract(self):
    with self.assertRaisesRegex(TypeError,
                                ("Can't instantiate abstract class Bijector "
                                 "with abstract methods? __init__")):
  def testDefaults(self):
    class _BareBonesBijector(bijector.Bijector):
      def __init__(self):
        super(_BareBonesBijector, self).__init__(forward_min_event_ndims=0)
    bij = _BareBonesBijector()
    self.assertEqual([], bij.graph_parents)
    self.assertEqual(False, bij.is_constant_jacobian)
    self.assertEqual(False, bij.validate_args)
    self.assertEqual(None, bij.dtype)
    self.assertEqual("bare_bones_bijector", bij.name)
    for shape in [[], [1, 2], [1, 2, 3]]:
      forward_event_shape_ = self.evaluate(
          bij.inverse_event_shape_tensor(shape))
      inverse_event_shape_ = self.evaluate(
          bij.forward_event_shape_tensor(shape))
      self.assertAllEqual(shape, forward_event_shape_)
      self.assertAllEqual(shape, bij.forward_event_shape(shape))
      self.assertAllEqual(shape, inverse_event_shape_)
      self.assertAllEqual(shape, bij.inverse_event_shape(shape))
    with self.assertRaisesRegex(NotImplementedError, "inverse not implemented"):
      bij.inverse(0)
    with self.assertRaisesRegex(NotImplementedError, "forward not implemented"):
      bij.forward(0)
    with self.assertRaisesRegex(NotImplementedError,
                                "inverse_log_det_jacobian not implemented"):
      bij.inverse_log_det_jacobian(0, event_ndims=0)
    with self.assertRaisesRegex(NotImplementedError,
                                "forward_log_det_jacobian not implemented"):
      bij.forward_log_det_jacobian(0, event_ndims=0)
class IntentionallyMissingError(Exception):
  pass
class BrokenBijector(bijector.Bijector):
  def __init__(
      self, forward_missing=False, inverse_missing=False, validate_args=False):
    super(BrokenBijector, self).__init__(
        validate_args=validate_args, forward_min_event_ndims=0, name="broken")
    self._forward_missing = forward_missing
    self._inverse_missing = inverse_missing
  def _forward(self, x):
    if self._forward_missing:
      raise IntentionallyMissingError
    return 2 * x
  def _inverse(self, y):
    if self._inverse_missing:
      raise IntentionallyMissingError
    return y / 2.
    if self._inverse_missing:
      raise IntentionallyMissingError
    return -math_ops.log(2.)
    if self._forward_missing:
      raise IntentionallyMissingError
    return math_ops.log(2.)
class BijectorTestEventNdims(test.TestCase):
  def testBijectorNonIntegerEventNdims(self):
    bij = BrokenBijector()
    with self.assertRaisesRegex(ValueError, "Expected integer"):
      bij.forward_log_det_jacobian(1., event_ndims=1.5)
    with self.assertRaisesRegex(ValueError, "Expected integer"):
      bij.inverse_log_det_jacobian(1., event_ndims=1.5)
  def testBijectorArrayEventNdims(self):
    bij = BrokenBijector()
    with self.assertRaisesRegex(ValueError, "Expected scalar"):
      bij.forward_log_det_jacobian(1., event_ndims=(1, 2))
    with self.assertRaisesRegex(ValueError, "Expected scalar"):
      bij.inverse_log_det_jacobian(1., event_ndims=(1, 2))
  @test_util.run_deprecated_v1
  def testBijectorDynamicEventNdims(self):
    bij = BrokenBijector(validate_args=True)
    event_ndims = array_ops.placeholder(dtype=np.int32, shape=None)
    with self.cached_session():
      with self.assertRaisesOpError("Expected scalar"):
        bij.forward_log_det_jacobian(1., event_ndims=event_ndims).eval({
            event_ndims: (1, 2)})
      with self.assertRaisesOpError("Expected scalar"):
        bij.inverse_log_det_jacobian(1., event_ndims=event_ndims).eval({
            event_ndims: (1, 2)})
@six.add_metaclass(abc.ABCMeta)
class BijectorCachingTestBase(object):
  @abc.abstractproperty
  def broken_bijector_cls(self):
    raise IntentionallyMissingError("Not implemented")
  def testCachingOfForwardResults(self):
    broken_bijector = self.broken_bijector_cls(inverse_missing=True)
    x = constant_op.constant(1.1)
    y = broken_bijector.forward(x)
    _ = broken_bijector.forward_log_det_jacobian(x, event_ndims=0)
    broken_bijector.inverse(y)
    broken_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    with self.assertRaises(IntentionallyMissingError):
      broken_bijector.inverse_log_det_jacobian(y, event_ndims=1)
  def testCachingOfInverseResults(self):
    broken_bijector = self.broken_bijector_cls(forward_missing=True)
    y = constant_op.constant(1.1)
    x = broken_bijector.inverse(y)
    _ = broken_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    broken_bijector.forward(x)
    broken_bijector.forward_log_det_jacobian(x, event_ndims=0)
    with self.assertRaises(IntentionallyMissingError):
      broken_bijector.forward_log_det_jacobian(x, event_ndims=1)
class BijectorCachingTest(BijectorCachingTestBase, test.TestCase):
  @property
  def broken_bijector_cls(self):
    return BrokenBijector
class ExpOnlyJacobian(bijector.Bijector):
  def __init__(self, forward_min_event_ndims=0):
    super(ExpOnlyJacobian, self).__init__(
        validate_args=False,
        is_constant_jacobian=False,
        forward_min_event_ndims=forward_min_event_ndims,
        name="exp")
  def _inverse_log_det_jacobian(self, y):
    return -math_ops.log(y)
  def _forward_log_det_jacobian(self, x):
    return math_ops.log(x)
class ConstantJacobian(bijector.Bijector):
  def __init__(self, forward_min_event_ndims=0):
    super(ConstantJacobian, self).__init__(
        validate_args=False,
        is_constant_jacobian=True,
        forward_min_event_ndims=forward_min_event_ndims,
        name="c")
  def _inverse_log_det_jacobian(self, y):
    return constant_op.constant(2., y.dtype)
  def _forward_log_det_jacobian(self, x):
    return constant_op.constant(-2., x.dtype)
class BijectorReduceEventDimsTest(test.TestCase):
  def testReduceEventNdimsForward(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian()
    self.assertAllClose(
        np.log(x),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        np.sum(np.log(x), axis=-1),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(np.log(x), axis=(-1, -2)),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=2)))
  def testReduceEventNdimsForwardRaiseError(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    with self.assertRaisesRegex(ValueError, "must be larger than"):
      bij.forward_log_det_jacobian(x, event_ndims=0)
  def testReduceEventNdimsInverse(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian()
    self.assertAllClose(
        -np.log(x),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=-1),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=(-1, -2)),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=2)))
  def testReduceEventNdimsInverseRaiseError(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    with self.assertRaisesRegex(ValueError, "must be larger than"):
      bij.inverse_log_det_jacobian(x, event_ndims=0)
  def testReduceEventNdimsForwardConstJacobian(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ConstantJacobian()
    self.assertAllClose(
        -2.,
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        -4.,
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        -8.,
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=2)))
  def testReduceEventNdimsInverseConstJacobian(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ConstantJacobian()
    self.assertAllClose(
        2.,
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        4.,
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        8.,
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=2)))
  @test_util.run_deprecated_v1
  def testHandlesNonStaticEventNdims(self):
    x_ = [[[1., 2.], [3., 4.]]]
    x = array_ops.placeholder_with_default(x_, shape=None)
    event_ndims = array_ops.placeholder(dtype=np.int32, shape=[])
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    bij.inverse_log_det_jacobian(x, event_ndims=event_ndims)
    with self.cached_session() as sess:
      ildj = sess.run(bij.inverse_log_det_jacobian(x, event_ndims=event_ndims),
                      feed_dict={event_ndims: 1})
    self.assertAllClose(-np.log(x_), ildj)
if __name__ == "__main__":
  test.main()
