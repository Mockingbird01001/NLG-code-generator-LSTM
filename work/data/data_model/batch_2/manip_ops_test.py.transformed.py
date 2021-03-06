
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import manip_ops
from tensorflow.python.platform import test as test_lib
try:
  from distutils.version import StrictVersion as Version
  NP_ROLL_CAN_MULTISHIFT = Version(np.version.version) >= Version("1.12.0")
except ImportError:
  NP_ROLL_CAN_MULTISHIFT = False
class RollTest(test_util.TensorFlowTestCase):
  def _testRoll(self, np_input, shift, axis):
    expected_roll = np.roll(np_input, shift, axis)
    with self.cached_session():
      roll = manip_ops.roll(np_input, shift, axis)
      self.assertAllEqual(roll, expected_roll)
  def _testGradient(self, np_input, shift, axis):
    with self.cached_session():
      inx = constant_op.constant(np_input.tolist())
      xs = list(np_input.shape)
      y = manip_ops.roll(inx, shift, axis)
      ys = xs
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, y, ys, x_init_value=np_input)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)
  def _testAll(self, np_input, shift, axis):
    self._testRoll(np_input, shift, axis)
    if np_input.dtype == np.float32:
      self._testGradient(np_input, shift, axis)
  @test_util.run_deprecated_v1
  def testIntTypes(self):
    for t in [np.int32, np.int64]:
      self._testAll(np.random.randint(-100, 100, (5)).astype(t), 3, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        self._testAll(
            np.random.randint(-100, 100, (4, 4, 3)).astype(t), [1, -2, 3],
            [0, 1, 2])
        self._testAll(
            np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t), [0, 1, -2],
            [1, 2, 3])
  @test_util.run_deprecated_v1
  def testFloatTypes(self):
    for t in [np.float32, np.float64]:
      self._testAll(np.random.rand(5).astype(t), 2, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        self._testAll(np.random.rand(3, 4).astype(t), [1, 2], [1, 0])
        self._testAll(np.random.rand(1, 3, 4).astype(t), [1, 0, -3], [0, 1, 2])
  @test_util.run_deprecated_v1
  def testComplexTypes(self):
    for t in [np.complex64, np.complex128]:
      x = np.random.rand(4, 4).astype(t)
      self._testAll(x + 1j * x, 2, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        x = np.random.rand(2, 5).astype(t)
        self._testAll(x + 1j * x, [1, 2], [1, 0])
        x = np.random.rand(3, 2, 1, 1).astype(t)
        self._testAll(x + 1j * x, [2, 1, 1, 0], [0, 3, 1, 2])
  @test_util.run_deprecated_v1
  def testNegativeAxis(self):
    self._testAll(np.random.randint(-100, 100, (5)).astype(np.int32), 3, -1)
    self._testAll(np.random.randint(-100, 100, (4, 4)).astype(np.int32), 3, -2)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is out of range"):
        manip_ops.roll(np.random.randint(-100, 100, (4, 4)).astype(np.int32),
                       3, -10).eval()
  @test_util.run_deprecated_v1
  def testEmptyInput(self):
    self._testAll(np.zeros([0, 1]), 1, 1)
    self._testAll(np.zeros([1, 0]), 1, 1)
  @test_util.run_deprecated_v1
  def testInvalidInputShape(self):
    with self.assertRaisesRegex(ValueError,
                                "Shape must be at least rank 1 but is rank 0"):
      manip_ops.roll(7, 1, 0)
  @test_util.run_deprecated_v1
  def testRollInputMustVectorHigherRaises(self):
    tensor = array_ops.placeholder(dtype=dtypes.int32)
    shift = 1
    axis = 0
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "input must be 1-D or higher"):
        manip_ops.roll(tensor, shift, axis).eval(feed_dict={tensor: 7})
  @test_util.run_deprecated_v1
  def testInvalidAxisShape(self):
    with self.assertRaisesRegex(ValueError,
                                "Shape must be at most rank 1 but is rank 2"):
      manip_ops.roll([[1, 2], [3, 4]], 1, [[0, 1]])
  @test_util.run_deprecated_v1
  def testRollAxisMustBeScalarOrVectorRaises(self):
    tensor = [[1, 2], [3, 4]]
    shift = 1
    axis = array_ops.placeholder(dtype=dtypes.int32)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "axis must be a scalar or a 1-D vector"):
        manip_ops.roll(tensor, shift, axis).eval(feed_dict={axis: [[0, 1]]})
  @test_util.run_deprecated_v1
  def testInvalidShiftShape(self):
    with self.assertRaisesRegex(ValueError,
                                "Shape must be at most rank 1 but is rank 2"):
      manip_ops.roll([[1, 2], [3, 4]], [[0, 1]], 1)
  @test_util.run_deprecated_v1
  def testRollShiftMustBeScalarOrVectorRaises(self):
    tensor = [[1, 2], [3, 4]]
    shift = array_ops.placeholder(dtype=dtypes.int32)
    axis = 1
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "shift must be a scalar or a 1-D vector"):
        manip_ops.roll(tensor, shift, axis).eval(feed_dict={shift: [[0, 1]]})
  @test_util.run_deprecated_v1
  def testInvalidShiftAndAxisNotEqualShape(self):
    with self.assertRaisesRegex(ValueError, "both shapes must be equal"):
      manip_ops.roll([[1, 2], [3, 4]], [1], [0, 1])
  @test_util.run_deprecated_v1
  def testRollShiftAndAxisMustBeSameSizeRaises(self):
    tensor = [[1, 2], [3, 4]]
    shift = array_ops.placeholder(dtype=dtypes.int32)
    axis = [0, 1]
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "shift and axis must have the same size"):
        manip_ops.roll(tensor, shift, axis).eval(feed_dict={shift: [1]})
  def testRollAxisOutOfRangeRaises(self):
    tensor = [1, 2]
    shift = 1
    axis = 1
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is out of range"):
        manip_ops.roll(tensor, shift, axis).eval()
if __name__ == "__main__":
  test_lib.main()
