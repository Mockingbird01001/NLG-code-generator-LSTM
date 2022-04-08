
from absl.testing import parameterized
import numpy as np
import scipy.special as sps
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
class TernaryOpsTest(xla_test.XLATestCase, parameterized.TestCase):
  def _testTernary(self, op, a, b, c, expected, rtol=1e-3, atol=1e-6):
    with self.session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name="b")
        pc = array_ops.placeholder(dtypes.as_dtype(c.dtype), c.shape, name="c")
        output = op(pa, pb, pc)
      result = session.run(output, {pa: a, pb: b, pc: c})
      self.assertAllClose(result, expected, rtol=rtol, atol=atol)
      return result
  @parameterized.parameters(
      {'start': 1, 'end': 2, 'num': 1},
      {'start': 1, 'end': 4, 'num': 3},
      {'start': 0, 'end': 41, 'num': 42})
  @test_util.disable_mlir_bridge(
      'TODO(b/156174708): Dynamic result types not supported')
  def testLinspace(self, start, end, num):
    expected = np.linspace(start, end, num, dtype=np.float32)
    result = self._testTernary(
        math_ops.linspace,
        np.float32(start),
        np.float32(end),
        np.int32(num),
        expected)
    self.assertEqual(result[-1], expected[-1])
    self.assertEqual(result[0], expected[0])
  def testRange(self):
    self._testTernary(
        math_ops.range,
        np.int32(1),
        np.int32(2),
        np.int32(1),
        expected=np.array([1], dtype=np.int32))
    self._testTernary(
        math_ops.range,
        np.int32(1),
        np.int32(7),
        np.int32(2),
        expected=np.array([1, 3, 5], dtype=np.int32))
  def testSelect(self):
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.where,
          np.array(False),
          np.array(2, dtype=dtype),
          np.array(7, dtype=dtype),
          expected=np.array(7, dtype=dtype))
      self._testTernary(
          array_ops.where,
          np.array(True),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([1, 2, 3, 4], dtype=dtype))
      self._testTernary(
          array_ops.where,
          np.array(False),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype))
      self._testTernary(
          array_ops.where,
          np.array([0, 1, 1, 0], dtype=np.bool_),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([5, 2, 3, 8], dtype=dtype))
      self._testTernary(
          array_ops.where,
          np.array([0, 1, 0], dtype=np.bool_),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [3, 4], [11, 12]], dtype=dtype))
  def testSelectV2(self):
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.where_v2,
          np.array(False),
          np.array(2, dtype=dtype),
          np.array(7, dtype=dtype),
          expected=np.array(7, dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array(True),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([1, 2, 3, 4], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array(False),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array([0, 1, 1, 0], dtype=np.bool_),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([5, 2, 3, 8], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array([0, 1], dtype=np.bool_),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 2], [9, 4], [11, 6]], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array([[0, 1], [1, 0], [1, 1]], dtype=np.bool_),
          np.array([[1, 2]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 2], [1, 10], [1, 2]], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [0, 0]], dtype=np.bool_),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[7, 2], [1, 10], [1, 2]], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [1, 1]], dtype=np.bool_),
          np.array(7, dtype=dtype),
          np.array(8, dtype=dtype),
          expected=np.array([[7, 8], [8, 7], [7, 7]], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [0, 0]], dtype=np.bool_),
          np.array(7, dtype=dtype),
          np.array([8, 9], dtype=dtype),
          expected=np.array([[7, 9], [8, 7], [8, 9]], dtype=dtype))
  def testSlice(self):
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.slice,
          np.array([[], [], []], dtype=dtype),
          np.array([1, 0], dtype=np.int32),
          np.array([2, 0], dtype=np.int32),
          expected=np.array([[], []], dtype=dtype))
      self._testTernary(
          array_ops.slice,
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype),
          np.array([0, 1], dtype=np.int32),
          np.array([2, 1], dtype=np.int32),
          expected=np.array([[2], [5]], dtype=dtype))
  def testClipByValue(self):
    for dtype in self.numeric_types - self.complex_types:
      test_cases = [
          (np.array([-2, 7, 7], dtype=dtype), np.array([-2, 9, 8], dtype=dtype))
      ]
      x = np.array([-2, 10, 6], dtype=dtype)
      for lower, upper in test_cases:
        self._testTernary(
            gen_math_ops._clip_by_value,
            x,
            lower,
            upper,
            expected=np.minimum(np.maximum(x, lower), upper))
  def testBetaincSanity(self):
    for dtype in self.numeric_types & {np.float32, np.float64}:
      a = np.array([.3, .4, .2, .2], dtype=dtype)
      b = np.array([1., 1., .4, .4], dtype=dtype)
      x = np.array([.3, .4, .0, .1], dtype=dtype)
      expected = sps.betainc(a, b, x)
      self._testTernary(
          math_ops.betainc, a, b, x, expected, rtol=5e-6, atol=6e-6)
  @parameterized.parameters(
      {
          'sigma': 1e15,
          'rtol': 1e-6,
          'atol': 1e-4
      },
      {
          'sigma': 30,
          'rtol': 1e-6,
          'atol': 2e-3
      },
      {
          'sigma': 1e-8,
          'rtol': 5e-4,
          'atol': 3e-4
      },
      {
          'sigma': 1e-16,
          'rtol': 1e-6,
          'atol': 2e-4
      },
  )
  def testBetainc(self, sigma, rtol, atol):
    for dtype in self.numeric_types & {np.float32, np.float64}:
      expected = sps.betainc(a, b, x, dtype=dtype)
      self._testTernary(
          math_ops.betainc, a, b, x, expected, rtol=rtol, atol=atol)
if __name__ == "__main__":
  googletest.main()
