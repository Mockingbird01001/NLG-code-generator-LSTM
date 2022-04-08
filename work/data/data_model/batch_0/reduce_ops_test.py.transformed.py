
import functools
import itertools
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
@parameterized.named_parameters(('32_bit_index', dtypes.int32),
                                ('64_bit_index', dtypes.int64))
class ReduceOpsTest(xla_test.XLATestCase, parameterized.TestCase):
  def _testReduction(self,
                     tf_reduce_fn,
                     np_reduce_fn,
                     dtype,
                     test_inputs,
                     index_dtype,
                     rtol=1e-4,
                     atol=1e-4):
    for test_input in test_inputs:
      with self.session() as sess:
        with self.test_scope():
          a = array_ops.placeholder(dtype)
          index = array_ops.placeholder(index_dtype)
          out = tf_reduce_fn(a, index)
        result = sess.run(out, {a: test_input, index: [0]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=0), rtol=rtol, atol=atol)
        result = sess.run(out, {a: test_input, index: [1]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=1), rtol=rtol, atol=atol)
        result = sess.run(out, {a: test_input, index: [-1]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=1), rtol=rtol, atol=atol)
        if not test_util.is_mlir_bridge_enabled():
          with self.assertRaisesWithPredicateMatch(
              errors_impl.InvalidArgumentError, 'Invalid reduction dim'):
            sess.run(out, {a: test_input, index: [-33]})
          with self.assertRaisesWithPredicateMatch(
              errors_impl.InvalidArgumentError, 'Invalid reduction dim'):
            sess.run(out, {a: test_input, index: [2]})
  REAL_DATA = [
      np.zeros(shape=(2, 0)),
      np.zeros(shape=(0, 30)),
      np.arange(1, 7).reshape(2, 3),
      np.arange(-10, -4).reshape(2, 3),
      np.arange(-4, 2).reshape(2, 3),
  ]
  COMPLEX_DATA = [
      np.zeros(shape=(2, 0)).astype(np.complex64),
      np.zeros(shape=(0, 30)).astype(np.complex64),
      np.arange(1, 13, dtype=np.float32).view(np.complex64).reshape(2, 3),
      np.arange(-14, -2, dtype=np.float32).view(np.complex64).reshape(2, 3),
      np.arange(-4, 8, dtype=np.float32).view(np.complex64).reshape(2, 3),
  ]
  NONEMPTY_REAL_DATA = [x for x in REAL_DATA if np.size(x) > 0]
  NONEMPTY_COMPLEX_DATA = [x for x in COMPLEX_DATA if np.size(x) > 0]
  BOOL_DATA = [
      np.array([], dtype=np.bool_).reshape(2, 0),
      np.array([], dtype=np.bool_).reshape(0, 3),
      np.array([[False, True, False], [True, True, False]]),
  ]
  ONES = [np.ones([34000, 2])]
  def testReduceSumF32(self, index_dtype):
    self._testReduction(math_ops.reduce_sum, np.sum, np.float32, self.REAL_DATA,
                        index_dtype)
  def testReduceSumC64(self, index_dtype):
    self._testReduction(math_ops.reduce_sum, np.sum, np.complex64,
                        self.COMPLEX_DATA, index_dtype)
  def testReduceProdF32(self, index_dtype):
    self._testReduction(math_ops.reduce_prod, np.prod, np.float32,
                        self.REAL_DATA, index_dtype)
  def testReduceProdC64(self, index_dtype):
    self._testReduction(math_ops.reduce_prod, np.prod, np.complex64,
                        self.COMPLEX_DATA, index_dtype)
  def testReduceMin(self, index_dtype):
    def reference_min(dtype, inp, axis):
      if inp.shape[axis] == 0:
        if np.issubdtype(dtype, np.floating):
          return np.full(inp.shape[0:axis] + inp.shape[axis + 1:], float('inf'))
        return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                       np.iinfo(dtype).max)
      return np.amin(inp, axis)
    for dtype in set(self.all_types).intersection(
        [np.float32, np.int32, np.int64]):
      self._testReduction(math_ops.reduce_min,
                          functools.partial(reference_min, dtype), dtype,
                          self.REAL_DATA, index_dtype)
  def testReduceMax(self, index_dtype):
    def reference_max(dtype, inp, axis):
      if inp.shape[axis] == 0:
        if np.issubdtype(dtype, np.floating):
          return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                         float('-inf'))
        return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                       np.iinfo(dtype).min)
      return np.amax(inp, axis)
    for dtype in set(self.all_types).intersection(
        [np.float32, np.int32, np.int64]):
      self._testReduction(math_ops.reduce_max,
                          functools.partial(reference_max, dtype), dtype,
                          self.REAL_DATA, index_dtype)
  def testReduceMeanF32(self, index_dtype):
    self._testReduction(math_ops.reduce_mean, np.mean, np.float32,
                        self.NONEMPTY_REAL_DATA, index_dtype)
  def testReduceMeanF16(self, index_dtype):
    if np.float16 in self.all_types:
      self._testReduction(math_ops.reduce_mean, np.mean, np.float16, self.ONES,
                          index_dtype)
  def testReduceMeanC64(self, index_dtype):
    self._testReduction(math_ops.reduce_mean, np.mean, np.complex64,
                        self.NONEMPTY_COMPLEX_DATA, index_dtype)
  def testReduceAll(self, index_dtype):
    self._testReduction(math_ops.reduce_all, np.all, np.bool_, self.BOOL_DATA,
                        index_dtype)
  def testReduceAny(self, index_dtype):
    self._testReduction(math_ops.reduce_any, np.any, np.bool_, self.BOOL_DATA,
                        index_dtype)
  @test_util.disable_mlir_bridge('Error messages differ')
  def testReduceSumWithDuplicateAxes(self, index_dtype):
    with self.session() as sess:
      with self.test_scope():
        a = array_ops.placeholder(np.float32)
        index = array_ops.placeholder(np.int32)
        out = math_ops.reduce_sum(a, index)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          'Axes contains duplicate dimension'):
        sess.run(out, {a: [10, 20, 30], index: [0, 0]})
class ReduceOpPrecisionTest(xla_test.XLATestCase):
  def _testReduceSum(self,
                     expected_result,
                     dtype,
                     test_inputs,
                     rtol=1e-3,
                     atol=1e-4):
    for test_input in test_inputs:
      with self.session() as sess:
        with self.test_scope():
          a = array_ops.placeholder(dtype)
          index = array_ops.placeholder(dtypes.int32)
          out = math_ops.reduce_sum(a, index)
        result = sess.run(out, {
            a: np.array(test_input, dtype=dtype),
            index: [0]
        })
        self.assertAllClose(
            np.float32(result),
            np.float32(expected_result),
            rtol=rtol,
            atol=atol)
  def testReduceSumF16(self):
    if np.float16 not in self.all_types:
      return
    f16_max = np.finfo(np.float16).max
    self._testReduceSum(
        f16_max, np.float16,
        itertools.permutations([f16_max, f16_max, f16_max * (-1.0)], 3))
  def testReduceSumBF16(self):
    if dtypes.bfloat16.as_numpy_dtype not in self.all_types:
      return
    bf16_max = np.float32(dtypes.bfloat16.max)
    f32_max = dtypes.float32.max
    value = min(bf16_max, f32_max - bf16_max) / 2
    self._testReduceSum(
        dtypes.bfloat16.as_numpy_dtype(value), dtypes.bfloat16.as_numpy_dtype,
        itertools.permutations([bf16_max, value, bf16_max * (-1.0)], 3))
if __name__ == '__main__':
  googletest.main()
