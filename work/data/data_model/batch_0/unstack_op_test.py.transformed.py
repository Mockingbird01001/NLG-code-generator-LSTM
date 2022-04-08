
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test
def np_split_squeeze(array, axis):
  axis_len = array.shape[axis]
  return [
      np.squeeze(
          arr, axis=(axis,)) for arr in np.split(
              array, axis_len, axis=axis)
  ]
class UnstackOpTest(test.TestCase):
  def randn(self, shape, dtype):
    data = np.random.randn(*shape)
    if dtype == np.bool_:
    else:
      return data.astype(dtype)
  def unstackReference(self, data, axis):
    result = []
    rank = len(data.shape)
    axis = axis + rank if axis < 0 else axis
    for k in range(data.shape[axis]):
      axis = rank + axis if axis < 0 else axis
      slice_spec = tuple(
          slice(None) if i != axis else k for i in range(rank))
      result.append(data.__getitem__(slice_spec))
    return result
  def testSimple(self):
    np.random.seed(7)
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      rank = len(shape)
      for axis in range(-rank, rank):
        for dtype in [
            np.bool_, np.float16, np.float32, np.float64, np.uint8, np.int32,
            np.int64
        ]:
          data = self.randn(shape, dtype)
          x = constant_op.constant(data)
          ref = self.unstackReference(data, axis)
          cs = array_ops.unstack(x, axis=axis)
          self.assertEqual(type(cs), list)
          self.assertEqual(len(cs), shape[axis])
          for k, c in enumerate(cs):
            with self.subTest(shape=shape, k=k, axis=axis, dtype=dtype):
              self.assertAllEqual(ref[k], self.evaluate(c))
  def testSimpleGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest('No GPU available')
    np.random.seed(7)
    with test_util.force_gpu():
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        rank = len(shape)
        for axis in range(-rank, rank):
          for dtype in [
              np.bool_, np.float16, np.float32, np.float64, np.uint8, np.int32,
              np.int64
          ]:
            data = self.randn(shape, dtype)
            x = constant_op.constant(data)
            ref = self.unstackReference(data, axis)
            cs = array_ops.unstack(x, axis=axis)
            self.assertEqual(type(cs), list)
            self.assertEqual(len(cs), shape[axis])
            for k, c in enumerate(cs):
              with self.subTest(shape=shape, k=k, axis=axis, dtype=dtype):
                self.assertAllEqual(ref[k], self.evaluate(c))
  def testGradientsAxis0(self):
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      x = constant_op.constant(data)
      for i in range(shape[0]):
        def func(x, shape=shape, i=i):
          return array_ops.unstack(x, num=shape[0])[i]
        with self.cached_session():
          err = gradient_checker_v2.max_error(
              *gradient_checker_v2.compute_gradient(func, [x]))
          self.assertLess(err, 1e-6)
  def testGradientsAxis1(self):
    for shape in (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      x = constant_op.constant(data)
      for i in range(shape[1]):
        def func(x, shape=shape, i=i):
          return array_ops.unstack(x, num=shape[1], axis=1)[i]
        with self.cached_session():
          err = gradient_checker_v2.max_error(
              *gradient_checker_v2.compute_gradient(func, [x]))
          self.assertLess(err, 1e-6)
  def testInferNum(self):
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      x = array_ops.ones(shape, dtype=np.float32)
      cs = array_ops.unstack(x)
      self.assertEqual(type(cs), list)
      self.assertEqual(len(cs), shape[0])
  def testCannotInferNumFromUnknownShape(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(np.float32)
      with self.assertRaisesRegex(
          ValueError, r'Cannot infer argument `num` from shape <unknown>'):
        array_ops.unstack(x)
  def testUnknownShapeOkWithNum(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(np.float32)
      array_ops.unstack(x, num=2)
  def testCannotInferNumFromNoneShape(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(np.float32, shape=(None,))
      with self.assertRaisesRegex(
          ValueError, r'Cannot infer argument `num` from shape \((\?|None),\)'):
        array_ops.unstack(x)
  def testAgainstNumpy(self):
    for i in range(1, 6):
      a = np.random.random(np.random.permutation(i) + 1)
      for j in range(-i, i):
        expected = np_split_squeeze(a, j)
        actual_unstack = self.evaluate(array_ops.unstack(a, axis=j))
        self.assertAllEqual(expected, actual_unstack)
  def testAxis0Default(self):
    a = constant_op.constant([[1, 2, 3], [4, 5, 6]], name='a')
    unstacked = self.evaluate(array_ops.unstack(a))
    self.assertEqual(len(unstacked), 2)
    self.assertAllEqual(unstacked[0], [1, 2, 3])
    self.assertAllEqual(unstacked[1], [4, 5, 6])
  def testAxisOutOfRange(self):
    a = constant_op.constant([[1, 2, 3], [4, 5, 6]], name='a')
    with self.assertRaisesRegex(ValueError,
                                r'Argument `axis` = 2 not in range \[-2, 2\)'):
      array_ops.unstack(a, axis=2)
  def testAxisOutOfNegativeRange(self):
    a = constant_op.constant([[1, 2, 3], [4, 5, 6]], name='a')
    with self.assertRaisesRegex(ValueError,
                                r'Argument `axis` = -3 not in range \[-2, 2\)'):
      array_ops.unstack(a, axis=-3)
  def testZeroLengthDim(self):
    x = array_ops.zeros(shape=(0, 1, 2))
    y = self.evaluate(array_ops.unstack(x, axis=1)[0])
    self.assertEqual(y.shape, (0, 2))
  def testComplexGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest('No GPU available')
    np.random.seed(7)
    with test_util.force_gpu():
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        for dtype in [np.complex64, np.complex128]:
          data = np.random.randn(*shape).astype(dtype)
          x = constant_op.constant(data)
          cs = array_ops.unstack(x, num=shape[0])
          self.assertEqual(type(cs), list)
          self.assertEqual(len(cs), shape[0])
          cs = [self.evaluate(c) for c in cs]
          self.assertAllEqual(cs, data)
if __name__ == '__main__':
  test.main()
