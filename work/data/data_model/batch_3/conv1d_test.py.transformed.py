
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
class Conv1DTest(test.TestCase):
  def testBasic(self):
    optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
    for dtype in [dtypes.float16, dtypes.float32] + optional_float64:
      x = constant_op.constant([1, 2, 3, 4], dtype=dtype)
      filters = constant_op.constant([2, 1], dtype=dtype)
      for stride in [1, 2]:
        with self.cached_session(use_gpu=test.is_gpu_available()):
          c = nn_ops.conv1d(x, filters, stride, padding="VALID")
          reduced = array_ops.squeeze(c)
          output = self.evaluate(reduced)
          if stride == 1:
            self.assertEqual(len(output), 3)
            self.assertAllClose(output,
                                [2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4])
          else:
            self.assertEqual(len(output), 2)
            self.assertAllClose(output, [2 * 1 + 1 * 2, 2 * 3 + 1 * 4])
  def testExpandedBatch(self):
    x = constant_op.constant([1, 2, 3, 4], dtype=dtypes.float32)
    filters = constant_op.constant([2, 1], dtype=dtypes.float32)
    for stride in [1, 2]:
      with self.cached_session(use_gpu=test.is_gpu_available()):
        c = nn_ops.conv1d(x, filters, stride, padding="VALID")
        output = self.evaluate(reduced)
        if stride == 1:
          self.assertAllClose(output,
                              [[2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4],
                               [2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4]])
        else:
          self.assertAllClose(
              output,
              [[2 * 1 + 1 * 2, 2 * 3 + 1 * 4], [2 * 1 + 1 * 2, 2 * 3 + 1 * 4]])
  def testConv1DTranspose(self):
    with self.cached_session():
      stride = 2
      x_shape = [2, 4, 3]
      y_shape = [2, 9, 2]
      f_shape = [3, 2, 3]
      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv1d_transpose(
          x, f, y_shape, strides=stride, padding="VALID")
      value = self.evaluate(output)
      cache_values = np.zeros(y_shape, dtype=np.float32)
      pad = 1
      for n in range(x_shape[0]):
        for k in range(f_shape[1]):
          for w in range(pad, y_shape[1] - pad):
            target = 3.0
            w_in = w % stride == 0 and w > pad and w < y_shape[1] - 1 - pad
            if w_in:
              target += 3.0
            cache_values[n, w, k] = target
          cache_values[n, 0, k] = cache_values[n, 1, k]
          cache_values[n, -1, k] = cache_values[n, -2, k]
    self.assertAllClose(cache_values, value)
if __name__ == "__main__":
  test.main()
