
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
class Conv2DTest(test.TestCase):
  def __init__(self, method_name="runTest"):
    super(Conv2DTest, self).__init__(method_name)
  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    expected):
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    x1 = np.array([f for f in range(1, total_size_1 + 1)])
    x1 = x1.astype(np.uint8).reshape(tensor_in_sizes)
    x1_min = 0.0
    x1_max = 255.0
    x2 = np.array([f for f in range(1, total_size_2 + 1)]).astype(np.uint8)
    x2 = x2.astype(np.uint8).reshape(filter_in_sizes)
    x2_min = 0.0
    x2_max = 255.0
    with self.cached_session(use_gpu=False) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtypes.quint8)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtypes.quint8)
      conv = nn_ops.quantized_conv2d(
          t1,
          t2,
          out_type=dtypes.qint32,
          strides=[1, stride, stride, 1],
          padding=padding,
          min_input=x1_min,
          max_input=x1_max,
          min_filter=x2_min,
          max_filter=x2_max)
      value = self.evaluate(conv)
    quantized_output = value[0]
    output_min = value[1]
    output_max = value[2]
    float_output = self._QuantizedOutputToFloat(quantized_output, output_min,
                                                output_max)
    self.assertArrayNear(expected, float_output.flatten(), 1.0)
    self.assertEqual(value[0].shape, conv[0].get_shape())
  def _assertQuantizedArrayEquals(self, iarray1, iarray2):
    for i1, i2 in zip(iarray1, iarray2):
      self.assertTrue(i1 == i2)
  def _QuantizedOutputToFloat(self, quantized, quantized_min, quantized_max):
    number_of_bits = 32
    number_of_steps = 1 << number_of_bits
    range_adjust = (number_of_steps / (number_of_steps - 1.0))
    quantized_range = ((quantized_max - quantized_min) * range_adjust)
    range_scale = (quantized_range / number_of_steps)
    lowest_quantized = -(1 << (number_of_bits - 1))
    result = np.array([(quantized_min +
                        ((float(x) - lowest_quantized) * range_scale))
                       for x in quantized.flatten()])
    return result
  def testConv2D1x1Filter(self):
    expected_output = [
        30, 36, 42, 66, 81, 96, 102, 126, 150, 138, 171, 204, 174, 216, 258,
        210, 261, 312
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)
  def testConv2D2x2Filter(self):
    expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)
  def testConv2D1x2Filter(self):
    expected_output = [
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)
  def testConv2D2x2FilterStride2(self):
    expected_output = [2271.0, 2367.0, 2463.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=2,
        padding="VALID",
        expected=expected_output)
  def testConv2D2x2FilterStride2Same(self):
    expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=2,
        padding="SAME",
        expected=expected_output)
if __name__ == "__main__":
  test.main()
