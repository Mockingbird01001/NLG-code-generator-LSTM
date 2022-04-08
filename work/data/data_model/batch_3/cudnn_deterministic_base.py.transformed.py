
import collections
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
LayerShapeNHWC = collections.namedtuple('LayerShapeNHWC',
                                        'batch, height, width, channels')
FilterShape2D = collections.namedtuple(
    'FilterShape2D', 'height, width, in_channels, out_channels')
LayerShapeNCDHW = collections.namedtuple(
    'LayerShapeNCDHW', 'batch, channels, depth, height, width')
FilterShape3D = collections.namedtuple(
    'FilterShape3D', 'depth, height, width, in_channels, out_channels')
class ConvolutionTest(test.TestCase):
  def _random_data_op(self, shape):
    return constant_op.constant(
        2 * np.random.random_sample(shape) - 1, dtype=dtypes.float32)
  def _random_out_op(self, in_shape, filter_shape, strides, padding):
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    conv_op = nn_ops.conv2d(in_op, filter_op, strides=strides, padding=padding)
    out_shape = conv_op.get_shape()
    out_op = self._random_data_op(out_shape)
    return out_op
  def _assert_reproducible(self, operation):
    with self.cached_session(force_gpu=True):
      result_1 = self.evaluate(operation)
      result_2 = self.evaluate(operation)
    self.assertAllEqual(result_1, result_2)
  @test_util.run_cuda_only
  def testForward(self):
    in_shape = LayerShapeNCDHW(batch=2, channels=3, depth=5, height=7, width=6)
    filter_shape = FilterShape3D(
        depth=3, height=3, width=3, in_channels=3, out_channels=2)
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    strides = [1, 1, 1, 1, 1]
    padding = 'VALID'
    dilations = [1, 1, 2, 2, 2]
    out_op = nn_ops.conv3d(
        in_op,
        filter_op,
        strides=strides,
        padding=padding,
        data_format='NCDHW',
        dilations=dilations)
    self._assert_reproducible(out_op)
  @test_util.run_cuda_only
  def testBackwardFilterGradient(self):
    in_shape = LayerShapeNHWC(batch=8, height=128, width=128, channels=8)
    filter_shape = FilterShape2D(
        height=3, width=3, in_channels=8, out_channels=8)
    in_op = self._random_data_op(in_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    out_op = self._random_out_op(in_shape, filter_shape, strides, padding)
    filter_gradient_op = nn_ops.conv2d_backprop_filter(
        in_op, filter_shape, out_op, strides=strides, padding=padding)
    self._assert_reproducible(filter_gradient_op)
  @test_util.run_cuda_only
  def testBackwardFilterGradientWithDilations(self):
    in_shape = LayerShapeNHWC(batch=8, height=128, width=128, channels=8)
    filter_shape = FilterShape2D(
        height=3, width=3, in_channels=8, out_channels=8)
    in_op = self._random_data_op(in_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    dilations = [1, 2, 2, 1]
    out_op = self._random_out_op(in_shape, filter_shape, strides, padding)
    filter_gradient_op = nn_ops.conv2d_backprop_filter(
        in_op, filter_shape, out_op, strides=strides, padding=padding,
        dilations=dilations)
    self._assert_reproducible(filter_gradient_op)
  @test_util.run_cuda_only
  def testBackwardInputGradient(self):
    in_shape = LayerShapeNHWC(batch=8, height=32, width=32, channels=8)
    filter_shape = FilterShape2D(
        height=7, width=7, in_channels=8, out_channels=128)
    filter_op = self._random_data_op(filter_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    out_op = self._random_out_op(in_shape, filter_shape, strides, padding)
    input_gradient_op = nn_ops.conv2d_backprop_input(
        in_shape, filter_op, out_op, strides=strides, padding=padding)
    self._assert_reproducible(input_gradient_op)
  @test_util.run_cuda_only
  def testBackwardInputGradientWithDilations(self):
    in_shape = LayerShapeNHWC(batch=8, height=32, width=32, channels=8)
    filter_shape = FilterShape2D(
        height=7, width=7, in_channels=8, out_channels=128)
    filter_op = self._random_data_op(filter_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    dilations = [1, 2, 2, 1]
    out_op = self._random_out_op(in_shape, filter_shape, strides, padding)
    input_gradient_op = nn_ops.conv2d_backprop_input(
        in_shape, filter_op, out_op, strides=strides, padding=padding,
        dilations=dilations)
    self._assert_reproducible(input_gradient_op)
