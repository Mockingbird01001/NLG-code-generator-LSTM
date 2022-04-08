
import numpy as np
from absl.testing import parameterized
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import image_grad_test_base as test_base
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
class ResizeNearestNeighborOpDeterminismExceptionsTest(test.TestCase,
                                                       parameterized.TestCase):
  @parameterized.parameters(
      {
          'align_corners': False,
          'half_pixel_centers': False,
          'data_type': dtypes.float16
      }, {
          'align_corners': False,
          'half_pixel_centers': False,
          'data_type': dtypes.float32
      }, {
          'align_corners': False,
          'half_pixel_centers': False,
          'data_type': dtypes.float64
      }, {
          'align_corners': True,
          'half_pixel_centers': False,
          'data_type': dtypes.float32
      }, {
          'align_corners': False,
          'half_pixel_centers': True,
          'data_type': dtypes.float32
      })
  @test_util.run_gpu_only
  @test_util.run_all_in_graph_and_eager_modes
  def testExceptionThrowing(self, align_corners, half_pixel_centers, data_type):
    with self.session(), test_util.force_gpu():
      input_image = array_ops.zeros((1, 2, 2, 1), dtype=data_type)
      with backprop.GradientTape() as tape:
        tape.watch(input_image)
        output_image = image_ops.resize_nearest_neighbor(
            input_image, (3, 3),
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers)
      with self.assertRaisesRegex(
          errors.UnimplementedError,
          'A deterministic GPU implementation of ResizeNearestNeighborGrad' +
          ' is not currently available.'):
        gradient = tape.gradient(output_image, input_image)
        self.evaluate(gradient)
class ResizeBilinearOpDeterministicTest(test_base.ResizeBilinearOpTestBase):
  def _randomNDArray(self, shape):
    return 2 * np.random.random_sample(shape) - 1
  def _randomDataOp(self, shape, data_type):
    return constant_op.constant(self._randomNDArray(shape), dtype=data_type)
  @parameterized.parameters(
      {
          'align_corners': False,
          'half_pixel_centers': False,
          'data_type': dtypes.float32
      },
      {
          'align_corners': False,
          'half_pixel_centers': False,
          'data_type': dtypes.float64
      },
      {
          'align_corners': True,
          'half_pixel_centers': False,
          'data_type': dtypes.float32
      },
      {
          'align_corners': False,
          'half_pixel_centers': True,
          'data_type': dtypes.float32
      })
  @test_util.run_in_graph_and_eager_modes
  @test_util.run_gpu_only
  def testDeterministicGradients(self, align_corners, half_pixel_centers,
                                 data_type):
    if not align_corners and test_util.is_xla_enabled():
      self.skipTest('align_corners==False not currently supported by XLA')
    with self.session(force_gpu=True):
      seed = (
          hash(align_corners) % 256 + hash(half_pixel_centers) % 256 +
          hash(data_type) % 256)
      np.random.seed(seed)
      output_shape = (1, 200, 250, 3)
      input_image = self._randomDataOp(input_shape, data_type)
      repeat_count = 3
      if context.executing_eagerly():
        def resize_bilinear_gradients(local_seed):
          np.random.seed(local_seed)
          upstream_gradients = self._randomDataOp(output_shape, dtypes.float32)
          with backprop.GradientTape(persistent=True) as tape:
            tape.watch(input_image)
            output_image = image_ops.resize_bilinear(
                input_image,
                output_shape[1:3],
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers)
            gradient_injector_output = output_image * upstream_gradients
          return tape.gradient(gradient_injector_output, input_image)
        for i in range(repeat_count):
          result_a = resize_bilinear_gradients(local_seed)
          result_b = resize_bilinear_gradients(local_seed)
          self.assertAllEqual(result_a, result_b)
        upstream_gradients = array_ops.placeholder(
            dtypes.float32, shape=output_shape, name='upstream_gradients')
        output_image = image_ops.resize_bilinear(
            input_image,
            output_shape[1:3],
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers)
        gradient_injector_output = output_image * upstream_gradients
        resize_bilinear_gradients = gradients_impl.gradients(
            gradient_injector_output,
            input_image,
            grad_ys=None,
            colocate_gradients_with_ops=True)[0]
        for i in range(repeat_count):
          feed_dict = {upstream_gradients: self._randomNDArray(output_shape)}
          result_a = resize_bilinear_gradients.eval(feed_dict=feed_dict)
          result_b = resize_bilinear_gradients.eval(feed_dict=feed_dict)
          self.assertAllEqual(result_a, result_b)
class CropAndResizeOpDeterminismExceptionsTest(test.TestCase):
  def _genParams(self, dtype=dtypes.float32):
    batch_size = 1
    image_height = 10
    image_width = 10
    channels = 1
    image_shape = (batch_size, image_height, image_width, channels)
    num_boxes = 3
    boxes_shape = (num_boxes, 4)
    random_seed.set_seed(123)
    image = random_ops.random_normal(shape=image_shape, dtype=dtype)
    boxes = random_ops.random_uniform(shape=boxes_shape, dtype=dtypes.float32)
    box_indices = random_ops.random_uniform(
        shape=(num_boxes,), minval=0, maxval=batch_size, dtype=dtypes.int32)
    crop_size = constant_op.constant([3, 3], dtype=dtypes.int32)
    return image, boxes, box_indices, crop_size
  @test_util.run_in_graph_and_eager_modes
  @test_util.run_gpu_only
  def testExceptionThrowing(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      image, boxes, box_indices, crop_size = self._genParams(dtype)
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(image)
        tape.watch(boxes)
        op_output = image_ops.crop_and_resize_v2(image, boxes, box_indices,
                                                 crop_size)
      image_error_message = ('Deterministic GPU implementation of' +
                             ' CropAndResizeBackpropImage not available')
      with self.assertRaisesRegex(errors_impl.UnimplementedError,
                                  image_error_message):
        result = tape.gradient(op_output, image)
        self.evaluate(result)
      expected_error_message = ('Deterministic GPU implementation of' +
                                ' CropAndResizeBackpropBoxes not available')
      if context.executing_eagerly():
        expected_error_message = image_error_message
      with self.assertRaisesRegex(errors_impl.UnimplementedError,
                                  expected_error_message):
        result = tape.gradient(op_output, boxes)
        self.evaluate(result)
class CropAndResizeOpDeterministicTest(test_base.CropAndResizeOpTestBase):
  def _randomFloats(self, shape, low=0.0, high=1.0, dtype=dtypes.float32):
    val = np.random.random_sample(
    diff = high - low
    val *= diff
    val += low
    return constant_op.constant(val, dtype=dtype)
  def _randomInts(self, shape, low, high):
    val = np.random.randint(low=low, high=high, size=shape)
    return constant_op.constant(val, dtype=dtypes.int32)
  def _genParams(self, dtype=dtypes.float32):
    batch_size = 16
    input_height = 64
    input_width = 64
    depth = 1
    input_shape = (batch_size, input_height, input_width, depth)
    np.random.seed(456)
    image = self._randomFloats(input_shape, low=-1.0, high=1.0, dtype=dtype)
    box_count = 4 * batch_size
    boxes = self._randomFloats((box_count, 4),
                               low=0.0,
                               high=1.01,
                               dtype=dtypes.float32)
    box_indices = self._randomInts((box_count,), low=0, high=batch_size)
    crop_size = [input_height * 2, input_width * 2]
    output_shape = (box_count, *crop_size, depth)
    injected_gradients = self._randomFloats(
        output_shape, low=-0.001, high=0.001, dtype=dtypes.float32)
    return image, boxes, box_indices, crop_size, injected_gradients
  def _testReproducibleBackprop(self, test_image_not_boxes):
    with test_util.force_cpu():
      for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
        params = self._genParams(dtype)
        image, boxes, box_indices, crop_size, injected_gradients = params
        with backprop.GradientTape(persistent=True) as tape:
          tape.watch([image, boxes])
          output = image_ops.crop_and_resize_v2(
              image, boxes, box_indices, crop_size, method='bilinear')
          upstream = output * injected_gradients
        image_gradients_a, boxes_gradients_a = tape.gradient(
            upstream, [image, boxes])
        for _ in range(5):
          image_gradients_b, boxes_gradients_b = tape.gradient(
              upstream, [image, boxes])
          if test_image_not_boxes:
            self.assertAllEqual(image_gradients_a, image_gradients_b)
          else:
            self.assertAllEqual(boxes_gradients_a, boxes_gradients_b)
  @test_util.run_in_graph_and_eager_modes
  def testReproducibleBackpropToImage(self):
    self._testReproducibleBackprop(test_image_not_boxes=True)
  @test_util.run_in_graph_and_eager_modes
  def testReproducibleBackpropToBoxes(self):
    self._testReproducibleBackprop(test_image_not_boxes=False)
if __name__ == '__main__':
  config.enable_op_determinism()
  test.main()
