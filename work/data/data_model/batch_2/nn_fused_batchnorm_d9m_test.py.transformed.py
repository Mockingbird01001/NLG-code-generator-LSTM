
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import test
@test_util.run_all_in_graph_and_eager_modes
class FusedBatchNormalizationDeterministicTest(test.TestCase,
                                               parameterized.TestCase):
  def _genParams(self, data_format, x_dtype, large_batch):
    if large_batch:
      batch_size = 5000
      height = width = 4
    else:
      batch_size = 10
      height = 5
      width = 5000
    channel_count = 3
    if data_format == 'NHWC':
      x_shape = (batch_size, height, width, channel_count)
      x_shape = (batch_size, channel_count, height, width)
    x = constant_op.constant(np.random.normal(size=x_shape), dtype=x_dtype)
    scale_shape = (channel_count,)
    scale = constant_op.constant(
        np.random.normal(size=scale_shape), dtype=dtypes.float32)
    offset = constant_op.constant(
        np.random.normal(size=scale_shape), dtype=dtypes.float32)
    mean = np.random.normal(size=scale_shape)
    variance = np.random.normal(size=scale_shape)
    y_shape = x_shape
    y_dtype = x_dtype
    upstream_gradients = constant_op.constant(
        np.random.normal(size=y_shape), dtype=y_dtype)
    return x, scale, offset, mean, variance, upstream_gradients
  @parameterized.parameters('NHWC', 'NCHW')
  def testForward(self, data_format):
    with self.cached_session():
      for large_batch in [False, True]:
          x, scale, offset, mean, variance, _ = self._genParams(
              data_format, x_dtype, large_batch)
          for is_training in [False, True]:
            op_output = nn_impl.fused_batch_norm(
                x,
                scale,
                offset,
                mean,
                variance,
                data_format=data_format,
                is_training=is_training,
                exponential_avg_factor=1.01)
            y_a, running_mean_a, running_var_a = op_output
            y_a = self.evaluate(y_a)
            if is_training:
              running_mean_a = self.evaluate(running_mean_a)
              running_var_a = self.evaluate(running_var_a)
            for _ in range(5):
              op_output_b = nn_impl.fused_batch_norm(
                  x,
                  scale,
                  offset,
                  mean,
                  variance,
                  data_format=data_format,
                  is_training=is_training,
                  exponential_avg_factor=1.01)
              y_b, running_mean_b, running_var_b = op_output_b
              y_b = self.evaluate(y_b)
              self.assertAllEqual(y_a, y_b)
              if is_training:
                running_mean_b = self.evaluate(running_mean_b)
                running_var_b = self.evaluate(running_var_b)
                self.assertAllEqual(running_mean_a, running_mean_b)
                self.assertAllEqual(running_var_a, running_var_b)
  @parameterized.parameters('NHWC', 'NCHW')
  @test_util.disable_xla('XLA is deterministic')
  def testBackward(self, data_format):
    with self.cached_session():
      for large_batch in [False, True]:
        params = self._genParams(data_format, dtypes.float32, large_batch)
        x, scale, offset, mean, variance, upstream_gradients = params
        for is_training in [False, True]:
          for backprop_to in [x, scale, offset]:
            with backprop.GradientTape(persistent=True) as tape:
              tape.watch(backprop_to)
              op_output = nn_impl.fused_batch_norm(
                  x,
                  scale,
                  offset,
                  mean,
                  variance,
                  data_format=data_format,
                  is_training=is_training,
                  exponential_avg_factor=0.99)
              gradient_injector_output = op_output[0] * upstream_gradients
            if (len(config.list_physical_devices('GPU')) and
                not is_training):
              with self.assertRaisesRegex(
                  errors_impl.UnimplementedError,
                  'A deterministic GPU implementation of fused batch-norm' +
                  ' backprop, when training is disabled, is not currently' +
                  ' available.'):
                grad = tape.gradient(gradient_injector_output, backprop_to)
                self.evaluate(grad)
            else:
              grad_a = tape.gradient(gradient_injector_output, backprop_to)
              grad_a = self.evaluate(grad_a)
              for _ in range(3):
                grad_b = tape.gradient(gradient_injector_output,
                                       backprop_to)
                grad_b = self.evaluate(grad_b)
                self.assertAllEqual(grad_a, grad_b)
if __name__ == '__main__':
  config.enable_op_determinism()
  test.main()
