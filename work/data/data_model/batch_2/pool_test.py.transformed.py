
import math
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
def pool_direct_single_axis(
    axis,
    window_size,
    pooling_type,
    padding,
    dilation_rate,
    stride):
  effective_window_size = (window_size - 1) * dilation_rate + 1
  input_size = input.shape[axis]
  if padding == "SAME":
    output_size = int(math.ceil(input_size / stride))
    total_padding_amount = max(0, (output_size - 1) * stride +
                               effective_window_size - input_size)
    before_padding = total_padding_amount // 2
  elif padding == "VALID":
    output_size = int(
        math.ceil((input_size - effective_window_size + 1) / stride))
    before_padding = 0
  else:
    raise ValueError("Unsupported padding type: %r" % (padding,))
  output_shape = input.shape[:axis] + (output_size,) + input.shape[axis + 1:]
  output = np.zeros(output_shape, input.dtype)
  initial_dim_selector = tuple(np.s_[:] for _ in range(axis))
  if pooling_type == "MAX":
    pooling_func = np.max
  elif pooling_type == "AVG":
    pooling_func = np.mean
  else:
    raise ValueError("Unsupported pooling type: %r" % (pooling_type,))
  for output_pos in range(output_size):
    input_start_pos = output_pos * stride - before_padding
    input_end_pos = min(input_start_pos + effective_window_size, input_size)
    if input_start_pos < 0:
      input_start_pos += dilation_rate
    input_slice = np.s_[input_start_pos:input_end_pos:dilation_rate]
    output[initial_dim_selector + (output_pos,)] = pooling_func(
        input[initial_dim_selector + (input_slice,)], axis=axis)
  return output
def pool_direct(
    window_shape,
    pooling_type,
    dilation_rate,
    strides,
    data_format=None):
  if data_format is None or not data_format.startswith("NC"):
    spatial_start_dim = 1
  else:
    spatial_start_dim = 2
  output = input
  for i in range(len(window_shape)):
    output = pool_direct_single_axis(
        input=output,
        axis=i + spatial_start_dim,
        window_size=window_shape[i],
        pooling_type=pooling_type,
        padding=padding,
        dilation_rate=dilation_rate[i],
        stride=strides[i])
  return output
class PoolingTest(test.TestCase):
  def _test(self, input_shape, **kwargs):
    x = -np.arange(
        np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1
    y1 = pool_direct(input=x, **kwargs)
    y2 = nn_ops.pool(input=x, **kwargs)
    self.assertAllClose(y1, self.evaluate(y2), rtol=1e-2, atol=1e-2)
  def testPoolSimple(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          self._test(
              input_shape=[1, 1, 10, 1],
              window_shape=[1, 3],
              padding=padding,
              pooling_type=pooling_type,
              dilation_rate=[1, 1],
              strides=[1, 2])
  def testPool1D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          for input_shape in [[2, 9, 2], [2, 10, 2]]:
            for window_shape in [[1], [2], [3]]:
              if padding != "SAME":
                for dilation_rate in [[1], [2], [3]]:
                  self._test(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1])
              for strides in [[1], [2], [3]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1],
                    strides=strides)
  def testPool2D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          for input_shape in [[2, 9, 10, 2], [2, 10, 9, 2]]:
            for window_shape in [[1, 1], [2, 1], [2, 3]]:
              if padding != "SAME":
                for dilation_rate in [[1, 1], [2, 1], [1, 2], [2, 3]]:
                  self._test(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1, 1])
              for strides in [[1, 1], [2, 1], [1, 2], [2, 3]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1, 1],
                    strides=strides)
  def testPool3D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          for input_shape in [[2, 9, 10, 11, 2], [2, 10, 9, 11, 2]]:
            for window_shape in [[1, 1, 1], [2, 1, 2], [2, 3, 2]]:
              if padding != "SAME":
                for dilation_rate in [[1, 1, 1], [2, 1, 2], [1, 2, 2],
                                      [2, 3, 3]]:
                  self._test(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1, 1, 1])
              for strides in [[1, 1, 1], [2, 1, 2], [1, 2, 2], [2, 3, 3]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1, 1, 1],
                    strides=strides)
  def testPoolNC(self):
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        for padding in ["SAME", "VALID"]:
          self._test(
              input_shape=[2, 2, 9],
              window_shape=[2],
              padding=padding,
              pooling_type="MAX",
              strides=[1],
              dilation_rate=[1],
              data_format="NCW")
          self._test(
              input_shape=[2, 2, 9],
              window_shape=[2],
              padding=padding,
              pooling_type="MAX",
              strides=[2],
              dilation_rate=[1],
              data_format="NCW")
          self._test(
              input_shape=[2, 2, 7, 9],
              window_shape=[2, 2],
              padding=padding,
              pooling_type="MAX",
              strides=[1, 2],
              dilation_rate=[1, 1],
              data_format="NCHW")
          self._test(
              input_shape=[2, 2, 7, 5, 3],
              window_shape=[2, 2, 2],
              padding=padding,
              pooling_type="MAX",
              strides=[1, 2, 1],
              dilation_rate=[1, 1, 1],
              data_format="NCDHW")
        self._test(
            input_shape=[2, 2, 7, 9],
            window_shape=[2, 2],
            padding="VALID",
            pooling_type="MAX",
            strides=[1, 1],
            dilation_rate=[2, 2],
            data_format="NCHW")
  def _test_gradient(self, input_shape, **kwargs):
    x_val = -np.arange(
        np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1
    x = constant_op.constant(x_val, name="x", dtype=dtypes.float32)
    output = nn_ops.pool(input=x, **kwargs)
    y_shape = output.get_shape().as_list()
    err = gradient_checker.compute_gradient_error([x], [input_shape],
                                                  output,
                                                  y_shape,
                                                  x_init_value=[x_val])
    err_tolerance = 1e-2
    self.assertLess(err, err_tolerance)
  @test_util.run_deprecated_v1
  def testGradient1D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["AVG", "MAX"]:
          for input_shape in [[2, 5, 2], [1, 4, 1]]:
            for window_shape in [[1], [2]]:
              if padding != "SAME":
                for dilation_rate in [[1], [2]]:
                  self._test_gradient(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1])
              for strides in [[1], [2]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1],
                    strides=strides)
  @test_util.run_deprecated_v1
  def testGradient2D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["AVG", "MAX"]:
          for input_shape in [[2, 4, 5, 2], [1, 5, 4, 1]]:
            for window_shape in [[1, 1], [2, 1], [2, 2]]:
              if padding != "SAME":
                for dilation_rate in [[1, 1], [2, 1], [2, 2]]:
                  self._test_gradient(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1, 1])
              for strides in [[1, 1], [2, 1], [1, 2], [2, 2]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1, 1],
                    strides=strides)
  @test_util.run_deprecated_v1
  def testGradient3D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["AVG", "MAX"]:
          for input_shape in [[1, 3, 5, 4, 1], [1, 5, 4, 3, 1]]:
            for window_shape in [[1, 1, 1], [2, 1, 2], [2, 2, 2]]:
              if padding != "SAME":
                for dilation_rate in [[1, 1, 1], [2, 1, 2], [2, 2, 2]]:
                  self._test_gradient(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1, 1, 1])
              for strides in [[1, 1, 1], [2, 1, 2], [2, 2, 2]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1, 1, 1],
                    strides=strides)
if __name__ == "__main__":
  test.main()
