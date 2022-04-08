
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
def make_conv_activation_tests(activation_op):
  def f(options):
    test_parameters = [
        {
            "input_shape": [[1, 3, 4, 3], [4, 6, 6, 1]],
            "filter_shape": [[1, 1], [2, 3], [3, 3]],
            "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
            "dilations": [[1, 1, 1, 1], [1, 3, 2, 1], [1, 2, 2, 1]],
            "padding": ["SAME", "VALID"],
            "constant_filter": [True, False],
            "channel_multiplier": [1, 2],
            "fully_quantize": [False],
            "quant_16x8": [False],
            "dynamic_range_quantize": [False],
        },
        {
            "input_shape": [[1, 3, 4, 3], [4, 6, 6, 1]],
            "filter_shape": [[1, 1], [2, 3]],
            "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
            "dilations": [[1, 1, 1, 1], [1, 3, 2, 1]],
            "padding": ["SAME", "VALID"],
            "constant_filter": [True],
            "channel_multiplier": [1, 2],
            "fully_quantize": [True],
            "quant_16x8": [False, True],
            "dynamic_range_quantize": [False],
        },
        {
            "input_shape": [[1, 3, 4, 3]],
            "filter_shape": [[1, 1], [2, 3], [3, 3]],
            "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
            "dilations": [[1, 1, 1, 1]],
            "padding": ["SAME", "VALID"],
            "data_format": ["NHWC"],
            "constant_filter": [True],
            "channel_multiplier": [1, 2],
            "fully_quantize": [False],
            "quant_16x8": [False],
            "dynamic_range_quantize": [True],
        },
    ]
    def get_tensor_shapes(parameters):
      input_shape = parameters["input_shape"]
      filter_size = parameters["filter_shape"]
      filter_shape = filter_size + [
          input_shape[3], parameters["channel_multiplier"]
      ]
      return [input_shape, filter_shape]
    def build_graph(parameters):
      input_shape, filter_shape = get_tensor_shapes(parameters)
      input_tensor = tf.compat.v1.placeholder(
          dtype=tf.float32, name="input", shape=input_shape)
      if parameters["constant_filter"]:
        filter_input = create_tensor_data(
            np.float32, filter_shape, min_value=-10, max_value=10)
        input_tensors = [input_tensor]
      else:
        filter_input = tf.compat.v1.placeholder(
            dtype=tf.float32, name="filter", shape=filter_shape)
        input_tensors = [input_tensor, filter_input]
      out = tf.nn.conv2d(
          input_tensor,
          filter_input,
          strides=parameters["strides"],
          dilations=parameters["dilations"],
          padding=parameters["padding"],
          data_format=parameters["data_format"])
      out = activation_op(out)
      return input_tensors, [out]
    def build_inputs(parameters, sess, inputs, outputs):
      input_shape, filter_shape = get_tensor_shapes(parameters)
      values = [
          create_tensor_data(
              np.float32, input_shape, min_value=-1, max_value=1)
      ]
      if not parameters["constant_filter"]:
        values.append(create_tensor_data(np.float32, filter_shape))
      return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))
    make_zip_of_tests(
        options,
        test_parameters,
        build_graph,
        build_inputs,
        expected_tf_failures=48)
  return f
@register_make_test_function()
def make_conv_relu6_tests(options):
  return make_conv_activation_tests(tf.nn.relu6)(options)
@register_make_test_function()
def make_conv_relu_tests(options):
  return make_conv_activation_tests(tf.nn.relu)(options)
def relu1(input_tensor):
  out = tf.minimum(1.0, tf.maximum(input_tensor, -1.0))
  return out
@register_make_test_function()
def make_conv_relu1_tests(options):
  return make_conv_activation_tests(relu1)(options)
