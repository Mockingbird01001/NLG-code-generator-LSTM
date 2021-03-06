
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_depthwiseconv_tests(options):
  test_parameters = [
      {
          "input_shape": [[1, 3, 4, 3], [1, 10, 10, 3]],
          "filter_size": [[1, 1], [1, 2], [3, 3]],
          "strides": [[1, 1, 1, 1], [1, 3, 3, 1]],
          "dilations": [[1, 1, 1, 1], [1, 3, 2, 1], [1, 2, 2, 1]],
          "channel_multiplier": [1, 2],
          "rate": [[1, 1]],
          "padding": ["SAME", "VALID"],
          "data_format": ["NHWC"],
          "constant_filter": [True, False],
          "fully_quantize": [False],
          "quant_16x8": [False]
      },
      {
          "input_shape": [[1, 3, 4, 3]],
          "filter_size": [[1, 1]],
          "dilations": [[1, 1, 1, 1], [1, 2, 2, 1]],
          "channel_multiplier": [2],
          "padding": ["SAME"],
          "data_format": ["NHWC"],
          "constant_filter": [True, False],
          "fully_quantize": [False],
          "quant_16x8": [False]
      },
      {
          "input_shape": [[1, 3, 4, 3], [1, 10, 10, 3]],
          "filter_size": [[1, 1], [1, 2], [3, 3]],
          "strides": [[1, 1, 1, 1], [1, 3, 3, 1]],
          "dilations": [[1, 1, 1, 1], [1, 3, 2, 1], [1, 2, 2, 1]],
          "channel_multiplier": [1, 2],
          "rate": [[1, 1]],
          "padding": ["SAME", "VALID"],
          "data_format": ["NHWC"],
          "constant_filter": [True],
          "fully_quantize": [True],
          "quant_16x8": [False]
      },
      {
          "input_shape": [[1, 3, 3, 3000]],
          "filter_size": [[3, 3]],
          "strides": [[1, 1, 1, 1]],
          "dilations": [[1, 1, 1, 1]],
          "channel_multiplier": [1],
          "rate": [[1, 1]],
          "padding": ["VALID"],
          "data_format": ["NHWC"],
          "constant_filter": [True],
          "fully_quantize": [True],
          "quant_16x8": [False]
      },
      {
          "input_shape": [[1, 3, 4, 3]],
          "filter_size": [[1, 2]],
          "strides": [[1, 3, 3, 1]],
          "dilations": [[1, 3, 2, 1]],
          "channel_multiplier": [1],
          "rate": [[1, 1]],
          "padding": ["SAME", "VALID"],
          "data_format": ["NHWC"],
          "constant_filter": [True],
          "fully_quantize": [True],
          "quant_16x8": [True]
      },
  ]
  def get_tensor_shapes(parameters):
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_size"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]
  def build_graph(parameters):
    input_shape, filter_shape = get_tensor_shapes(parameters)
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)
    if parameters["constant_filter"]:
      filter_input = create_tensor_data(np.float32, filter_shape)
      input_tensors = [input_tensor]
    else:
      filter_input = tf.compat.v1.placeholder(
          dtype=tf.float32, name="filter", shape=filter_shape)
      input_tensors = [input_tensor, filter_input]
    out = tf.nn.depthwise_conv2d(
        input_tensor,
        filter_input,
        strides=parameters["strides"],
        rate=parameters["rate"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    return input_tensors, [out]
  def build_inputs(parameters, sess, inputs, outputs):
    """Build list of input values.
    It either contains 1 tensor (input) or 2 tensors (input, filter) based on
    whether filter is constant or variable input.
    """
    input_shape, filter_shape = get_tensor_shapes(parameters)
    values = [
        create_tensor_data(np.float32, input_shape, min_value=-1, max_value=1)
    ]
    if not parameters["constant_filter"]:
      values.append(
          create_tensor_data(
              np.float32, filter_shape, min_value=-1, max_value=1))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=4)
