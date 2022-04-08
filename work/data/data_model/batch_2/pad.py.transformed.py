
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_pad_tests(options):
  test_parameters = [
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 1, 2, 1, 1], [2, 1, 1, 1, 1]],
          "paddings": [[[0, 0], [0, 1], [2, 3], [0, 0], [1, 0]],
                       [[0, 1], [0, 0], [0, 0], [2, 3], [1, 0]]],
          "constant_paddings": [True, False],
          "fully_quantize": [False],
          "quant_16x8": [False]
      },
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 1, 2, 1], [2, 1, 1, 1]],
          "paddings": [[[0, 0], [0, 1], [2, 3], [0, 0]],
                       [[0, 1], [0, 0], [0, 0], [2, 3]]],
          "constant_paddings": [True, False],
          "fully_quantize": [False],
          "quant_16x8": [False]
      },
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 2]],
          "paddings": [[[0, 1], [2, 3]]],
          "constant_paddings": [True, False],
          "fully_quantize": [False],
          "quant_16x8": [False]
      },
      {
          "dtype": [tf.int32],
          "input_shape": [[1]],
          "paddings": [[[1, 2]]],
          "constant_paddings": [False],
          "fully_quantize": [False],
          "quant_16x8": [False]
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 1, 2, 1], [2, 1, 1, 1]],
          "paddings": [[[0, 0], [0, 1], [2, 3], [0, 0]],
                       [[0, 1], [0, 0], [0, 0], [2, 3]],
                       [[0, 0], [0, 0], [0, 0], [0, 0]]],
          "constant_paddings": [True],
          "fully_quantize": [True],
          "quant_16x8": [False, True]
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 2]],
          "paddings": [[[0, 1], [2, 3]]],
          "constant_paddings": [True],
          "fully_quantize": [True],
          "quant_16x8": [False, True],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[1]],
          "paddings": [[[1, 2]]],
          "constant_paddings": [True],
          "fully_quantize": [True],
          "quant_16x8": [False, True],
      },
  ]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    if parameters["constant_paddings"]:
      paddings = parameters["paddings"]
      input_tensors = [input_tensor]
    else:
      shape = [len(parameters["paddings"]), 2]
      paddings = tf.compat.v1.placeholder(
          dtype=tf.int32, name="padding", shape=shape)
      input_tensors = [input_tensor, paddings]
    out = tf.pad(input_tensor, paddings=paddings)
    return input_tensors, [out]
  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            parameters["dtype"],
            parameters["input_shape"],
            min_value=-1,
            max_value=1)
    ]
    if not parameters["constant_paddings"]:
      values.append(np.array(parameters["paddings"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
