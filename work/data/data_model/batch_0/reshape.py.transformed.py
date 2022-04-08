
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_reshape_tests(options):
  test_parameters = [
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape": [[3, 4, 5, 7], [4, 105], [21, 5, 2, 2], [420]],
          "output_shape": [[15, 28], [420], [1, -1, 5, 7], [-1]],
          "constant_shape": [True, False],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[1]],
          "output_shape": [[]],
          "constant_shape": [True, False],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[3, 4, 5, 7], [4, 105], [21, 5, 2, 2], [420]],
          "output_shape": [[15, 28], [420], [1, -1, 5, 7], [-1]],
          "constant_shape": [True],
          "fully_quantize": [True],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 4, 0]],
          "output_shape": [[2, -1], [2, 0, -1]],
          "constant_shape": [True, False],
          "fully_quantize": [False],
      }
  ]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    if parameters["constant_shape"]:
      output_shape = parameters["output_shape"]
      input_tensors = [input_tensor]
    else:
      shape_tensor_shape = [len(parameters["output_shape"])]
      output_shape = tf.compat.v1.placeholder(
          dtype=tf.int32, name="output_shape", shape=shape_tensor_shape)
      input_tensors = [input_tensor, output_shape]
    out = tf.reshape(input_tensor, shape=output_shape)
    return input_tensors, [out]
  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            parameters["dtype"],
            parameters["input_shape"],
            min_value=-1,
            max_value=1)
    ]
    if not parameters["constant_shape"]:
      values.append(np.array(parameters["output_shape"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
