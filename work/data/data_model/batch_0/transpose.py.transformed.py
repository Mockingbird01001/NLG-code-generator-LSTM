
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_transpose_tests(options):
  test_parameters = [{
      "dtype": [tf.int32, tf.int64, tf.float32],
      "input_shape": [[2, 2, 3]],
      "perm": [[0, 1, 2], [0, 2, 1]],
      "constant_perm": [True, False],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4]],
      "perm": [[0, 1, 2, 3], [3, 0, 1, 2]],
      "constant_perm": [True, False],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4, 5]],
      "perm": [[4, 3, 2, 1, 0]],
      "constant_perm": [True, False],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[2, 2, 3]],
      "perm": [[0, 1, 2], [0, 2, 1]],
      "constant_perm": [True],
      "fully_quantize": [True],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4]],
      "perm": [[0, 1, 2, 3], [3, 0, 1, 2]],
      "constant_perm": [True],
      "fully_quantize": [True],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4, 5]],
      "perm": [[0, 1, 2, 3, 4], [3, 4, 0, 1, 2]],
      "constant_perm": [True],
      "fully_quantize": [True, False],
  }]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    if parameters["constant_perm"]:
      perm = parameters["perm"]
      input_tensors = [input_tensor]
    else:
      shape = [len(parameters["perm"]), 2]
      perm = tf.compat.v1.placeholder(dtype=tf.int32, name="perm", shape=shape)
      input_tensors = [input_tensor, perm]
    out = tf.transpose(input_tensor, perm=perm)
    return input_tensors, [out]
  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"],
                           min_value=-1, max_value=1)
    ]
    if not parameters["constant_perm"]:
      values.append(np.array(parameters["perm"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=9)