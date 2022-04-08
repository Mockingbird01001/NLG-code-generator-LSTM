
import random
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_arg_min_max_tests(options):
  test_parameters = [
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape": [[], [1, 1, 1, 3], [2, 3, 4, 5], [2, 3, 3], [5, 5],
                          [10]],
          "output_type": [tf.int32, tf.int64],
          "is_arg_max": [True],
          "is_last_axis": [False],
          "dynamic_range_quantize": [False, True],
      },
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                          [2, 10], [3, 4, 50], [2, 3, 5, 100]],
          "output_type": [tf.int32, tf.int64],
          "is_arg_max": [False, True],
          "is_last_axis": [True],
          "dynamic_range_quantize": [False, True],
      },
      {
          "input_dtype": [tf.bool],
          "input_shape": [[1, 1, 1, 3], [2, 3, 4, 5], [2, 3, 3], [5, 5], [10]],
          "output_type": [tf.int32, tf.int64],
          "is_arg_max": [True],
          "is_last_axis": [False],
      },
  ]
  def build_graph(parameters):
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    if not parameters["is_last_axis"]:
      axis = random.randint(0, max(len(parameters["input_shape"]) - 1, 0))
    else:
      axis = -1
    if parameters["is_arg_max"]:
      out = tf.math.argmax(
          input_value, axis, output_type=parameters["output_type"])
    else:
      out = tf.math.argmin(
          input_value, axis, output_type=parameters["output_type"])
    return [input_value], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=8)
