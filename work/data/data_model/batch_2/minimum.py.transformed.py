
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_minimum_tests(options):
  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape_1": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3],
                        [5, 32, 32, 1, 1], [5, 32, 32, 1, 1]],
      "input_shape_2": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3],
                        [5, 32, 32, 1, 1], [5, 32, 32, 1, 3]],
      "fully_quantize": [False, True],
  }]
  def build_graph(parameters):
    input_tensor_1 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input_1",
        shape=parameters["input_shape_1"])
    input_tensor_2 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input_2",
        shape=parameters["input_shape_2"])
    out = tf.minimum(input_tensor_1, input_tensor_2)
    return [input_tensor_1, input_tensor_2], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            parameters["input_dtype"],
            parameters["input_shape_1"],
            min_value=-1,
            max_value=1),
        create_tensor_data(
            parameters["input_dtype"],
            parameters["input_shape_2"],
            min_value=-1,
            max_value=1)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=44)
