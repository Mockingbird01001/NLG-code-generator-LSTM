
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_zeros_like_tests(options):
  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
  }]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    zeros = tf.zeros_like(input_tensor)
    out = tf.maximum(zeros, input_tensor)
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    values = create_tensor_data(parameters["input_dtype"],
                                parameters["input_shape"])
    return [values], sess.run(outputs, feed_dict=dict(zip(inputs, [values])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
