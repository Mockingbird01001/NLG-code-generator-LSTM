
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_cumsum_tests(options):
  test_parameters = [{
      "shape": [(3, 6), (8, 9, 7)],
      "dtype": [tf.int32, tf.int64, tf.float32],
      "axis": [0, 1],
      "exclusive": [True, False],
      "reverse": [True, False],
  }]
  def build_graph(parameters):
    input1 = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], shape=parameters["shape"])
    out = tf.math.cumsum(
        input1,
        parameters["axis"],
        exclusive=parameters["exclusive"],
        reverse=parameters["reverse"])
    return [input1], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input1 = create_tensor_data(parameters["dtype"], parameters["shape"])
    return [input1], sess.run(outputs, feed_dict=dict(zip(inputs, [input1])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
