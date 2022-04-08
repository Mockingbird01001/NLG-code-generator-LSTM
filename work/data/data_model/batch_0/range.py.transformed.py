
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_scalar_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_range_tests(options):
  test_parameters = [{
      "dtype": [tf.int32, tf.float32],
      "offset": [10, 100, 1000, 0],
      "delta": [1, 2, 3, 4, -1, -2, -3, -4],
  }]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name=("start"), shape=[])
    if parameters["delta"] < 0:
      offset = parameters["offset"] * -1
    else:
      offset = parameters["offset"]
    delta = parameters["delta"]
    limit_tensor = input_tensor + offset
    delta_tensor = tf.constant(delta, dtype=parameters["dtype"])
    out = tf.range(input_tensor, limit_tensor, delta_tensor)
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_scalar_data(parameters["dtype"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
