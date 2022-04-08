
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_unpack_tests(options):
  test_parameters = [{
      "base_shape": [[3, 4, 3], [3, 4], [5, 6, 7, 8]],
      "axis": [0, 1, 2, 3],
      "dtype": [tf.int32, tf.bool, tf.float32],
  }]
  def get_valid_axis(parameters):
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    while axis > len(shape) - 1:
      axis -= 1
    return axis
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name=("input"),
        shape=parameters["base_shape"])
    outs = tf.unstack(input_tensor, axis=get_valid_axis(parameters))
    return [input_tensor], [outs[0]]
  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(
        parameters["dtype"], shape=parameters["base_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
