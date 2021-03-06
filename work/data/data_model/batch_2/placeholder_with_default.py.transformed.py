
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import MAP_TF_TO_NUMPY_TYPE
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_placeholder_with_default_tests(options):
  test_parameters = [{
      "dtype": [tf.float32, tf.int32, tf.int64],
  }]
  def build_graph(parameters):
    const_node = tf.constant([1, 2, 2, 0],
                             shape=[2, 2],
                             dtype=parameters["dtype"])
    input_tensor = tf.compat.v1.placeholder_with_default(
        const_node, shape=[2, 2], name="input")
    out = tf.equal(input_tensor, const_node, name="output")
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    numpy_type = MAP_TF_TO_NUMPY_TYPE[parameters["dtype"]]
    input_value = np.array([[1, 0], [2, 1]], numpy_type)
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
