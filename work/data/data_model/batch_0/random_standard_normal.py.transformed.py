
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_random_standard_normal_tests(options):
  test_parameters = [{
      "input_shape": [[1]],
      "input_dtype": [tf.int32],
      "shape": [[10]],
      "seed": [None, 0, 1234],
      "seed2": [0, 5678],
      "dtype": [tf.float32],
  }, {
      "input_shape": [[3]],
      "input_dtype": [tf.int32],
      "shape": [[2, 3, 4]],
      "seed": [0, 1234],
      "seed2": [None, 0, 5678],
      "dtype": [tf.float32],
  }]
  def build_graph(parameters):
    tf.set_random_seed(seed=parameters["seed"])
    input_value = tf.compat.v1.placeholder(
        name="shape",
        shape=parameters["input_shape"],
        dtype=parameters["input_dtype"])
    out = tf.random.normal(
        shape=input_value, dtype=parameters["dtype"], seed=parameters["seed2"])
    return [input_value], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(
        parameters["input_dtype"],
        parameters["input_shape"],
        min_value=1,
        max_value=10)
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
