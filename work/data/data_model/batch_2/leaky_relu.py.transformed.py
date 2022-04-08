
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_leaky_relu_tests(options):
  test_parameters = [{
      "input_shape": [[], [1], [5], [1, 10, 10, 3], [3, 3, 3, 3]],
      "alpha": [0.1, 1.0, 2.0, -0.1, -1.0, -2.0],
      "fully_quantize": [False, True],
      "input_range": [(-3, 10)],
      "quant_16x8": [False, True],
  }]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.nn.leaky_relu(input_tensor, alpha=parameters["alpha"])
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-3, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)