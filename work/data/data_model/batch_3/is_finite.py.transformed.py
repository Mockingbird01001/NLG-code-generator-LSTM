
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_is_finite_tests(options):
  test_parameters = [
      {
          "input_shape": [[100], [3, 15, 14, 3]],
      },
  ]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.math.is_finite(input_tensor)
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-10, max_value=10)
    def random_index(shape):
      result = []
      for dim in shape:
        result.append(np.random.randint(low=0, high=dim))
      return tuple(result)
    input_values[random_index(input_values.shape)] = np.Inf
    input_values[random_index(input_values.shape)] = -np.Inf
    input_values[random_index(input_values.shape)] = np.NAN
    input_values[random_index(input_values.shape)] = tf.float32.max
    input_values[random_index(input_values.shape)] = tf.float32.min
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)