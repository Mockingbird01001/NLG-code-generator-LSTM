
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing import zip_test_utils
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util
@register_make_test_function("make_while_tests")
@test_util.enable_control_flow_v2
def make_while_tests(options):
  test_parameters = [{
      "num_iterations": range(20),
      "increment_value": [[1]],
      "dtype": [tf.int32],
  }, {
      "num_iterations": range(20),
      "increment_value": [["a"]],
      "dtype": [tf.string],
  }]
  def build_graph(parameters):
    num_iterations = tf.placeholder(
        dtype=tf.int32, name="num_iterations", shape=(1,))
    increment_value = tf.placeholder(
        dtype=parameters["dtype"], name="increment_value", shape=(1,))
    num_iterations_scalar = tf.reshape(num_iterations, ())
    def cond_fn(counter, value, increment_value):
      del value
      del increment_value
      return counter < num_iterations_scalar
    def body_fn(counter, value, increment_value):
      new_counter = counter + 1
      if parameters["dtype"] == tf.string:
        del value
        new_value = tf.fill([1], tf.reshape(increment_value, ()))
      else:
        new_value = value + increment_value
      return [new_counter, new_value, increment_value]
    counter, value, result_increment_value = tf.while_loop(
        cond_fn, body_fn, loop_vars=[1, increment_value, increment_value])
    return [num_iterations,
            increment_value], [counter, value, result_increment_value]
  def build_inputs(parameters, sess, inputs, outputs):
    numpy_type = zip_test_utils.MAP_TF_TO_NUMPY_TYPE[parameters["dtype"]]
    input_values = [
        np.array([parameters["num_iterations"]], dtype=np.int32),
        np.array(parameters["increment_value"], dtype=numpy_type)
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
