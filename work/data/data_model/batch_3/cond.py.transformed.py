
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util
@register_make_test_function("make_cond_tests")
@test_util.enable_control_flow_v2
def make_cond_tests(options):
  test_parameters = [{
      "dtype": [tf.float32, tf.string],
      "pred": [False, True],
  }]
  def build_graph(parameters):
    input1 = tf.placeholder(dtype=parameters["dtype"], shape=(1,))
    input2 = tf.placeholder(dtype=parameters["dtype"], shape=(1,))
    pred = tf.placeholder(dtype=tf.bool, shape=(1,))
    pred_scalar = tf.reshape(pred, ())
    out = tf.cond(pred_scalar, lambda: input1, lambda: input2)
    return [input1, input2, pred], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_values = [
        create_tensor_data(parameters["dtype"], (1,)),
        create_tensor_data(parameters["dtype"], (1,)),
        np.array([parameters["pred"]], dtype=np.bool_),
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
