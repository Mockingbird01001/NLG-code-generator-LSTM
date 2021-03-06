
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_global_batch_norm_tests(options):
  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[1, 1, 6, 2], [3, 4, 5, 4]],
      "epsilon": [0.1, 0.0001],
      "scale_after": [True, False],
  }]
  def build_graph(parameters):
    input_shape = parameters["input_shape"]
    scale_shape = input_shape[3]
    scale = create_tensor_data(parameters["dtype"], scale_shape)
    offset = create_tensor_data(parameters["dtype"], scale_shape)
    mean = create_tensor_data(parameters["dtype"], scale_shape)
    variance = create_tensor_data(parameters["dtype"], scale_shape)
    x = create_tensor_data(parameters["dtype"], parameters["input_shape"])
    x_norm = tf.nn.batch_norm_with_global_normalization(
        x, mean, variance, scale, offset, parameters["epsilon"],
        parameters["scale_after"])
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.add(input_tensor, x_norm)
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
