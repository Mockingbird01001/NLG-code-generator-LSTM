
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_nearest_upsample_tests(options):
  test_parameters = [{
      "input_shape": [[1, 10, 10, 64], [3, 8, 32]],
      "scale_n_axis": [([2, 2], [1, 2]), ([3, 4], [1, 2]), ([3], [1])],
      "dtype": [tf.float32, tf.int32],
  }]
  def new_shape_for_upsample(original_shape, scales, axis):
    input_new_shape = []
    ones_new_shape = []
    upsample_new_shape = []
    j = 0
    for i in range(len(original_shape)):
      input_new_shape.append(original_shape[i])
      ones_new_shape.append(1)
      if j < len(scales) and axis[j] == i:
        input_new_shape.append(1)
        ones_new_shape.append(scales[j])
        upsample_new_shape.append(original_shape[i] * scales[j])
        j += 1
      else:
        upsample_new_shape.append(original_shape[i])
    return input_new_shape, ones_new_shape, upsample_new_shape
  def build_graph(parameters):
    input_shape = parameters["input_shape"]
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="input", shape=input_shape)
    scales, axis = parameters["scale_n_axis"]
    input_new_shape, ones_new_shape, new_shape = new_shape_for_upsample(
        input_shape, scales, axis)
    out = tf.compat.v1.reshape(input_tensor,
                               input_new_shape) * tf.compat.v1.ones(
                                   ones_new_shape, dtype=parameters["dtype"])
    out = tf.compat.v1.reshape(out, new_shape)
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        parameters["dtype"],
        parameters["input_shape"],
        min_value=-10,
        max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
