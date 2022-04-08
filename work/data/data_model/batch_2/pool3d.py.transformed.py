
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
def make_pool3d_tests(pool_op):
  """Make a set of tests to do pooling.
  Args:
    pool_op: TensorFlow pooling operation to test  i.e. `tf.nn.max_pool3d`.
  Returns:
    A function representing the true generator (after curried pool_op).
  """
  def f(options, expected_tf_failures=0):
    test_parameters = [
        {
            "ksize": [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 3, 4, 1]],
            "strides": [[1, 1, 1, 1, 1], [1, 2, 1, 2, 1], [1, 2, 2, 4, 1]],
            "input_shape": [[1, 1, 1, 1, 1], [1, 16, 15, 14, 1],
                            [3, 16, 15, 14, 3]],
            "padding": ["SAME", "VALID"],
            "data_format": ["NDHWC"],
        },
    ]
    def build_graph(parameters):
      input_tensor = tf.compat.v1.placeholder(
          dtype=tf.float32, name="input", shape=parameters["input_shape"])
      out = pool_op(
          input_tensor,
          ksize=parameters["ksize"],
          strides=parameters["strides"],
          data_format=parameters["data_format"],
          padding=parameters["padding"])
      return [input_tensor], [out]
    def build_inputs(parameters, sess, inputs, outputs):
      input_values = create_tensor_data(tf.float32, parameters["input_shape"])
      return [input_values], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_values])))
    extra_convert_options = ExtraConvertOptions()
    extra_convert_options.allow_custom_ops = True
    make_zip_of_tests(
        options,
        test_parameters,
        build_graph,
        build_inputs,
        extra_convert_options,
        expected_tf_failures=expected_tf_failures)
  return f
@register_make_test_function()
def make_avg_pool3d_tests(options):
  make_pool3d_tests(tf.nn.avg_pool3d)(options, expected_tf_failures=6)
@register_make_test_function()
def make_max_pool3d_tests(options):
  make_pool3d_tests(tf.nn.max_pool3d)(options, expected_tf_failures=6)
