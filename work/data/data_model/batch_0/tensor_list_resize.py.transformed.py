
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.ops import list_ops
@register_make_test_function()
def make_tensor_list_resize_tests(options):
  test_parameters = [
      {
          "element_dtype": [tf.float32, tf.int32],
          "num_elements": [4, 5, 6],
          "element_shape": [[], [5], [3, 3]],
          "new_size": [1, 3, 5, 7],
      },
  ]
  def build_graph(parameters):
    data = tf.placeholder(
        dtype=parameters["element_dtype"],
        shape=[parameters["num_elements"]] + parameters["element_shape"])
    tensor_list = list_ops.tensor_list_from_tensor(data,
                                                   parameters["element_shape"])
    tensor_list = list_ops.tensor_list_resize(tensor_list,
                                              parameters["new_size"])
    out = list_ops.tensor_list_stack(
        tensor_list, element_dtype=parameters["element_dtype"])
    return [data], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    data = create_tensor_data(parameters["element_dtype"],
                              [parameters["num_elements"]] +
                              parameters["element_shape"])
    return [data], sess.run(outputs, feed_dict=dict(zip(inputs, [data])))
  extra_convert_options = ExtraConvertOptions()
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_convert_options)
