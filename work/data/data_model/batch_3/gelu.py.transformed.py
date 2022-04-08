
import functools
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
def _tflite_convert_verify_op(tflite_convert_function, *args, **kwargs):
  result = tflite_convert_function(*args, **kwargs)
  tflite_model_binary = result[0]
  if not result[0]:
    raise RuntimeError("Failed to build model: \n\n" + result[1])
  interpreter = tf.lite.Interpreter(model_content=tflite_model_binary)
  interpreter.allocate_tensors()
    if op["op_name"] == "GELU":
      return result
  raise RuntimeError("Expected to generate GELU op node in graph.")
@register_make_test_function()
def make_gelu_tests(options):
  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                      [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
      "fully_quantize": [False, True],
      "input_range": [(-10, 10)],
      "approximate": [True, False],
  }]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.nn.gelu(input_tensor, approximate=parameters["approximate"])
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            parameters["input_dtype"],
            parameters["input_shape"],
            min_value=-8,
            max_value=8)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))
  if not options.run_with_flex:
    options.tflite_convert_function = functools.partial(
        _tflite_convert_verify_op,
        options.tflite_convert_function)
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
