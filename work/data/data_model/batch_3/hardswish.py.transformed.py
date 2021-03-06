
import functools
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
def _tflite_convert_verify_num_ops(tflite_convert_function, *args, **kwargs):
  num_ops = kwargs.pop("num_ops", 2)
  result = tflite_convert_function(*args, **kwargs)
  tflite_model_binary = result[0]
  if not result[0]:
    raise RuntimeError("Failed to build model: \n\n" + result[1])
  interpreter = tf.lite.Interpreter(model_content=tflite_model_binary)
  interpreter.allocate_tensors()
  if len(interpreter.get_tensor_details()) != num_ops:
    raise RuntimeError(
        "Expected to generate two node graph got %s " %
        "\n".join(str(x) for x in interpreter.get_tensor_details()))
  return result
@register_make_test_function()
def make_hardswish_tests(options):
  if options.run_with_flex:
    test_parameters = [{
        "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                        [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
    }]
  else:
    test_parameters = [{
        "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                        [3, 15, 14, 3]],
    }]
  def build_graph(parameters):
    inp = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = inp * tf.nn.relu6(inp + np.float32(3)) * np.float32(1. / 6.)
    return [inp], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-10, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))
  if not options.run_with_flex:
    options.tflite_convert_function = functools.partial(
        _tflite_convert_verify_num_ops,
        options.tflite_convert_function,
        num_ops=2)
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
