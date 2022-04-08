
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
@register_make_test_function()
def make_stft_tests(options):
  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[8], [8, 16], [3, 1, 4]],
      "frame_length": [4, 8],
      "frame_step": [1, 2, 4],
      "fft_length": [None, 2, 4, 8],
  }]
  def build_graph(parameters):
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    outs = tf.signal.stft(
        input_value,
        frame_length=parameters["frame_length"],
        frame_step=parameters["frame_step"],
        fft_length=parameters["fft_length"])
    return [input_value], [outs]
  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))
  extra_convert_options = ExtraConvertOptions()
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_convert_options)
