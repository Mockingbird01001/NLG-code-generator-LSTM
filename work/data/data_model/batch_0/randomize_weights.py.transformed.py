
r
from absl import app
from absl import flags
from tensorflow.lite.tools import flatbuffer_utils
FLAGS = flags.FLAGS
flags.DEFINE_string('input_tflite_file', None,
                    'Full path name to the input TFLite file.')
flags.DEFINE_string('output_tflite_file', None,
                    'Full path name to the output randomized TFLite file.')
flags.DEFINE_multi_integer(
    'buffers_to_skip', [], 'Buffer indices in the TFLite model to be skipped, '
    'i.e., to be left unmodified.')
flags.DEFINE_integer('random_seed', 0, 'Input to the random number generator.')
flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_tflite_file')
def main(_):
  model = flatbuffer_utils.read_model(FLAGS.input_tflite_file)
  flatbuffer_utils.randomize_weights(model, FLAGS.random_seed,
                                     FLAGS.buffers_to_skip)
  flatbuffer_utils.write_model(model, FLAGS.output_tflite_file)
if __name__ == '__main__':
  app.run(main)
