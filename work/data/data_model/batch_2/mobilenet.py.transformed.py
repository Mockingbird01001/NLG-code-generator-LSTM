
import argparse
import sys
import tensorflow.compat.v1 as tf
from tensorflow.lite.experimental.acceleration.mini_benchmark.metrics import kl_divergence
parser = argparse.ArgumentParser(
    description='Script to generate a metrics model for mobilenet v1.')
parser.add_argument('output', help='Output filepath')
def main(output_path):
  tf.reset_default_graph()
  with tf.Graph().as_default():
    expected_scores = tf.placeholder(dtype=tf.float32, shape=[1, 1001])
    actual_scores = tf.placeholder(dtype=tf.float32, shape=[1, 1001])
    mse = tf.reshape(
        tf.math.reduce_mean((expected_scores - actual_scores)**2), [1],
        name='mse')
    kld_metric = kl_divergence.symmetric_kl_divergence(expected_scores,
                                                       actual_scores)
    kld_metric = tf.reshape(kld_metric, [1], name='symmetric_kl_divergence')
    ok = tf.reshape(
        tf.logical_and(kld_metric < 5.5, mse < 0.003), [1], name='ok')
    sess = tf.compat.v1.Session()
    converter = tf.lite.TFLiteConverter.from_session(sess, [
        expected_scores,
        actual_scores,
    ], [kld_metric, mse, ok])
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open(output_path, 'wb').write(tflite_model)
if __name__ == '__main__':
  flags, unparsed = parser.parse_known_args()
  if unparsed:
    parser.print_usage()
    sys.stderr.write('\nGot the following unparsed args, %r please fix.\n' %
                     unparsed)
    exit(1)
  else:
    main(flags.output)
    exit(0)
