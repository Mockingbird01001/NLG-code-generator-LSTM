
import argparse
import sys
import tensorflow as tf
FLAGS = None
def load_graph(filename):
  with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
def load_labels(filename):
  return [line.rstrip() for line in tf.io.gfile.GFile(filename)]
def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.compat.v1.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
    return 0
def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  if not wav or not tf.io.gfile.exists(wav):
    raise ValueError('Audio file does not exist at {0}'.format(wav))
  if not labels or not tf.io.gfile.exists(labels):
    raise ValueError('Labels file does not exist at {0}'.format(labels))
  if not graph or not tf.io.gfile.exists(graph):
    raise ValueError('Graph file does not exist at {0}'.format(graph))
  labels_list = load_labels(labels)
  load_graph(graph)
  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()
  run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
def main(_):
  label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
