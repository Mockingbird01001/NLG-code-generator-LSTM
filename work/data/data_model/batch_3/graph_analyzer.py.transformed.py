
import argparse
import sys
from absl import app
from tensorflow.python.grappler import _pywrap_graph_analyzer as tf_wrap
def main(_):
  tf_wrap.GraphAnalyzer(FLAGS.input, FLAGS.n)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input",
      type=str,
      default=None,
      help="Input file path for a TensorFlow MetaGraphDef.")
  parser.add_argument(
      "--n", type=int, default=None, help="The size of the subgraphs.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
