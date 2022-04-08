
import argparse
import sys
from absl import app
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import strip_unused_lib
FLAGS = None
def main(unused_args):
  strip_unused_lib.strip_unused_from_files(FLAGS.input_graph,
                                           FLAGS.input_binary,
                                           FLAGS.output_graph,
                                           FLAGS.output_binary,
                                           FLAGS.input_node_names,
                                           FLAGS.output_node_names,
                                           FLAGS.placeholder_type_enum)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--input_graph',
      type=str,
      default='',
      help='TensorFlow \'GraphDef\' file to load.')
  parser.add_argument(
      '--input_binary',
      nargs='?',
      const=True,
      type='bool',
      default=False,
      help='Whether the input files are in binary format.')
  parser.add_argument(
      '--output_graph',
      type=str,
      default='',
      help='Output \'GraphDef\' file name.')
  parser.add_argument(
      '--output_binary',
      nargs='?',
      const=True,
      type='bool',
      default=True,
      help='Whether to write a binary format graph.')
  parser.add_argument(
      '--input_node_names',
      type=str,
      default='',
      help='The name of the input nodes, comma separated.')
  parser.add_argument(
      '--output_node_names',
      type=str,
      default='',
      help='The name of the output nodes, comma separated.')
  parser.add_argument(
      '--placeholder_type_enum',
      type=int,
      default=dtypes.float32.as_datatype_enum,
      help='The AttrValue enum to use for placeholders.')
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
