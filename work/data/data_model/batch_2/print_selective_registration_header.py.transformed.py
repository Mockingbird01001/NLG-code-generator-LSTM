
import argparse
import sys
from absl import app
from tensorflow.python.tools import selective_registration_header_lib
FLAGS = None
def main(unused_argv):
  graphs = FLAGS.graphs.split(',')
  print(
      selective_registration_header_lib.get_header(graphs,
                                                   FLAGS.proto_fileformat,
                                                   FLAGS.default_ops))
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--graphs',
      type=str,
      default='',
      help='Comma-separated list of paths to model files to be analyzed.',
      required=True)
  parser.add_argument(
      '--proto_fileformat',
      type=str,
      default='rawproto',
      help='Format of proto file, either textproto, rawproto or ops_list. The '
      'ops_list is the file contains the list of ops in JSON format. Ex: '
      '"[["Add", "BinaryOp<CPUDevice, functor::add<float>>"]]".')
  parser.add_argument(
      '--default_ops',
      type=str,
      default='NoOp:NoOp,_Recv:RecvOp,_Send:SendOp',
      help='Default operator:kernel pairs to always include implementation for.'
      'Pass "all" to have all operators and kernels included; note that this '
      'should be used only when it is useful compared with simply not using '
      'selective registration, as it can in some cases limit the effect of '
      'compilation caches')
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
