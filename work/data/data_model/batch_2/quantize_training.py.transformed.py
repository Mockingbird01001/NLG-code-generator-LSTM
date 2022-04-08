
from tensorflow.core.framework import graph_pb2
from tensorflow.python._pywrap_quantize_training import DoQuantizeTrainingOnGraphDefHelper
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(
    None,
    "GraphDef quantized training rewriter is deprecated in the long term.")
@tf_export(v1=["train.do_quantize_training_on_graphdef"])
def do_quantize_training_on_graphdef(input_graph, num_bits):
  graph = graph_pb2.GraphDef()
  result_graph_string = DoQuantizeTrainingOnGraphDefHelper(
      input_graph.SerializeToString(), num_bits)
  graph.ParseFromString(result_graph_string)
  return graph
