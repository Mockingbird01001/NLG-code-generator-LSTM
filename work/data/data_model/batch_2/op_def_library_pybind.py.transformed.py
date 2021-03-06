
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import _op_def_library_pybind
from tensorflow.core.framework import attr_value_pb2
def process_inputs(op_name, producer_version, keywords):
  attr_protos, inputs, input_types, output_structure = (
      _op_def_library_pybind.process_inputs(op_name, producer_version,
                                            keywords))
  for k, attr in attr_protos.items():
    attr_protos[k] = attr_value_pb2.AttrValue.FromString(attr)
  return attr_protos, inputs, input_types, output_structure
