
import copy
import random
import re
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.python.platform import gfile
_TFLITE_FILE_IDENTIFIER = b'TFL3'
def convert_bytearray_to_object(model_bytearray):
  model_object = schema_fb.Model.GetRootAsModel(model_bytearray, 0)
  return schema_fb.ModelT.InitFromObj(model_object)
def read_model(input_tflite_file):
  if not gfile.Exists(input_tflite_file):
    raise RuntimeError('Input file not found at %r\n' % input_tflite_file)
  with gfile.GFile(input_tflite_file, 'rb') as input_file_handle:
    model_bytearray = bytearray(input_file_handle.read())
  return convert_bytearray_to_object(model_bytearray)
def read_model_with_mutable_tensors(input_tflite_file):
  """Reads a tflite model as a python object with mutable tensors.
  Similar to read_model() with the addition that the returned object has
  mutable tensors (read_model() returns an object with immutable tensors).
  Args:
    input_tflite_file: Full path name to the input tflite file
  Raises:
    RuntimeError: If input_tflite_file path is invalid.
    IOError: If input_tflite_file cannot be opened.
  Returns:
    A mutable python object corresponding to the input tflite file.
  """
  return copy.deepcopy(read_model(input_tflite_file))
def convert_object_to_bytearray(model_object):
  builder = flatbuffers.Builder(1024)
  model_offset = model_object.Pack(builder)
  builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
  model_bytearray = bytes(builder.Output())
  return model_bytearray
def write_model(model_object, output_tflite_file):
  model_bytearray = convert_object_to_bytearray(model_object)
  with gfile.GFile(output_tflite_file, 'wb') as output_file_handle:
    output_file_handle.write(model_bytearray)
def strip_strings(model):
  """Strips all nonessential strings from the model to reduce model size.
  We remove the following strings:
  (find strings by searching ":string" in the tensorflow lite flatbuffer schema)
  1. Model description
  2. SubGraph name
  3. Tensor names
  We retain OperatorCode custom_code and Metadata name.
  Args:
    model: The model from which to remove nonessential strings.
  """
  model.description = None
  for subgraph in model.subgraphs:
    subgraph.name = None
    for tensor in subgraph.tensors:
      tensor.name = None
  model.signatureDefs = None
def randomize_weights(model, random_seed=0, buffers_to_skip=None):
  """Randomize weights in a model.
  Args:
    model: The model in which to randomize weights.
    random_seed: The input to the random number generator (default value is 0).
    buffers_to_skip: The list of buffer indices to skip. The weights in these
                     buffers are left unmodified.
  """
  random.seed(random_seed)
  buffers = model.buffers
  if buffers_to_skip is not None:
    buffer_ids = [idx for idx in buffer_ids if idx not in buffers_to_skip]
  for i in buffer_ids:
    buffer_i_data = buffers[i].data
    buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
    for j in range(buffer_i_size):
      buffer_i_data[j] = random.randint(0, 255)
def rename_custom_ops(model, map_custom_op_renames):
  for op_code in model.operatorCodes:
    if op_code.customCode:
      op_code_str = op_code.customCode.decode('ascii')
      if op_code_str in map_custom_op_renames:
        op_code.customCode = map_custom_op_renames[op_code_str].encode('ascii')
def xxd_output_to_bytes(input_cc_file):
  """Converts xxd output C++ source file to bytes (immutable).
  Args:
    input_cc_file: Full path name to th C++ source file dumped by xxd
  Raises:
    RuntimeError: If input_cc_file path is invalid.
    IOError: If input_cc_file cannot be opened.
  Returns:
    A bytearray corresponding to the input cc file array.
  """
  pattern = re.compile(r'\W*(0x[0-9a-fA-F,x ]+).*')
  model_bytearray = bytearray()
  with open(input_cc_file) as file_handle:
    for line in file_handle:
      values_match = pattern.match(line)
      if values_match is None:
        continue
      list_text = values_match.group(1)
      values_text = filter(None, list_text.split(','))
      values = [int(x, base=16) for x in values_text]
      model_bytearray.extend(values)
  return bytes(model_bytearray)
def xxd_output_to_object(input_cc_file):
  model_bytes = xxd_output_to_bytes(input_cc_file)
  return convert_bytearray_to_object(model_bytes)
