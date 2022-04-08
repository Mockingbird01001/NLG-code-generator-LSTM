
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools.optimize.python import _pywrap_modify_model_interface
from tensorflow.lite.tools.optimize.python import modify_model_interface_constants as mmi_constants
def _parse_type_to_int(dtype, flag):
  if dtype not in mmi_constants.TFLITE_TYPES:
    raise ValueError(
        "Unsupported value '{0}' for {1}. Only {2} are supported.".format(
            dtype, flag, mmi_constants.TFLITE_TYPES))
  dtype_str = mmi_constants.TFLITE_TO_STR_TYPES[dtype]
  dtype_int = schema_fb.TensorType.__dict__[dtype_str]
  return dtype_int
def modify_model_interface(input_file, output_file, input_type, output_type):
  """Modify a quantized model's interface (input/output) from float to integer.
  Args:
    input_file: Full path name to the input tflite file.
    output_file: Full path name to the output tflite file.
    input_type: Final input interface type.
    output_type: Final output interface type.
  Raises:
    RuntimeError: If the modification of the model interface was unsuccessful.
    ValueError: If the input_type or output_type is unsupported.
  """
  input_type_int = _parse_type_to_int(input_type, 'input_type')
  output_type_int = _parse_type_to_int(output_type, 'output_type')
  status = _pywrap_modify_model_interface.modify_model_interface(
      input_file, output_file, input_type_int, output_type_int)
  if status != 0:
    raise RuntimeError(
        'Error occurred when trying to modify the model input type from float '
        'to {input_type} and output type from float to {output_type}.'.format(
            input_type=input_type, output_type=output_type))
