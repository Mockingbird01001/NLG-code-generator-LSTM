
from tensorflow.python import _pywrap_toco_api
def wrapped_toco_convert(model_flags_str, toco_flags_str, input_data_str,
                         debug_info_str, enable_mlir_converter):
  return _pywrap_toco_api.TocoConvert(
      model_flags_str,
      toco_flags_str,
      input_data_str,
      debug_info_str,
      enable_mlir_converter)
def wrapped_experimental_mlir_quantize(input_data_str, disable_per_channel,
                                       fully_quantize, inference_type,
                                       input_data_type, output_data_type,
                                       enable_numeric_verify,
                                       enable_whole_model_verify,
                                       denylisted_ops, denylisted_nodes):
  return _pywrap_toco_api.ExperimentalMlirQuantizeModel(
      input_data_str, disable_per_channel, fully_quantize, inference_type,
      input_data_type, output_data_type, enable_numeric_verify,
      enable_whole_model_verify, denylisted_ops, denylisted_nodes)
def wrapped_experimental_mlir_sparsify(input_data_str):
  return _pywrap_toco_api.ExperimentalMlirSparsifyModel(input_data_str)
def wrapped_register_custom_opdefs(custom_opdefs_list):
  return _pywrap_toco_api.RegisterCustomOpdefs(custom_opdefs_list)
def wrapped_retrieve_collected_errors():
  return _pywrap_toco_api.RetrieveCollectedErrors()
def wrapped_flat_buffer_file_to_mlir(model, input_is_filepath):
  return _pywrap_toco_api.FlatBufferToMlir(model, input_is_filepath)
