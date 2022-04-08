
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *
def import_graphdef(graphdef,
                    pass_pipeline,
                    show_debug_info,
                    input_names=None,
                    input_data_types=None,
                    input_data_shapes=None,
                    output_names=[]):
  if input_names is not None:
    return ImportGraphDef(
        str(graphdef).encode('utf-8'), pass_pipeline.encode('utf-8'),
        show_debug_info, ','.join(input_names).encode('utf-8'),
        ','.join(input_data_types).encode('utf-8'),
        ':'.join(input_data_shapes).encode('utf-8'),
        ','.join(output_names).encode('utf-8'))
  return ImportGraphDef(
      str(graphdef).encode('utf-8'), pass_pipeline.encode('utf-8'),
      show_debug_info)
def import_function(concrete_function, pass_pipeline, show_debug_info):
  ctxt = context.context()
  ctxt.ensure_initialized()
  return ImportFunction(ctxt._handle,
                        str(concrete_function.function_def).encode('utf-8'),
                        pass_pipeline.encode('utf-8'), show_debug_info)
def experimental_convert_saved_model_to_mlir(saved_model_path, exported_names,
                                             show_debug_info):
  return ExperimentalConvertSavedModelToMlir(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'), show_debug_info)
def experimental_convert_saved_model_v1_to_mlir_lite(saved_model_path,
                                                     exported_names, tags,
                                                     upgrade_legacy,
                                                     show_debug_info):
  return ExperimentalConvertSavedModelV1ToMlirLite(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'),
      str(tags).encode('utf-8'), upgrade_legacy, show_debug_info)
def experimental_convert_saved_model_v1_to_mlir(saved_model_path,
                                                exported_names, tags,
                                                lift_variables, upgrade_legacy,
                                                show_debug_info):
  return ExperimentalConvertSavedModelV1ToMlir(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'),
      str(tags).encode('utf-8'), lift_variables, upgrade_legacy,
      show_debug_info)
def experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info):
  return ExperimentalRunPassPipeline(
      mlir_txt.encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info)
