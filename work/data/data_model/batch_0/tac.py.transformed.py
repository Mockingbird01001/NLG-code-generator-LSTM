
from tensorflow.compiler.mlir.lite.experimental.tac.py_wrapper import _pywrap_tac_wrapper
def run_tac(model_path, targets, output_path):
  if not model_path:
    raise ValueError("Invalid model_path.")
  if not targets:
    raise ValueError("Targets are not specified.")
  if not output_path:
    raise ValueError("Invalid output_path.")
  return _pywrap_tac_wrapper.run_tac(model_path, targets, output_path)
