
from tensorflow.compiler.mlir.tfrt.jit.python_binding import _tfrt_fallback
def run_tfrt_fallback(module_ir, entrypoint, arguments):
  return _tfrt_fallback.run_tfrt_fallback(module_ir, entrypoint, arguments)
