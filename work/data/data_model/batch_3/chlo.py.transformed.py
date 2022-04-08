
from ._chlo_ops_gen import *
def register_chlo_dialect(context, load=True):
  from .._mlir_libs import _mlirHlo
  _mlirHlo.register_chlo_dialect(context, load=load)
