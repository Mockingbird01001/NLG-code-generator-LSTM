
from tensorflow.python.framework import ops
@ops.RegisterGradient("XlaClusterOutput")
def _XlaClusterOutputGrad(_, grad):
  raise RuntimeError("Gradient computation of graph in xla.compile() is "
                     "prohibited because it can cause performance degradation."
                     "Please move gradient computation inside xla.compile().")
