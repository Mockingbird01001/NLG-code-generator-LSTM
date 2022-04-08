
import collections
from tensorflow.python import pywrap_tfe
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.util import compat
VSpace = collections.namedtuple("VSpace", [
    "aggregate_fn", "num_elements_fn", "zeros_fn", "ones_fn",
    "zeros_like_fn", "ones_like_fn", "graph_shape_fn"
])
def imperative_grad(tape,
                    target,
                    sources,
                    output_gradients=None,
                    sources_raw=None,
                    unconnected_gradients=UnconnectedGradients.NONE):
  try:
    unconnected_gradients = UnconnectedGradients(unconnected_gradients)
  except ValueError:
    raise ValueError(
        "Unknown value for unconnected_gradients: %r" % unconnected_gradients)
  return pywrap_tfe.TFE_Py_TapeGradient(
      target,
      sources,
      output_gradients,
      sources_raw,
      compat.as_str(unconnected_gradients.value))
