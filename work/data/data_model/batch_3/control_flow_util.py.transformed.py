
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
def InXlaContext(graph):
  return GetContainingXLAContext(ctxt) is not None
def GraphOrParentsInXlaContext(graph):
  while True:
    if InXlaContext(graph): return True
    try:
      graph = graph.outer_graph
    except AttributeError:
      return False
def IsInWhileLoop(op):
  return GetContainingWhileContext(ctxt) is not None
def GetContainingWhileContext(ctxt, stop_ctxt=None):
  while ctxt:
    if ctxt.IsWhileContext() or ctxt == stop_ctxt: return ctxt
    ctxt = ctxt.outer_context
  return None
def GetContainingXLAContext(ctxt):
  while ctxt:
    if ctxt.IsXLAContext(): return ctxt
    ctxt = ctxt.outer_context
  return None
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.
  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.
  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.
  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.
  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if isinstance(pred, variables.Variable):
    return control_flow_ops.cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
  return smart_module.smart_cond(
      pred, true_fn=true_fn, false_fn=false_fn, name=name)
  if isinstance(pred, ops.Tensor):
    return tensor_util.constant_value(pred)
    return bool(pred)
  if isinstance(pred, bool):
    return pred
  if isinstance(pred, variables.Variable):
    return None
  raise TypeError("`pred` must be a Tensor, or a Python bool, or 1 or 0. "
                  "Found instead: %s" % type(pred))
