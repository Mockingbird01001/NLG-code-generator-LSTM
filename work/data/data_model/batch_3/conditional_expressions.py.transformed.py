
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.ops import control_flow_ops
def if_exp(cond, if_true, if_false, expr_repr):
  if tensors.is_dense_tensor(cond):
    return _tf_if_exp(cond, if_true, if_false, expr_repr)
  else:
    return _py_if_exp(cond, if_true, if_false)
def _tf_if_exp(cond, if_true, if_false, expr_repr):
  true_val = []
  false_val = []
  def true_fn():
    true_val.append(if_true())
    if true_val and false_val:
      control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return true_val[0]
  def false_fn():
    false_val.append(if_false())
    if true_val and false_val:
      control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return false_val[0]
  return control_flow_ops.cond(cond, true_fn, false_fn)
def _py_if_exp(cond, if_true, if_false):
  return if_true() if cond else if_false()
