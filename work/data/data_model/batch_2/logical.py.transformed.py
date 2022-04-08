
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
def not_(a):
  if tensor_util.is_tf_type(a):
    return _tf_not(a)
  return _py_not(a)
def _tf_not(a):
  return gen_math_ops.logical_not(a)
def _py_not(a):
  return not a
def and_(a, b):
  a_val = a()
  if tensor_util.is_tf_type(a_val):
    return _tf_lazy_and(a_val, b)
  return _py_lazy_and(a_val, b)
def _tf_lazy_and(cond, b):
  return control_flow_ops.cond(cond, b, lambda: cond)
def _py_lazy_and(cond, b):
  return cond and b()
def or_(a, b):
  a_val = a()
  if tensor_util.is_tf_type(a_val):
    return _tf_lazy_or(a_val, b)
  return _py_lazy_or(a_val, b)
def _tf_lazy_or(cond, b):
  return control_flow_ops.cond(cond, lambda: cond, b)
def _py_lazy_or(cond, b):
  return cond or b()
def eq(a, b):
  if tensor_util.is_tf_type(a) or tensor_util.is_tf_type(b):
    return _tf_equal(a, b)
  return _py_equal(a, b)
def _tf_equal(a, b):
  return gen_math_ops.equal(a, b)
def _py_equal(a, b):
  return a == b
def not_eq(a, b):
  return not_(eq(a, b))
