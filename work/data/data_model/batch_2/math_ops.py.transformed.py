
from tensorflow.python.framework.experimental import _math_ops
from tensorflow.python.framework.experimental import context_stack as context
def add(a, b, name=None):
  ctx = context.get_default()
  return _math_ops.add(ctx, a, b, name)
def mat_mul(a, b, name=None):
  ctx = context.get_default()
  return _math_ops.mat_mul(ctx, a, b, name)
def neg(a, name=None):
  ctx = context.get_default()
  return _math_ops.neg(ctx, a, name)
def sub(a, b, name=None):
  ctx = context.get_default()
  return _math_ops.sub(ctx, a, b, name)
def mul(a, b, name=None):
  ctx = context.get_default()
  return _math_ops.mul(ctx, a, b, name)
def log1p(a, name=None):
  ctx = context.get_default()
  return _math_ops.log1p(ctx, a, name)
def div_no_nan(a, b, name=None):
  ctx = context.get_default()
  return _math_ops.div_no_nan(ctx, a, b, name)
