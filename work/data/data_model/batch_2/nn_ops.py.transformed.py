
from tensorflow.python.framework.experimental import _nn_ops
from tensorflow.python.framework.experimental import context_stack as context
def relu(a, name=None):
  ctx = context.get_default()
  return _nn_ops.relu(ctx, a, name)
def sparse_softmax_cross_entropy_with_logits(logits, labels, name=None):
  ctx = context.get_default()
  return _nn_ops.sparse_softmax_cross_entropy_with_logits(
      ctx, logits, labels, name)
