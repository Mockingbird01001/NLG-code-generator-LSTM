
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
@ops.RegisterGradient("OptionalFromValue")
def _OptionalFromValueGrad(op, grad):
  return gen_dataset_ops.optional_get_value(
      grad, [t.dtype for t in op.inputs], [t.shape for t in op.inputs])
@ops.RegisterGradient("OptionalGetValue")
def _OptionalGetValueGrad(unused_op, *grads):
  return gen_dataset_ops.optional_from_value(grads)
