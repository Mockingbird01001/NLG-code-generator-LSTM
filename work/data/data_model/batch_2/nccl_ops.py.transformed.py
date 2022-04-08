
import threading
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops
_module_lock = threading.Lock()
_shared_name_counter = 0
def all_sum(tensors):
  return _apply_all_reduce('sum', tensors)
@ops.RegisterGradient('NcclAllReduce')
def _all_sum_grad(op, grad):
  if op.get_attr('reduction') != b'sum':
    raise LookupError('No gradient defined for NcclAllReduce except for '
                      'reduction="sum".')
  _check_device(grad, expected=op.device)
  num_devices = op.get_attr('num_devices')
  shared_name = op.get_attr('shared_name') + b'_grad'
  with ops.device(op.device):
    return gen_nccl_ops.nccl_all_reduce(
        input=grad,
        reduction='sum',
        num_devices=num_devices,
        shared_name=shared_name)
def all_prod(tensors):
  return _apply_all_reduce('prod', tensors)
def all_min(tensors):
  return _apply_all_reduce('min', tensors)
def all_max(tensors):
  return _apply_all_reduce('max', tensors)
def reduce_sum(tensors):
  return _apply_reduce('sum', tensors)
@ops.RegisterGradient('NcclReduce')
def _reduce_sum_grad(op, grad):
  if op.get_attr('reduction') != b'sum':
    raise LookupError('No gradient defined for NcclAllReduce except for '
                      'reduction="sum".')
  _check_device(grad, expected=op.device)
  with ops.device(op.device):
    result = gen_nccl_ops.nccl_broadcast(input=grad, shape=grad.shape)
  return [result] * len(op.inputs)
def broadcast(tensor):
  _check_device(tensor)
  with ops.device(tensor.device):
    return gen_nccl_ops.nccl_broadcast(input=tensor, shape=tensor.shape)
@ops.RegisterGradient('NcclBroadcast')
def _broadcast_grad(op, accumulated_grad):
  grads = [t for t in accumulated_grad.op.inputs]
  for t in grads:
    _check_device(t)
  with ops.device(op.device):
    return gen_nccl_ops.nccl_reduce(input=grads, reduction='sum')
def _apply_all_reduce(reduction, tensors):
  if not tensors:
    raise ValueError('Must pass >0 tensors to all reduce operations')
  shared_name = _get_shared_name()
  def _all_reduce():
    res = []
    for t in tensors:
      _check_device(t)
      with ops.device(t.device):
        res.append(
            gen_nccl_ops.nccl_all_reduce(
                input=t,
                reduction=reduction,
                num_devices=len(tensors),
                shared_name=shared_name))
    return res
  if context.executing_eagerly():
    return def_function.function(_all_reduce)()
  else:
    return _all_reduce()
def _apply_reduce(reduction, tensors):
  if not tensors:
    raise ValueError('Must pass >0 tensors to reduce operations')
  for t in tensors:
    _check_device(t)
  result = gen_nccl_ops.nccl_reduce(input=tensors, reduction=reduction)
  try:
    next(t for t in tensors if t.device == result.device)
  except StopIteration:
    raise ValueError('One input tensor must be assigned to current device')
  return result
def _get_shared_name():
  global _shared_name_counter
  with _module_lock:
    val = _shared_name_counter
    _shared_name_counter += 1
  return 'c%s' % val
def _check_device(tensor, expected=None):
  if not device.canonical_name(tensor.device):
    raise ValueError(f'Device assignment for tensor={tensor} required for nccl '
                     'collective ops')
  if expected and expected != tensor.device:
    raise ValueError(f'Expected device {expected}, got {tensor.device} for '
                     f'tensor={tensor}.')
