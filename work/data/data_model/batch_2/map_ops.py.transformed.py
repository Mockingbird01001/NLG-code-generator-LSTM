
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_map_ops
from tensorflow.python.ops.gen_map_ops import *
ops.NotDifferentiable("EmptyTensorMap")
def empty_tensor_map():
  return gen_map_ops.empty_tensor_map()
def tensor_map_size(input_handle):
  return gen_map_ops.tensor_map_size(input_handle)
def tensor_map_insert(input_handle, key, value):
  return gen_map_ops.tensor_map_insert(input_handle, key, value)
def tensor_map_lookup(input_handle, key, value_dtype):
  return gen_map_ops.tensor_map_lookup(input_handle, key, value_dtype)
def tensor_map_erase(input_handle, key, value_dtype):
  return gen_map_ops.tensor_map_erase(input_handle, key, value_dtype)
def tensor_map_has_key(input_handle, key):
  return gen_map_ops.tensor_map_has_key(input_handle, key)
def tensor_map_stack_keys(input_handle, key_dtype):
  return gen_map_ops.tensor_map_stack_keys(input_handle, key_dtype)
@ops.RegisterGradient("TensorMapLookup")
def LookupGrad(op, dval):
  _, k = op.inputs
  map_grad = empty_tensor_map()
  map_grad = tensor_map_insert(map_grad, k, dval)
  key_grad = None
  return map_grad, key_grad
@ops.RegisterGradient("TensorMapInsert")
def InsertGrad(op, dmap):
  _, k, v = op.inputs
  key_grad = None
  (value_grad, map_grad) = control_flow_ops.cond(
      tensor_map_has_key(dmap, k), lambda:
      (tensor_map_lookup(dmap, k, v.dtype), tensor_map_erase(dmap, k, v.dtype)),
      lambda: (array_ops.zeros_like(v), dmap))
  return map_grad, key_grad, value_grad
@ops.RegisterGradient("TensorMapErase")
def EraseGrad(op, dmap):
  key_grad = None
  map_grad = dmap
  return map_grad, key_grad
