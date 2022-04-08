
from tensorflow.python.framework import ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def dynamic_list_append(target, element):
  if isinstance(target, tensor_array_ops.TensorArray):
    return target.write(target.size(), element)
  if isinstance(target, ops.Tensor):
    return list_ops.tensor_list_push_back(target, element)
  target.append(element)
  return target
class TensorList(object):
  def __init__(self, shape, dtype):
    self.dtype = dtype
    self.shape = shape
    self.clear()
  def append(self, value):
    self.list_ = list_ops.tensor_list_push_back(self.list_, value)
  def pop(self):
    self.list_, value = list_ops.tensor_list_pop_back(self.list_, self.dtype)
    return value
  def clear(self):
    self.list_ = list_ops.empty_tensor_list(self.shape, self.dtype)
  def count(self):
    return list_ops.tensor_list_length(self.list_)
  def __getitem__(self, key):
    return list_ops.tensor_list_get_item(self.list_, key, self.dtype)
  def __setitem__(self, key, value):
    self.list_ = list_ops.tensor_list_set_item(self.list_, key, value)
