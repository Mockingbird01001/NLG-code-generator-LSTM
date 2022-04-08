
from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
def _validate_list_constructor(elements, element_dtype, element_shape):
  if element_dtype is not None and element_shape is not None:
    return
  if tensor_util.is_tf_type(elements):
    return
  if isinstance(elements, (list, tuple)):
    if elements:
      return
    else:
      raise ValueError(
          'element_dtype and element_shape are required when elements are'
          ' empty')
  raise ValueError(
      'unknown type for elements: {}; only Tensor, list and tuple are'
      ' allowed'.format(type(elements)))
def match_staging_level(value, like_value):
  if tensor_util.is_tf_type(like_value):
    return constant_op.constant(value)
  return value
def tensor_list(elements,
                element_dtype=None,
                element_shape=None,
                use_tensor_array=False):
  _validate_list_constructor(elements, element_dtype, element_shape)
  if use_tensor_array:
    return data_structures.tf_tensor_array_new(elements, element_dtype,
                                               element_shape)
  else:
    return data_structures.tf_tensor_list_new(elements, element_dtype,
                                              element_shape)
def stack(list_or_tensor, element_dtype=None, strict=True):
  if strict:
    def raise_error(x):
      raise ValueError('%s must be stackable when strict=True' % x)
    original_call = raise_error
  else:
    original_call = lambda x: x
  return data_structures.list_stack(
      list_or_tensor,
      data_structures.ListStackOpts(
          element_dtype=element_dtype, original_call=original_call))
