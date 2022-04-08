
import six
from tensorflow.python.framework import tensor_shape
def _broadcast_shape_helper(shape_x, shape_y):
  broadcasted_dims = reversed(list(six.moves.zip_longest(
      reversed(shape_x.dims),
      reversed(shape_y.dims),
      fillvalue=tensor_shape.Dimension(1))))
  return_dims = []
  for (dim_x, dim_y) in broadcasted_dims:
    if dim_x.value is None or dim_y.value is None:
      if dim_x.value is not None and dim_x.value > 1:
        return_dims.append(dim_x)
      elif dim_y.value is not None and dim_y.value > 1:
        return_dims.append(dim_y)
      else:
        return_dims.append(None)
    elif dim_x.value == 1:
      return_dims.append(dim_y)
    elif dim_y.value == 1:
      return_dims.append(dim_x)
    elif dim_x.value == dim_y.value:
      return_dims.append(dim_x.merge_with(dim_y))
    else:
      return None
  return return_dims
def is_broadcast_compatible(shape_x, shape_y):
  if shape_x.ndims is None or shape_y.ndims is None:
    return False
  return _broadcast_shape_helper(shape_x, shape_y) is not None
def broadcast_shape(shape_x, shape_y):
  if shape_x.ndims is None or shape_y.ndims is None:
    return tensor_shape.unknown_shape()
  return_dims = _broadcast_shape_helper(shape_x, shape_y)
  if return_dims is None:
    raise ValueError('Incompatible shapes for broadcasting. Two shapes are '
                     'compatible if for each dimension pair they are either '
                     'equal or one of them is 1. '
                     f'Received: {shape_x} and {shape_y}.')
  return tensor_shape.TensorShape(return_dims)
