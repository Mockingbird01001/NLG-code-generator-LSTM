
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
def eye(num_rows,
        num_columns=None,
        batch_shape=None,
        dtype=dtypes.float32,
        name=None):
  with ops.name_scope(
      name, default_name='eye', values=[num_rows, num_columns, batch_shape]):
    is_square = num_columns is None
    batch_shape = [] if batch_shape is None else batch_shape
    num_columns = num_rows if num_columns is None else num_columns
    if (isinstance(num_rows, ops.Tensor) or
        isinstance(num_columns, ops.Tensor)):
      diag_size = math_ops.minimum(num_rows, num_columns)
    else:
      if not isinstance(num_rows, compat.integral_types) or not isinstance(
          num_columns, compat.integral_types):
        raise TypeError(
            'Arguments `num_rows` and `num_columns` must be positive integer '
            f'values. Received: num_rows={num_rows}, num_columns={num_columns}')
      is_square = num_rows == num_columns
      diag_size = np.minimum(num_rows, num_columns)
    if isinstance(batch_shape, ops.Tensor) or isinstance(diag_size, ops.Tensor):
      batch_shape = ops.convert_to_tensor(
          batch_shape, name='shape', dtype=dtypes.int32)
      diag_shape = array_ops.concat((batch_shape, [diag_size]), axis=0)
      if not is_square:
        shape = array_ops.concat((batch_shape, [num_rows, num_columns]), axis=0)
    else:
      batch_shape = list(batch_shape)
      diag_shape = batch_shape + [diag_size]
      if not is_square:
        shape = batch_shape + [num_rows, num_columns]
    diag_ones = array_ops.ones(diag_shape, dtype=dtype)
    if is_square:
      return array_ops.matrix_diag(diag_ones)
    else:
      zero_matrix = array_ops.zeros(shape, dtype=dtype)
      return array_ops.matrix_set_diag(zero_matrix, diag_ones)
