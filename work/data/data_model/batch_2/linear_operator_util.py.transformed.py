
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def convert_nonref_to_tensor(value, dtype=None, dtype_hint=None, name=None):
  """Converts the given `value` to a `Tensor` if input is nonreference type.
  This function converts Python objects of various types to `Tensor` objects
  except if the input has nonreference semantics. Reference semantics are
  characterized by `is_ref` and is any object which is a
  `tf.Variable` or instance of `tf.Module`. This function accepts any input
  which `tf.convert_to_tensor` would also.
  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.
  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    dtype_hint: Optional element type for the returned tensor,
      used when dtype is None. In some cases, a caller may not have a
      dtype in mind when converting to a tensor, so dtype_hint
      can be used as a soft preference.  If the conversion to
      `dtype_hint` is not possible, this argument has no effect.
    name: Optional name to use if a new `Tensor` is created.
  Returns:
    tensor: A `Tensor` based on `value`.
  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.
  ```python
  x = tf.Variable(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  x = tf.constant(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  x = np.array(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  tf.is_tensor(y)
  x = tfp.util.DeferredTensor(13.37, lambda x: x)
  y = convert_nonref_to_tensor(x)
  x is y
  tf.is_tensor(y)
  tf.equal(y, 13.37)
  ```
  """
  if value is None:
    return None
  if is_ref(value):
    if dtype is None:
      return value
    dtype_base = base_dtype(dtype)
    value_dtype_base = base_dtype(value.dtype)
    if dtype_base != value_dtype_base:
      raise TypeError(
          f"Argument `value` must be of dtype `{dtype_name(dtype_base)}` "
          f"Received: `{dtype_name(value_dtype_base)}`.")
    return value
  return ops.convert_to_tensor_v2_with_dispatch(
      value, dtype=dtype, dtype_hint=dtype_hint, name=name)
def base_dtype(dtype):
  dtype = dtypes.as_dtype(dtype)
  if hasattr(dtype, "base_dtype"):
    return dtype.base_dtype
  return dtype
def dtype_name(dtype):
  dtype = dtypes.as_dtype(dtype)
  if hasattr(dtype, "name"):
    return dtype.name
  if hasattr(dtype, "__name__"):
    return dtype.__name__
  return str(dtype)
def check_dtype(arg, dtype):
  if arg.dtype.base_dtype != dtype:
    raise TypeError(
        f"Expected argument to have dtype {dtype}. Found: {arg.dtype} in "
        f"tensor {arg}.")
def is_ref(x):
  return (
      isinstance(x, variables_module.Variable) or
      (isinstance(x, module.Module) and hasattr(x, "dtype") and
       hasattr(x, "shape")))
def assert_not_ref_type(x, arg_name):
  if is_ref(x):
    raise TypeError(
        f"Argument {arg_name} cannot be reference type. Found: {type(x)}.")
def assert_no_entries_with_modulus_zero(
    x, message=None, name="assert_no_entries_with_modulus_zero"):
  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor_v2_with_dispatch(x, name="x")
    dtype = x.dtype.base_dtype
    should_be_nonzero = math_ops.abs(x)
    zero = ops.convert_to_tensor_v2_with_dispatch(0, dtype=dtype.real_dtype)
    return check_ops.assert_less(zero, should_be_nonzero, message=message)
def assert_zero_imag_part(x, message=None, name="assert_zero_imag_part"):
  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor_v2_with_dispatch(x, name="x")
    dtype = x.dtype.base_dtype
    if dtype.is_floating:
      return control_flow_ops.no_op()
    zero = ops.convert_to_tensor_v2_with_dispatch(0, dtype=dtype.real_dtype)
    return check_ops.assert_equal(zero, math_ops.imag(x), message=message)
def assert_compatible_matrix_dimensions(operator, x):
  """Assert that an argument to solve/matmul has proper domain dimension.
  If `operator.shape[-2:] = [M, N]`, and `x.shape[-2:] = [Q, R]`, then
  `operator.matmul(x)` is defined only if `N = Q`.  This `Op` returns an
  `Assert` that "fires" if this is not the case.  Static checks are already
  done by the base class `LinearOperator`.
  Args:
    operator:  `LinearOperator`.
    x:  `Tensor`.
  Returns:
    `Assert` `Op`.
  """
  assert_same_dd = check_ops.assert_equal(
      array_ops.shape(x)[-2],
      operator.domain_dimension_tensor(),
      message=("Dimensions are not compatible.  "
               "shape[-2] of argument to be the same as this operator"))
  return assert_same_dd
def assert_is_batch_matrix(tensor):
  sh = tensor.shape
  if sh.ndims is not None and sh.ndims < 2:
    raise ValueError(
        f"Expected [batch] matrix to have at least two dimensions. Found: "
        f"{tensor}.")
def shape_tensor(shape, name=None):
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int32
  else:
    dtype = None
  return ops.convert_to_tensor_v2_with_dispatch(shape, dtype=dtype, name=name)
def broadcast_matrix_batch_dims(batch_matrices, name=None):
  """Broadcast leading dimensions of zero or more [batch] matrices.
  Example broadcasting one batch dim of two simple matrices.
  ```python
  x = [[1, 2],
  x_bc, y_bc = broadcast_matrix_batch_dims([x, y])
  x_bc
  ==> [[[1, 2],
  y_bc
  ==> same as y
  ```
  Example broadcasting many batch dims
  ```python
  x = tf.random.normal(shape=(2, 3, 1, 4, 4))
  y = tf.random.normal(shape=(1, 3, 2, 5, 5))
  x_bc, y_bc = broadcast_matrix_batch_dims([x, y])
  x_bc.shape
  ==> (2, 3, 2, 4, 4)
  y_bc.shape
  ==> (2, 3, 2, 5, 5)
  ```
  Args:
    batch_matrices:  Iterable of `Tensor`s, each having two or more dimensions.
    name:  A string name to prepend to created ops.
  Returns:
    bcast_matrices: List of `Tensor`s, with `bcast_matrices[i]` containing
      the values from `batch_matrices[i]`, with possibly broadcast batch dims.
  Raises:
    ValueError:  If any input `Tensor` is statically determined to have less
      than two dimensions.
  """
  with ops.name_scope(
      name or "broadcast_matrix_batch_dims", values=batch_matrices):
    check_ops.assert_proper_iterable(batch_matrices)
    batch_matrices = list(batch_matrices)
    for i, mat in enumerate(batch_matrices):
      batch_matrices[i] = ops.convert_to_tensor_v2_with_dispatch(mat)
      assert_is_batch_matrix(batch_matrices[i])
    if len(batch_matrices) < 2:
      return batch_matrices
    bcast_batch_shape = batch_matrices[0].shape[:-2]
    for mat in batch_matrices[1:]:
      bcast_batch_shape = array_ops.broadcast_static_shape(
          bcast_batch_shape,
          mat.shape[:-2])
    if bcast_batch_shape.is_fully_defined():
      for i, mat in enumerate(batch_matrices):
        if mat.shape[:-2] != bcast_batch_shape:
          bcast_shape = array_ops.concat(
              [bcast_batch_shape.as_list(), array_ops.shape(mat)[-2:]], axis=0)
          batch_matrices[i] = array_ops.broadcast_to(mat, bcast_shape)
      return batch_matrices
    bcast_batch_shape = array_ops.shape(batch_matrices[0])[:-2]
    for mat in batch_matrices[1:]:
      bcast_batch_shape = array_ops.broadcast_dynamic_shape(
          bcast_batch_shape,
          array_ops.shape(mat)[:-2])
    for i, mat in enumerate(batch_matrices):
      batch_matrices[i] = array_ops.broadcast_to(
          mat,
          array_ops.concat(
              [bcast_batch_shape, array_ops.shape(mat)[-2:]], axis=0))
    return batch_matrices
def matrix_solve_with_broadcast(matrix, rhs, adjoint=False, name=None):
  with ops.name_scope(name, "MatrixSolveWithBroadcast", [matrix, rhs]):
    matrix = ops.convert_to_tensor_v2_with_dispatch(matrix, name="matrix")
    rhs = ops.convert_to_tensor_v2_with_dispatch(
        rhs, name="rhs", dtype=matrix.dtype)
    matrix, rhs, reshape_inv, still_need_to_transpose = _reshape_for_efficiency(
        matrix, rhs, adjoint_a=adjoint)
    matrix, rhs = broadcast_matrix_batch_dims([matrix, rhs])
    solution = linalg_ops.matrix_solve(
        matrix, rhs, adjoint=adjoint and still_need_to_transpose)
    return reshape_inv(solution)
def _reshape_for_efficiency(a,
                            b,
                            transpose_a=False,
                            transpose_b=False,
                            adjoint_a=False,
                            adjoint_b=False):
  def identity(x):
    return x
  still_need_to_transpose = True
  if a.shape.ndims is None or b.shape.ndims is None:
    return a, b, identity, still_need_to_transpose
  if a.shape.ndims >= b.shape.ndims:
    return a, b, identity, still_need_to_transpose
  b_extra_ndims = b.shape.ndims - a.shape.ndims
  b_extra_sh = array_ops.shape(b)[:b_extra_ndims]
  b_main_sh = array_ops.shape(b)[b_extra_ndims:]
  a_domain_sz_ = a.shape[-2 if adjoint_a or transpose_a else -1]
  b_eq_sz_ = b.shape[-2 if adjoint_b or transpose_b else -1]
  b_extra_sz_ = (
      np.prod(b.shape[:b_extra_ndims].as_list())
      if b.shape[:b_extra_ndims].is_fully_defined() else None)
  if (a_domain_sz_ is not None and b_eq_sz_ is not None and
      b_extra_sz_ is not None):
    if b_extra_sz_ < 2 or a_domain_sz_ <= b_eq_sz_:
      return a, b, identity, still_need_to_transpose
  if adjoint_a:
    a = array_ops.matrix_transpose(a, conjugate=True)
  elif transpose_a:
    a = array_ops.matrix_transpose(a, conjugate=False)
  if adjoint_b:
    b = array_ops.matrix_transpose(b, conjugate=True)
  elif transpose_a:
    b = array_ops.matrix_transpose(b, conjugate=False)
  still_need_to_transpose = False
  b_extra_sh = array_ops.shape(b)[:b_extra_ndims]
  b_main_sh = array_ops.shape(b)[b_extra_ndims:]
  perm = (
      np.concatenate(
          (np.arange(b_extra_ndims, b.shape.ndims),
           np.arange(0, b_extra_ndims)), 0))
  b_extra_on_end = array_ops.transpose(b, perm=perm)
  b_squashed_end = array_ops.reshape(
      b_extra_on_end, array_ops.concat((b_main_sh[:-1], [-1]), 0))
  def reshape_inv(y):
    y_extra_shape = array_ops.concat(
        (array_ops.shape(y)[:-1], [b_main_sh[-1]], b_extra_sh), 0)
    y_extra_on_end = array_ops.reshape(y, y_extra_shape)
    inverse_perm = np.argsort(perm)
    return array_ops.transpose(y_extra_on_end, perm=inverse_perm)
  return a, b_squashed_end, reshape_inv, still_need_to_transpose
def use_operator_or_provided_hint_unless_contradicting(
    operator, hint_attr_name, provided_hint_value, message):
  op_hint = getattr(operator, hint_attr_name)
  if op_hint is False and provided_hint_value:
    raise ValueError(message)
  if op_hint and provided_hint_value is False:
    raise ValueError(message)
  if op_hint or provided_hint_value:
    return True
  if op_hint is False or provided_hint_value is False:
    return False
  return None
def arg_is_blockwise(block_dimensions, arg, arg_split_dim):
  if (isinstance(arg, (tuple, list)) and len(arg) == len(block_dimensions)):
    if not any(nest.is_nested(x) for x in arg):
      return True
    else:
      arg_dims = [ops.convert_to_tensor_v2_with_dispatch(
          x).shape[arg_split_dim] for x in arg]
      self_dims = [dim.value for dim in block_dimensions]
      if all(self_d is None for self_d in self_dims):
        if len(arg_dims) == 1:
          return False
        elif any(dim != arg_dims[0] for dim in arg_dims):
          return True
        else:
          raise ValueError(
              "Parsing of the input structure is ambiguous. Please input "
              "a blockwise iterable of `Tensor`s or a single `Tensor`.")
      if all(self_d == arg_d or self_d is None
             for self_d, arg_d in zip(self_dims, arg_dims)):
        return True
      self_dim = sum(self_d for self_d in self_dims if self_d is not None)
      if all(s == arg_dims[0] for s in arg_dims) and arg_dims[0] >= self_dim:
        return False
      raise ValueError("Input dimension does not match operator dimension.")
  else:
    return False
def split_arg_into_blocks(block_dims, block_dims_fn, arg, axis=-1):
  block_sizes = [dim.value for dim in block_dims]
  if any(d is None for d in block_sizes):
    block_sizes = block_dims_fn()
  return array_ops.split(arg, block_sizes, axis=axis)
