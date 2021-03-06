
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.types import core
_autocast_dtype = threading.local()
def numpy_text(tensor, is_repr=False):
  if tensor.dtype.is_numpy_compatible:
    text = repr(tensor._numpy()) if is_repr else str(tensor._numpy())
  else:
    text = '<unprintable>'
  if '\n' in text:
    text = '\n' + text
  return text
class AutoCastVariable(variables.Variable, core.Tensor):
  """Variable that will cast itself to a different dtype in applicable contexts.
  This class wraps a floating-point `tf.Variable`. It emulates the variable
  interface and delegates to the wrapped variable, but it additionally will cast
  the wrapped variable under an `enable_auto_cast_variables(dtype)` context
  manager.
  For example:
  >>> v = tf.Variable(1.0, dtype=tf.float32)
  >>> v = AutoCastVariable(v)
  >>> tf.identity(v).dtype
  tf.float32
  >>> with enable_auto_cast_variables(tf.float16):
  ...   tf.identity(v).dtype
  tf.float16
  The purpose of this class is to allow Keras layers to create variables in
  float32, and automatically cast them to float16 or bfloat16 when the layer is
  called.
  """
  def __init__(self, variable):
    if not isinstance(variable, variables.Variable):
      raise ValueError('variable must be of type tf.ResourceVariable, but got: '
                       '%s' % variable)
    if not variable.dtype.is_floating:
      raise ValueError('variable must be a floating point variable but has '
                       'type: %s' % variable.dtype.name)
    self._variable = variable
    self._op = 'delegate'
  def _should_cast(self):
    autocast_dtype = getattr(_autocast_dtype, 'dtype', None)
    return autocast_dtype is not None and self.dtype != autocast_dtype
  @property
  def dtype(self):
    return self._variable.dtype
  @property
  def true_dtype(self):
    return self._variable.dtype
  @property
  def _cast_dtype(self):
    dtype = getattr(_autocast_dtype, 'dtype', None)
    return dtype or self._variable.dtype
  def value(self):
    val = self._variable.value()
    if not self._should_cast():
      return val
    return math_ops.cast(val, self._cast_dtype)
  def read_value(self):
    val = self._variable.read_value()
    return math_ops.cast(val, self._cast_dtype)
  def sparse_read(self, indices, name=None):
    val = self._variable.sparse_read(indices, name=name)
    return math_ops.cast(val, self._cast_dtype)
  def gather_nd(self, indices, name=None):
    val = self._variable.gather_nd(indices, name=name)
    return math_ops.cast(val, self._cast_dtype)
  def __getattr__(self, name):
    return getattr(self._variable, name)
  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    if as_ref:
      raise ValueError('Cannot convert AutoCastVariable to a tensor if '
                       'as_ref=True is passed to convert_to_tensor')
    if not self._should_cast():
      return ops.convert_to_tensor_v2_with_dispatch(self._variable, dtype=dtype,
                                                    name=name)
    if dtype is not None and not dtype.is_compatible_with(self._cast_dtype):
      raise ValueError(
          'Incompatible type conversion requested to type {!r} for '
          'AutoCastVariable which is casted to type {!r}'.format(
              dtype.name, self._cast_dtype.name))
    val = ops.convert_to_tensor_v2_with_dispatch(
        self._variable, dtype=self._variable.dtype, name=name)
    return math_ops.cast(val, self._cast_dtype)
  def _should_act_as_resource_variable(self):
    pass
  def __repr__(self):
    if context.executing_eagerly() and not self._in_graph_mode:
      repr_str = ("<AutoCastVariable '{v.name}' shape={v.shape} "
                  'dtype={v.dtype.name} dtype_to_cast_to={v._cast_dtype.name}, '
                  'numpy={np_repr}>')
      return repr_str.format(
          v=self, np_repr=numpy_text(self.read_value(), is_repr=True))
    else:
      repr_str = ("<AutoCastVariable '{v.name}' shape={v.shape} "
                  'dtype={v.dtype.name} dtype_to_cast_to={v._cast_dtype.name}>')
      return repr_str.format(v=self)
  def set_shape(self, shape):
    return self._variable.set_shape(self, shape)
  @property
  def trainable(self):
    return self._variable.trainable
  @property
  def synchronization(self):
    return self._variable.synchronization
  @property
  def aggregation(self):
    return self._variable.aggregation
  def eval(self, session=None):
    return self._variable.eval(session)
  def initialized_value(self):
    return self._variable.initialized_value()
  @property
  def initial_value(self):
    return self._variable.initial_value
  @property
  def constraint(self):
    return self._variable.constraint
  def _apply_assign_update(self,
                           update_fn,
                           value,
                           use_locking=None,
                           name=None,
                           read_value=True):
    if ops.executing_eagerly_outside_functions():
      assign_op = update_fn(value, use_locking, name, False)
      if read_value:
        var = create_autocast_variable(self._variable)
        return var
      return assign_op
    assign_var = update_fn(value, use_locking, name, read_value)
    if read_value and resource_variable_ops.is_resource_variable(assign_var):
      return create_autocast_variable(assign_var)
    return assign_var
  def _apply_update(self, update_fn, *args, **kwargs):
    update_var = update_fn(*args, **kwargs)
    if ops.executing_eagerly_outside_functions():
      return self
    if resource_variable_ops.is_resource_variable(update_var):
      return create_autocast_variable(update_var)
    return update_var
  def assign(self, value, use_locking=None, name=None, read_value=True):
    return self._apply_assign_update(self._variable.assign, value, use_locking,
                                     name, read_value)
  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    return self._apply_assign_update(self._variable.assign_add, delta,
                                     use_locking, name, read_value)
  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    return self._apply_assign_update(self._variable.assign_sub, delta,
                                     use_locking, name, read_value)
  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.scatter_sub, sparse_delta,
                              use_locking, name)
  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.scatter_add, sparse_delta,
                              use_locking, name)
  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.scatter_max, sparse_delta,
                              use_locking, name)
  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.scatter_min, sparse_delta,
                              use_locking, name)
  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.scatter_mul, sparse_delta,
                              use_locking, name)
  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.scatter_div, sparse_delta,
                              use_locking, name)
  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.scatter_update, sparse_delta,
                              use_locking, name)
  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.batch_scatter_update, sparse_delta,
                              use_locking, name)
  def scatter_nd_sub(self, indices, updates, name=None):
    return self._apply_update(self._variable.scatter_nd_sub, indices, updates,
                              name)
  def scatter_nd_add(self, indices, updates, name=None):
    return self._apply_update(self._variable.scatter_nd_add, indices, updates,
                              name)
  def scatter_nd_update(self, indices, updates, name=None):
    return self._apply_update(self._variable.scatter_nd_update, indices,
                              updates, name)
  def load(self, value, session=None):
    return self._variable.load(value, session)
  @property
  def name(self):
    return self._variable.name
  @property
  def _shared_name(self):
  @property
  def initializer(self):
    return self._variable.initializer
  @property
  def device(self):
    return self._variable.device
  @property
  def op(self):
    if self._op == 'delegate':
      return self._variable.op
    return self._op
  def _as_graph_element(self):
    if graph_element is None:
      return self._op
    return graph_element
  @property
  def graph(self):
    return self._variable.graph
  @property
  def shape(self):
    return self._variable.shape
  def get_shape(self):
    return self._variable.get_shape()
  def _gather_saveables_for_checkpoint(self):
  def _map_resources(self, save_options):
    obj_map[self] = obj_map[self._variable]
    return obj_map, resource_map
  def to_proto(self, export_scope=None):
    return self._variable.to_proto(export_scope)
  def from_proto(self, variable_def, import_scope=None):
    return self._variable.from_proto(variable_def, import_scope)
  @property
  def _handle_name(self):
  @_handle_name.setter
  def _handle_name(self, handle_name):
  @property
  def _initializer_op(self):
  @_initializer_op.setter
  def _initializer_op(self, initializer_op):
  def __add__(self, o):
    return self.read_value() + o
  def __radd__(self, o):
    return o + self.read_value()
  def __sub__(self, o):
    return self.read_value() - o
  def __rsub__(self, o):
    return o - self.read_value()
  def __mul__(self, o):
    return self.read_value() * o
  def __rmul__(self, o):
    return o * self.read_value()
  def __truediv__(self, o):
    return self.read_value() / o
  def __rtruediv__(self, o):
    return o / self.read_value()
  def __floordiv__(self, o):
    return self.read_value() // o
  def __rfloordiv__(self, o):
    return o // self.read_value()
  def __mod__(self, o):
    return self.read_value() % o
  def __rmod__(self, o):
    return o % self.read_value()
  def __lt__(self, o):
    return self.read_value() < o
  def __le__(self, o):
    return self.read_value() <= o
  def __gt__(self, o):
    return self.read_value() > o
  def __ge__(self, o):
    return self.read_value() >= o
  def __getitem__(self, o):
    return self.read_value()[o]
  def __pow__(self, o, modulo=None):
    return pow(self.read_value(), o, modulo)
  def __rpow__(self, o):
    return pow(o, self.read_value())
  def __neg__(self):
  def __abs__(self):
    return abs(self.read_value())
  def __div__(self, o):
    try:
      return self.read_value().__div__(o)
    except AttributeError:
      return NotImplemented
  def __rdiv__(self, o):
    try:
      return self.read_value().__rdiv__(o)
    except AttributeError:
      return NotImplemented
  def __matmul__(self, o):
    try:
      return self.read_value().__matmul__(o)
    except AttributeError:
      return NotImplemented
  def __rmatmul__(self, o):
    try:
      return self.read_value().__rmatmul__(o)
    except AttributeError:
      return NotImplemented
ops.register_tensor_conversion_function(AutoCastVariable,
def create_autocast_variable(variable):
  """Creates an AutoCastVariable that wraps another variable.
  This typically just returns `AutoCastVariable(variable)`. But, if the variable
  is a DistributedVariable or one of its subclasses, we instead dynamically
  create a class that subclasses from both AutoCastVariable and
  variable.__class__. This is so the returned variable will still pass
  `isinstance(variable, variable.__class__)`, which is required for
  DistributedVariables and its subclasses to work properly.
  Args:
    variable: A floating-point resource variable to wrap.
  Returns:
    An AutoCastVariable that wraps the variable.
  """
  if not distributed_training_utils.is_distributed_variable(variable):
    return AutoCastVariable(variable)
  class AutoCastDistributedVariable(AutoCastVariable, variable.__class__):
    def __repr__(self):
      return ('<AutoCastDistributedVariable dtype={v.dtype.name} '
              'dtype_to_cast_to={v._cast_dtype.name} '
              'inner_variable={v._variable}>'
             ).format(v=self)
  return AutoCastDistributedVariable(variable)
  __slots__ = ['_dtype', '_prev_dtype']
  def __init__(self, dtype):
    if dtype and not dtype.is_floating:
      dtype = None
    self._dtype = dtype
  def __enter__(self):
    self._prev_dtype = getattr(_autocast_dtype, 'dtype', None)
    _autocast_dtype.dtype = self._dtype
  def __exit__(self, type_arg, value_arg, traceback_arg):
    _autocast_dtype.dtype = self._prev_dtype
