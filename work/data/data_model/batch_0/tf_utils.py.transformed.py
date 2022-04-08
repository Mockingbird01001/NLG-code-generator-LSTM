
import collections
import copy
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
def is_tensor_or_tensor_list(v):
  v = nest.flatten(v)
  if v and isinstance(v[0], ops.Tensor):
    return True
  else:
    return False
def get_reachable_from_inputs(inputs, targets=None):
  """Returns the set of tensors/ops reachable from `inputs`.
  Stops if all targets have been found (target is optional).
  Only valid in Symbolic mode, not Eager mode.
  Args:
    inputs: List of tensors.
    targets: List of tensors.
  Returns:
    A set of tensors reachable from the inputs (includes the inputs themselves).
  """
  inputs = nest.flatten(inputs, expand_composites=True)
  reachable = object_identity.ObjectIdentitySet(inputs)
  if targets:
    remaining_targets = object_identity.ObjectIdentitySet(nest.flatten(targets))
  queue = collections.deque(inputs)
  while queue:
    x = queue.pop()
    if isinstance(x, tuple(_user_convertible_tensor_types)):
      continue
    if isinstance(x, ops.Operation):
      outputs = x.outputs[:] or []
    elif isinstance(x, variables.Variable):
      try:
        outputs = [x.op]
      except AttributeError:
        outputs = []
    elif tensor_util.is_tf_type(x):
      outputs = x.consumers()
    else:
      raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))
    for y in outputs:
      if y not in reachable:
        reachable.add(y)
        if targets:
          remaining_targets.discard(y)
        queue.appendleft(y)
    if targets and not remaining_targets:
      return reachable
  return reachable
def map_structure_with_atomic(is_atomic_fn, map_fn, nested):
  if is_atomic_fn(nested):
    return map_fn(nested)
  if not nest.is_nested(nested):
    raise ValueError(
        'Received non-atomic and non-sequence element: {}'.format(nested))
  if nest.is_mapping(nested):
    values = [nested[k] for k in sorted(nested.keys())]
  elif nest.is_attrs(nested):
    values = _astuple(nested)
  else:
    values = nested
  mapped_values = [
      map_structure_with_atomic(is_atomic_fn, map_fn, ele) for ele in values
  ]
  return nest._sequence_like(nested, mapped_values)
def get_shapes(tensors):
  return nest.map_structure(lambda x: x.shape, tensors)
def convert_shapes(input_shape, to_tuples=True):
  def _is_shape_component(value):
    return value is None or isinstance(value, (int, tensor_shape.Dimension))
  def _is_atomic_shape(input_shape):
    if _is_shape_component(input_shape):
      return True
    if isinstance(input_shape, tensor_shape.TensorShape):
      return True
    if (isinstance(input_shape, (tuple, list)) and
        all(_is_shape_component(ele) for ele in input_shape)):
      return True
    return False
  def _convert_shape(input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if to_tuples:
      input_shape = tuple(input_shape.as_list())
    return input_shape
  return map_structure_with_atomic(_is_atomic_shape, _convert_shape,
                                   input_shape)
class ListWrapper(object):
  def __init__(self, list_to_wrap):
    self._list = list_to_wrap
  def as_list(self):
    return self._list
def convert_inner_node_data(nested, wrap=False):
  def _is_serialized_node_data(nested):
    if (isinstance(nested, list) and (len(nested) in [3, 4]) and
        isinstance(nested[0], str)):
      return True
    return False
  def _is_atomic_nested(nested):
    if isinstance(nested, ListWrapper):
      return True
    if _is_serialized_node_data(nested):
      return True
    return not nest.is_nested(nested)
  def _convert_object_or_list(nested):
    if wrap:
      if isinstance(nested, ListWrapper):
        return nested
      if _is_serialized_node_data(nested):
        return ListWrapper(nested)
      return nested
    else:
      if isinstance(nested, ListWrapper):
        return nested.as_list()
      return nested
  return map_structure_with_atomic(_is_atomic_nested, _convert_object_or_list,
                                   nested)
def shape_type_conversion(fn):
  def wrapper(instance, input_shape):
    if input_shape is not None:
      input_shape = convert_shapes(input_shape, to_tuples=True)
    output_shape = fn(instance, input_shape)
    if output_shape is not None:
      output_shape = convert_shapes(output_shape, to_tuples=False)
    return output_shape
  return wrapper
def are_all_symbolic_tensors(tensors):
  return all(map(is_symbolic_tensor, tensors))
_user_convertible_tensor_types = set()
def is_extension_type(tensor):
  return isinstance(tensor, composite_tensor.CompositeTensor)
def is_symbolic_tensor(tensor):
  """Returns whether a tensor is symbolic (from a TF graph) or an eager tensor.
  A Variable can be seen as either: it is considered symbolic
  when we are in a graph scope, and eager when we are in an eager scope.
  Args:
    tensor: A tensor instance to test.
  Returns:
    True for symbolic tensors, False for eager tensors.
  """
  if isinstance(tensor, ops.Tensor):
    return hasattr(tensor, 'graph')
  elif is_extension_type(tensor):
    component_tensors = nest.flatten(tensor, expand_composites=True)
    return any(hasattr(t, 'graph') for t in component_tensors)
  elif isinstance(tensor, variables.Variable):
    return (getattr(tensor, '_keras_history', False) or
            not context.executing_eagerly())
  elif isinstance(tensor, tuple(_user_convertible_tensor_types)):
    tensor = ops.convert_to_tensor_or_composite(tensor)
    return is_symbolic_tensor(tensor)
  else:
    return False
@keras_export('keras.__internal__.utils.register_symbolic_tensor_type', v1=[])
def register_symbolic_tensor_type(cls):
  """Allows users to specify types regarded as symbolic `Tensor`s.
  Used in conjunction with `tf.register_tensor_conversion_function`, calling
  `tf.keras.__internal__.utils.register_symbolic_tensor_type(cls)`
  allows non-`Tensor` objects to be plumbed through Keras layers.
  Example:
  ```python
  class Foo(object):
    def __init__(self, input_):
      self._input = input_
    def value(self):
      return tf.constant(42.)
  tf.register_tensor_conversion_function(
      Foo, lambda x, *args, **kwargs: x.value())
  tf.keras.__internal__.utils.register_symbolic_tensor_type(Foo)
  layer = tf.keras.layers.Lambda(lambda input_: Foo(input_))
  ```
  Args:
    cls: A `class` type which shall be regarded as a symbolic `Tensor`.
  """
  global _user_convertible_tensor_types
  if cls not in _user_convertible_tensor_types:
    keras_tensor.register_keras_tensor_specialization(
        cls, keras_tensor.UserRegisteredTypeKerasTensor)
  _user_convertible_tensor_types.add(cls)
def type_spec_from_value(value):
  if is_extension_type(value):
  if hasattr(value, 'shape') and hasattr(value, 'dtype'):
    return tensor_spec.TensorSpec(value.shape, value.dtype)
  else:
    return type_spec.type_spec_from_value(value)
def is_ragged(tensor):
  return isinstance(
      tensor,
      (ragged_tensor.RaggedTensor, ragged_tensor_value.RaggedTensorValue))
def is_sparse(tensor):
  return isinstance(
      tensor,
      (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue))
def is_tensor_or_variable(x):
  return tensor_util.is_tf_type(x) or isinstance(x, variables.Variable)
def assert_no_legacy_layers(layers):
  legacy_layers = [l for l in layers if getattr(l, '_is_legacy_layer', None)]
  if legacy_layers:
    layer_str = '\n'.join('  ' + str(l) for l in legacy_layers)
    raise TypeError(
        'The following are legacy tf.layers.Layers:\n{}\nTo use keras as a '
        'framework (for instance using the Network, Model, or Sequential '
        'classes), please use the tf.keras.layers implementation instead. '
        '(Or, if writing custom layers, subclass from tf.keras.layers rather '
        'than tf.layers)'.format(layer_str))
@tf_contextlib.contextmanager
def maybe_init_scope(layer):
  if (ops.executing_eagerly_outside_functions() and
      getattr(layer, '_keras_style', True)):
    with ops.init_scope():
      yield
  else:
    yield
@tf_contextlib.contextmanager
def graph_context_for_symbolic_tensors(*args, **kwargs):
  if any(is_symbolic_tensor(v) for v in list(args) + list(kwargs.values())):
    with K.get_graph().as_default():
      yield
  else:
    yield
def dataset_is_infinite(dataset):
  if ops.executing_eagerly_outside_functions():
    return math_ops.equal(
        cardinality.cardinality(dataset), cardinality.INFINITE)
  else:
    dataset_size = K.get_session().run(cardinality.cardinality(dataset))
    return dataset_size == cardinality.INFINITE
def get_tensor_spec(t, dynamic_batch=False, name=None):
  if isinstance(t, type_spec.TypeSpec):
    spec = t
  elif is_extension_type(t):
    spec = t._type_spec
  elif (hasattr(t, '_keras_history') and
        hasattr(t._keras_history[0], '_type_spec')):
    return t._keras_history[0]._type_spec
  elif hasattr(t, 'shape') and hasattr(t, 'dtype'):
    spec = tensor_spec.TensorSpec(shape=t.shape, dtype=t.dtype, name=name)
  else:
  if not dynamic_batch:
    return spec
  dynamic_batch_spec = copy.deepcopy(spec)
  shape = dynamic_batch_spec._shape
  if shape.rank is not None and shape.rank > 0:
    shape_list = shape.as_list()
    shape_list[0] = None
    dynamic_batch_spec._shape = tensor_shape.TensorShape(shape_list)
  return dynamic_batch_spec
def sync_to_numpy_or_python_type(tensors):
  """Syncs and converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.
  For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
  it converts it to a Python type, such as a float or int, by calling
  `result.item()`.
  Numpy scalars are converted, as Python types are often more convenient to deal
  with. This is especially useful for bfloat16 Numpy scalars, which don't
  support as many operations as other Numpy values.
  Async strategies (such as `TPUStrategy` and `ParameterServerStrategy`) are
  forced to
  sync during this process.
  Args:
    tensors: A structure of tensors.
  Returns:
    `tensors`, but scalar tensors are converted to Python types and non-scalar
    tensors are converted to Numpy arrays.
  """
  if isinstance(tensors, coordinator_lib.RemoteValue):
    return tensors.fetch()
  def _to_single_numpy_or_python_type(t):
    if isinstance(t, ops.Tensor):
      x = t.numpy()
      return x.item() if np.ndim(x) == 0 else x
  return nest.map_structure(_to_single_numpy_or_python_type, tensors)
def _astuple(attrs):
  cls = type(attrs)
  fields = getattr(cls, '__attrs_attrs__', None)
  if fields is None:
    raise ValueError('%r is not an attrs-decorated class.' % cls)
  values = []
  for field in fields:
    values.append(getattr(attrs, field.name))
  return tuple(values)
