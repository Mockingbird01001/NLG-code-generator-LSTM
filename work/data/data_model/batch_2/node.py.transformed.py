
import collections
import copy
import json
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
_CONSTANT_VALUE = '_CONSTANT_VALUE'
class Node:
  def __init__(self,
               layer,
               call_args=None,
               call_kwargs=None,
               outputs=None):
    call_args = [] if call_args is None else call_args
    call_kwargs = {} if call_kwargs is None else call_kwargs
    outputs = [] if outputs is None else outputs
    self.layer = layer
    self.is_input = not call_args and not call_kwargs
    call_args = nest.map_structure(lambda t: t, call_args)
    call_kwargs = nest.map_structure(lambda t: t, call_kwargs)
    self.outputs = nest.map_structure(lambda t: t, outputs)
    self.call_args = call_args
    self.call_kwargs = call_kwargs
    self._flat_arguments = nest.flatten((self.call_args, self.call_kwargs))
    self._single_positional_tensor_passed = (not self.call_kwargs and len(
        self.call_args) == 1 and tensor_util.is_tf_type(self.call_args[0]))
    if not ops.executing_eagerly_outside_functions():
      for obj in self._flat_arguments:
        if (isinstance(obj, ops.Tensor) and
            base_layer_utils.needs_keras_history(
                obj, ignore_call_context=True)):
          base_layer_utils.create_keras_history(obj)
    self._keras_inputs = []
    self._keras_inputs_ids_and_indices = []
    for i, ele in enumerate(self._flat_arguments):
      if is_keras_tensor(ele):
        self._keras_inputs.append(ele)
        kt_id = str(id(ele))
        kt_index = i
        self._keras_inputs_ids_and_indices.append((kt_id, kt_index))
    self.layer._inbound_nodes.append(self)
    for kt in self.keras_inputs:
      inbound_layer = kt._keras_history.layer
        inbound_layer._outbound_nodes.append(self)
    node_index = len(self.layer._inbound_nodes) - 1
    for i, tensor in enumerate(nest.flatten(outputs)):
      tensor._keras_history = KerasHistory(
          layer=layer, node_index=node_index, tensor_index=i)
    self.flat_input_ids = [str(id(t)) for t in self._keras_inputs]
    self.flat_output_ids = [str(id(t)) for t in nest.flatten(self.outputs)]
  @property
  def keras_inputs(self):
    return self._keras_inputs
  @property
  def parent_nodes(self):
    node_deps = []
    for kt in self.keras_inputs:
      layer = kt._keras_history.layer
      node_index = kt._keras_history.node_index
        node_deps.append(layer._inbound_nodes[node_index])
    return node_deps
  def iterate_inbound(self):
    """Yields tuples representing the data inbound from other nodes.
    Yields:
      tuples like: (inbound_layer, node_index, tensor_index, tensor).
    """
    for kt in self.keras_inputs:
      keras_history = kt._keras_history
      layer = keras_history.layer
      node_index = keras_history.node_index
      tensor_index = keras_history.tensor_index
      yield layer, node_index, tensor_index, kt
  def map_arguments(self, tensor_dict):
    if self._single_positional_tensor_passed:
      kt_id, _ = self._keras_inputs_ids_and_indices[0]
      return (tensor_dict[kt_id].pop(),), {}
    else:
      flat_arguments = copy.copy(self._flat_arguments)
      for kt_id, kt_index in self._keras_inputs_ids_and_indices:
        flat_arguments[kt_index] = tensor_dict[kt_id].pop()
      args, kwargs = nest.pack_sequence_as((self.call_args, self.call_kwargs),
                                           flat_arguments)
      return args, kwargs
  def serialize(self, make_node_key, node_conversion_map):
    args, kwargs = self.call_args, self.call_kwargs
    inputs, args, kwargs = self.layer._split_out_first_arg(args, kwargs)
    arguments = dict(zip(self.layer._call_fn_args[1:], args))
    arguments.update(kwargs)
    kwargs = arguments
    def _serialize_keras_tensor(t):
      if hasattr(t, '_keras_history'):
        kh = t._keras_history
        node_index = kh.node_index
        node_key = make_node_key(kh.layer.name, node_index)
        new_node_index = node_conversion_map.get(node_key, 0)
        return [kh.layer.name, new_node_index, kh.tensor_index]
      if isinstance(t, np.ndarray):
        return t.tolist()
      if isinstance(t, ops.Tensor):
        return backend.get_value(t).tolist()
      return t
    kwargs = nest.map_structure(_serialize_keras_tensor, kwargs)
    try:
      json.dumps(kwargs, default=json_utils.get_json_type)
    except TypeError:
      kwarg_types = nest.map_structure(type, kwargs)
      raise TypeError('Layer ' + self.layer.name +
                      ' was passed non-JSON-serializable arguments. ' +
                      'Arguments had types: ' +
                      str(kwarg_types) + '. They cannot be serialized out '
                      'when saving the model.')
    def serialize_first_arg_tensor(t):
      if is_keras_tensor(t):
        kh = t._keras_history
        node_index = kh.node_index
        node_key = make_node_key(kh.layer.name, node_index)
        new_node_index = node_conversion_map.get(node_key, 0)
        data = [kh.layer.name, new_node_index, kh.tensor_index, kwargs]
      else:
        data = [_CONSTANT_VALUE, -1, _serialize_keras_tensor(t), kwargs]
      return tf_utils.ListWrapper(data)
    data = nest.map_structure(serialize_first_arg_tensor, inputs)
    if (not nest.is_nested(data) and
        not self.layer._preserve_input_structure_in_config):
      data = [data]
    data = tf_utils.convert_inner_node_data(data)
    return data
  @property
  def input_tensors(self):
    if self.is_input:
    return self.call_args[0]
  @property
  def output_tensors(self):
    if self.is_input:
    return self.outputs
  @property
  def input_shapes(self):
    input_shapes = nest.map_structure(backend.int_shape, self.input_tensors)
    if len(input_shapes) == 1 and not self.is_input:
      return input_shapes[0]
    return input_shapes
  @property
  def output_shapes(self):
    return nest.map_structure(backend.int_shape, self.output_tensors)
  @property
  def outbound_layer(self):
    return self.layer
  @property
  def inbound_layers(self):
    if self.is_input:
      return []
    inbound_layers = nest.map_structure(lambda t: t._keras_history.layer,
                                        self.call_args[0])
    return inbound_layers
class KerasHistory(
    collections.namedtuple('KerasHistory',
                           ['layer', 'node_index', 'tensor_index'])):
  __slots__ = ()
def is_keras_tensor(obj):
  return hasattr(obj, '_keras_history')
