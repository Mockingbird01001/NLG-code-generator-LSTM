
import collections
import copy
import itertools
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class Functional(training_lib.Model):
  """A `Functional` model is a `Model` defined as a directed graph of layers.
  Three types of `Model` exist: subclassed `Model`, `Functional` model,
  and `Sequential` (a special case of `Functional`).
  In general, more Keras features are supported with `Functional`
  than with subclassed `Model`s, specifically:
  - Model cloning (`keras.models.clone`)
  - Serialization (`model.get_config()/from_config`, `model.to_json()`
  - Whole-model saving (`model.save()`)
  A `Functional` model can be instantiated by passing two arguments to
  `__init__`. The first argument is the `keras.Input` Tensors that represent
  the inputs to the model. The second argument specifies the output
  tensors that represent the outputs of this model. Both arguments can be a
  nested structure of tensors.
  Example:
  ```
  inputs = {'x1': keras.Input(shape=(10,)), 'x2': keras.Input(shape=(1,))}
  t = keras.layers.Dense(1, activation='relu')(inputs['x1'])
  outputs = keras.layers.Add()([t, inputs['x2'])
  model = keras.Model(inputs, outputs)
  ```
  A `Functional` model constructed using the Functional API can also include raw
  TensorFlow functions, with the exception of functions that create Variables
  or assign ops.
  Example:
  ```
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(1)(inputs)
  outputs = tf.nn.relu(x)
  model = keras.Model(inputs, outputs)
  ```
  Args:
    inputs: List of input tensors (must be created via `tf.keras.Input()`).
    outputs: List of output tensors.
    name: String, optional. Name of the model.
    trainable: Boolean, optional. If the model's variables should be trainable.
  """
  _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
      ('_layer_call_argspecs', '_compiled_trainable_state',
       '_output_mask_cache', '_output_tensor_cache', '_output_shape_cache'),
      training_lib.Model._TF_MODULE_IGNORED_PROPERTIES
  ))
  @trackable.no_automatic_dependency_tracking
  def __init__(self, inputs, outputs, name=None, trainable=True,
               **kwargs):
    skip_init = kwargs.pop('skip_init', False)
    if skip_init:
      return
    generic_utils.validate_kwargs(kwargs, {})
    super(Functional, self).__init__(name=name, trainable=trainable)
    self._init_graph_network(inputs, outputs)
  @trackable.no_automatic_dependency_tracking
  def _init_graph_network(self, inputs, outputs):
    self._is_graph_network = True
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
      inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
      outputs = outputs[0]
    self._nested_inputs = inputs
    self._nested_outputs = outputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)
    if not nest.is_nested(self._nested_inputs):
      self._enable_dict_to_input_mapping = True
    elif (isinstance(self._nested_inputs, (list, tuple)) and
          not any(nest.is_nested(t) for t in self._nested_inputs)):
      self._enable_dict_to_input_mapping = True
    elif (isinstance(self._nested_inputs, dict) and
          not any(nest.is_nested(t) for t in self._nested_inputs.values())):
      self._enable_dict_to_input_mapping = True
    else:
      self._enable_dict_to_input_mapping = False
    if not ops.executing_eagerly_outside_functions():
      if any(not hasattr(tensor, '_keras_history') for tensor in self.outputs):
        base_layer_utils.create_keras_history(self._nested_outputs)
    self._validate_graph_inputs_and_outputs()
    self.built = True
    self._build_input_shape = nest.map_structure(lambda x: x.shape, inputs)
    self._compute_output_and_mask_jointly = True
    self._expects_training_arg = True
    self._expects_mask_arg = True
    self._autocast = False
    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []
    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}
    for x in self.outputs:
      self._output_layers.append(layer)
      self._output_coordinates.append((layer, node_index, tensor_index))
    for x in self.inputs:
      assert node_index == 0
      assert tensor_index == 0
      self._input_layers.append(layer)
      self._input_coordinates.append((layer, node_index, tensor_index))
    nodes, nodes_by_depth, layers, _ = _map_graph_network(
        self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._self_tracked_trackables = layers
    self._layer_call_argspecs = {}
    for layer in self._self_tracked_trackables:
      self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
    self._set_output_names()
    self.input_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for layer in self._input_layers:
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        self._feed_input_shapes.append(layer._batch_input_shape)
        self._feed_inputs.append(layer.input)
    self._compute_tensor_usage_count()
    self._set_save_spec(self._nested_inputs)
    tf_utils.assert_no_legacy_layers(self.layers)
  @property
  def input(self):
    """Retrieves the input tensor(s) of a layer.
    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.
    Returns:
        Input tensor or list of input tensors.
    Raises:
      RuntimeError: If called in Eager mode.
      AttributeError: If no inbound nodes are found.
    """
    return self._nested_inputs
  @property
  def input_shape(self):
    """Retrieves the input shape(s) of a layer.
    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.
    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).
    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    """
    return nest.map_structure(backend.int_shape, self.input)
  @property
  def input_spec(self):
    if hasattr(self, '_manual_input_spec'):
      return self._manual_input_spec
    if (isinstance(self._nested_inputs, (dict, list, tuple)) and
        len(self._nested_inputs) != len(self.inputs)):
      return None
    if isinstance(self._nested_inputs, dict):
      names = sorted(self._nested_inputs.keys())
      return [input_spec.InputSpec(
          shape=shape_with_no_batch_size(self._nested_inputs[name]),
          allow_last_axis_squeeze=True, name=name) for name in names]
    else:
      return [input_spec.InputSpec(
          shape=shape_with_no_batch_size(x), allow_last_axis_squeeze=True,
          name=x._keras_history.layer.name) for x in self.inputs]
  @input_spec.setter
  def input_spec(self, value):
    self._manual_input_spec = value
  @property
  def output(self):
    """Retrieves the output tensor(s) of a layer.
    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.
    Returns:
      Output tensor or list of output tensors.
    Raises:
      AttributeError: if the layer is connected to more than one incoming
        layers.
      RuntimeError: if called in Eager mode.
    """
    return self._nested_outputs
  @property
  def output_shape(self):
    """Retrieves the output shape(s) of a layer.
    Only applicable if the layer has one output,
    or if all outputs have the same shape.
    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).
    Raises:
        AttributeError: if the layer has no defined output shape.
        RuntimeError: if called in Eager mode.
    """
    return nest.map_structure(backend.int_shape, self.output)
  def _set_output_names(self):
    uniquified = []
    output_names = set()
    prefix_count = {}
    for layer in self._output_layers:
      proposal = layer.name
      while proposal in output_names:
        existing_count = prefix_count.get(layer.name, 1)
        proposal = '{}_{}'.format(layer.name, existing_count)
        prefix_count[layer.name] = existing_count + 1
      output_names.add(proposal)
      uniquified.append(proposal)
    self.output_names = uniquified
  @property
  def _layer_checkpoint_dependencies(self):
    weight_layer_index = 0
    dependencies = collections.OrderedDict()
    for layer_index, layer in enumerate(self.layers):
      try:
        if layer.weights:
          dependencies['layer_with_weights-%d' % weight_layer_index] = layer
          weight_layer_index += 1
      except ValueError:
        pass
      dependencies['layer-%d' % layer_index] = layer
    return dependencies
  def _trackable_children(self,
                          save_type=trackable.SaveType.CHECKPOINT,
                          **kwargs):
    dependencies = self._layer_checkpoint_dependencies
    dependencies.update(
        super(Functional, self)._trackable_children(save_type, **kwargs))
    return dependencies
  def _lookup_dependency(self, name):
    layer_dependencies = self._layer_checkpoint_dependencies
    if name in layer_dependencies:
      return layer_dependencies[name]
    return super(Functional, self)._lookup_dependency(name)
  def _handle_deferred_layer_dependencies(self, layers):
    layer_checkpoint_dependencies = self._layer_checkpoint_dependencies
    layer_to_name = {v: k for k, v in layer_checkpoint_dependencies.items()}
    for layer in layers:
      if layer in layer_to_name:
        self._handle_deferred_dependencies(name=layer_to_name[layer],
                                           trackable=layer)
  @property
  def _should_compute_mask(self):
    return True
  def compute_mask(self, inputs, mask):
    output_tensors = self._run_internal_graph(inputs, mask=mask)
    return nest.map_structure(lambda t: getattr(t, '_keras_mask', None),
                              output_tensors)
  @doc_controls.do_not_doc_inheritable
  def call(self, inputs, training=None, mask=None):
    """Calls the model on new inputs.
    In this case `call` just reapplies
    all ops in the graph to the new inputs
    (e.g. build a new computational graph from the provided inputs).
    Args:
        inputs: A tensor or list of tensors.
        training: Boolean or boolean scalar tensor, indicating whether to run
          the `Network` in training mode or inference mode.
        mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).
    Returns:
        A tensor if there is a single output, or
        a list of tensors if there are more than one outputs.
    """
    return self._run_internal_graph(
        inputs, training=training, mask=mask)
  def compute_output_shape(self, input_shape):
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
    if len(nest.flatten(input_shape)) != len(nest.flatten(self._input_layers)):
      raise ValueError('Invalid input_shape argument ' + str(input_shape) +
                       ': model has ' + str(len(self._input_layers)) +
                       ' tensor inputs.')
    try:
      cache_key = tuple(tf_utils.convert_shapes(input_shape, to_tuples=True))
      if cache_key in self._output_shape_cache:
        return self._output_shape_cache[cache_key]
    except ValueError:
      pass
    layers_to_output_shapes = {}
    for layer, shape in zip(self._input_layers, nest.flatten(input_shape)):
      shape_key = layer.name + '_0_0'
      layers_to_output_shapes[shape_key] = shape
    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    if len(depth_keys) > 1:
      for depth in depth_keys:
        nodes = self._nodes_by_depth[depth]
        for node in nodes:
          layer = node.layer
          if layer in self._input_layers:
            continue
          layer_input_shapes = []
          layer_inputs = node.call_args[0]
          for layer_input in nest.flatten(layer_inputs):
            kh = layer_input._keras_history
            input_layer_key = kh.layer.name + '_%s_%s' % (kh.node_index,
                                                          kh.tensor_index)
            layer_input_shapes.append(layers_to_output_shapes[input_layer_key])
          layer_input_shapes = nest.pack_sequence_as(layer_inputs,
                                                     layer_input_shapes)
          layer_input_shapes = tf_utils.convert_shapes(
              layer_input_shapes, to_tuples=True)
          layer_output_shapes = layer.compute_output_shape(layer_input_shapes)
          layer_output_shapes = tf_utils.convert_shapes(
              layer_output_shapes, to_tuples=False)
          for j, shape in enumerate(nest.flatten(layer_output_shapes)):
            shape_key = layer.name + '_%s_%s' % (node_index, j)
            layers_to_output_shapes[shape_key] = shape
      output_shapes = []
      for i in range(len(self._output_layers)):
        layer, node_index, tensor_index = self._output_coordinates[i]
        shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
        output_shapes.append(layers_to_output_shapes[shape_key])
      output_shapes = nest.pack_sequence_as(self._nested_outputs, output_shapes)
      self._output_shape_cache[cache_key] = output_shapes
    return output_shapes
  def _init_set_name(self, name, zero_based=True):
    if not name:
      cls_name = self.__class__.__name__
      if self.__class__ == Functional:
        cls_name = 'Model'
      self._name = backend.unique_object_name(
          generic_utils.to_snake_case(cls_name),
          zero_based=zero_based)
    else:
      self._name = name
  def _run_internal_graph(self, inputs, training=None, mask=None):
    """Computes output tensors for new inputs.
        - Can be run on non-Keras tensors.
    Args:
        inputs: Tensor or nested structure of Tensors.
        training: Boolean learning phase.
        mask: (Optional) Tensor or nested structure of Tensors.
    Returns:
        output_tensors
    """
    inputs = self._flatten_to_reference_inputs(inputs)
    if mask is None:
      masks = [None] * len(inputs)
    else:
      masks = self._flatten_to_reference_inputs(mask)
    for input_t, mask in zip(inputs, masks):
      input_t._keras_mask = mask
    tensor_dict = {}
    tensor_usage_count = self._tensor_usage_count
    for x, y in zip(self.inputs, inputs):
      y = self._conform_to_reference_input(y, ref_input=x)
      x_id = str(id(x))
      tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
    nodes_by_depth = self._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
      nodes = nodes_by_depth[depth]
      for node in nodes:
        if node.is_input:
        if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
        args, kwargs = node.map_arguments(tensor_dict)
        outputs = node.layer(*args, **kwargs)
        for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
          tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
    output_tensors = []
    for x in self.outputs:
      x_id = str(id(x))
      assert x_id in tensor_dict, 'Could not compute output ' + str(x)
      output_tensors.append(tensor_dict[x_id].pop())
    return nest.pack_sequence_as(self._nested_outputs, output_tensors)
  def _flatten_to_reference_inputs(self, tensors):
    if self._enable_dict_to_input_mapping and isinstance(tensors, dict):
      ref_inputs = self._nested_inputs
      if not nest.is_nested(ref_inputs):
        ref_inputs = [self._nested_inputs]
      if isinstance(ref_inputs, dict):
        ref_input_names = sorted(ref_inputs.keys())
      else:
        ref_input_names = [inp._keras_history.layer.name for inp in ref_inputs]
      if len(tensors) > len(ref_input_names):
        warnings.warn(
            'Input dict contained keys {} which did not match any model input. '
            'They will be ignored by the model.'.format(
                [n for n in tensors.keys() if n not in ref_input_names])
            )
      try:
        return [tensors[n] for n in ref_input_names]
      except KeyError:
        return nest.flatten(tensors)
    return nest.flatten(tensors)
  def _conform_to_reference_input(self, tensor, ref_input):
    if isinstance(tensor, ops.Tensor):
      t_shape = tensor.shape
      t_rank = t_shape.rank
      ref_shape = ref_input.shape
      ref_rank = ref_shape.rank
      keras_history = getattr(tensor, '_keras_history', None)
      if t_rank is not None and ref_rank is not None:
        if (t_rank == ref_rank + 1 and t_shape[-1] == 1):
          tensor = array_ops.squeeze_v2(tensor, axis=-1)
        elif (t_rank == ref_rank - 1 and ref_shape[-1] == 1):
          tensor = array_ops.expand_dims_v2(tensor, axis=-1)
        tensor._keras_history = keras_history
      if not context.executing_eagerly():
        try:
          tensor.set_shape(tensor.shape.merge_with(ref_input.shape))
        except ValueError:
          logging.warning(
              'Model was constructed with shape {} for input {}, but it was '
              'called on an input with incompatible shape {}.'.format(
                  ref_input.shape, ref_input, tensor.shape))
      tensor = math_ops.cast(tensor, dtype=ref_input.dtype)
    elif tf_utils.is_extension_type(tensor):
      ref_input_dtype = getattr(ref_input, 'dtype', None)
      if ref_input_dtype is not None and ref_input_dtype != dtypes.variant:
        tensor = math_ops.cast(tensor, dtype=ref_input_dtype)
    return tensor
  def get_config(self):
    return copy.deepcopy(get_network_config(self))
  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Instantiates a Model from its config (output of `get_config()`).
    Args:
        config: Model config dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
    Returns:
        A model instance.
    Raises:
        ValueError: In case of improperly formatted config dict.
    """
    with generic_utils.SharedObjectLoadingScope():
      input_tensors, output_tensors, created_layers = reconstruct_from_config(
          config, custom_objects)
      model = cls(inputs=input_tensors, outputs=output_tensors,
                  name=config.get('name'))
      connect_ancillary_layers(model, created_layers)
      return model
  def _validate_graph_inputs_and_outputs(self):
    if len({id(i) for i in self.inputs}) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))
    for x in self.inputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Input tensors to a ' + cls_name + ' ' +
                         'must come from `tf.keras.Input`. '
                         'Received: ' + str(x) +
                         ' (missing previous layer metadata).')
      layer = x._keras_history.layer
      if len(layer._inbound_nodes) > 1 or (
          layer._inbound_nodes and not layer._inbound_nodes[0].is_input):
        cls_name = self.__class__.__name__
        logging.warning(cls_name + ' model inputs must come from '
                        '`tf.keras.Input` (thus holding past layer metadata), '
                        'they cannot be the output of '
                        'a previous non-Input layer. '
                        'Here, a tensor specified as '
                        'input to "' + self.name + '" was not an Input tensor, '
                        'it was generated by layer ' + layer.name + '.\n'
                        'Note that input tensors are '
                        'instantiated via `tensor = tf.keras.Input(shape)`.\n'
                        'The tensor that caused the issue was: ' + str(x.name))
    input_batch_sizes = [
        training_utils.get_static_batch_size(x._keras_history.layer)
        for x in self.inputs
    ]
    consistent_batch_size = None
    for batch_size in input_batch_sizes:
      if batch_size is not None:
        if (consistent_batch_size is not None and
            batch_size != consistent_batch_size):
          raise ValueError('The specified batch sizes of the Input Layers'
                           ' are incompatible. Found batch sizes: {}'.format(
                               input_batch_sizes))
        consistent_batch_size = batch_size
    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors of a ' + cls_name + ' model must be '
                         'the output of a TensorFlow `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))
  def _insert_layers(self, layers, relevant_nodes=None):
    layers = nest.flatten(layers)
    tf_utils.assert_no_legacy_layers(layers)
    node_to_depth = {}
    for depth, nodes in self._nodes_by_depth.items():
      node_to_depth.update({node: depth for node in nodes})
    if not relevant_nodes:
      relevant_nodes = nest.flatten([layer._inbound_nodes for layer in layers])
    network_nodes = set(relevant_nodes + list(node_to_depth.keys()))
    def _get_min_depth(node):
      min_depth = 0
      for layer, node_id, _, _ in node.iterate_inbound():
        inbound_node = layer._inbound_nodes[node_id]
        if inbound_node in node_to_depth:
          min_depth = min(min_depth, node_to_depth[inbound_node])
        elif inbound_node not in network_nodes:
          continue
        else:
          return None
      return min_depth - 1
    unprocessed_nodes = copy.copy(relevant_nodes)
    i = 0
    while unprocessed_nodes:
      i += 1
      if i > 10000:
        raise ValueError('Layers could not be added due to missing '
                         'dependencies.')
      node = unprocessed_nodes.pop(0)
      depth = _get_min_depth(node)
        unprocessed_nodes.append(node)
        continue
      node_key = _make_node_key(node.layer.name,
                                node.layer._inbound_nodes.index(node))
      if node_key not in self._network_nodes:
        node_to_depth[node] = depth
        self._network_nodes.add(node_key)
        self._nodes_by_depth[depth].append(node)
    layer_set = set(self._self_tracked_trackables)
    deferred_layers = []
    for layer in layers:
      if layer not in layer_set:
        self._self_tracked_trackables.append(layer)
        deferred_layers.append(layer)
        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
        layer_set.add(layer)
    self._handle_deferred_layer_dependencies(deferred_layers)
    self._compute_tensor_usage_count()
  def _compute_tensor_usage_count(self):
    The computed tensor usage count is saved as `self._tensor_usage_count`. This
    is later used for saving memory in eager computation by releasing
    no-longer-needed tensors as early as possible.
    """
    tensor_usage_count = collections.Counter()
    available_tensors = set(str(id(tensor)) for tensor in self.inputs)
    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    depth_keys = depth_keys[1:]
    for depth in depth_keys:
      for node in self._nodes_by_depth[depth]:
        input_tensors = {
            str(id(tensor)) for tensor in nest.flatten(node.keras_inputs)
        }
        if input_tensors.issubset(available_tensors):
          for tensor in nest.flatten(node.keras_inputs):
            tensor_usage_count[str(id(tensor))] += 1
          for output_tensor in nest.flatten(node.outputs):
            available_tensors.add(str(id(output_tensor)))
    for tensor in self.outputs:
      tensor_usage_count[str(id(tensor))] += 1
    self._tensor_usage_count = tensor_usage_count
  def _assert_weights_created(self):
    return
  def _graph_network_add_loss(self, symbolic_loss):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [symbolic_loss])
    add_loss_layer = base_layer.AddLoss(
        unconditional=False, dtype=symbolic_loss.dtype)
    add_loss_layer(symbolic_loss)
    new_nodes.extend(add_loss_layer.inbound_nodes)
    new_layers.append(add_loss_layer)
    self._insert_layers(new_layers, new_nodes)
  def _graph_network_add_metric(self, value, aggregation, name):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [value])
    add_metric_layer = base_layer.AddMetric(
        aggregation, name, dtype=value.dtype)
    add_metric_layer(value)
    new_nodes.extend(add_metric_layer.inbound_nodes)
    new_layers.append(add_metric_layer)
    self._insert_layers(new_layers, new_nodes)
  @property
  def _trackable_saved_model_saver(self):
    return network_serialization.NetworkSavedModelSaver(self)
  def _get_save_spec(self, dynamic_batch=True):
    if getattr(self, '_has_explicit_input_shape', True):
      dynamic_batch = False
    return super(Functional, self)._get_save_spec(dynamic_batch)
def _make_node_key(layer_name, node_index):
  return layer_name + '_ib-' + str(node_index)
def _map_graph_network(inputs, outputs):
  """Validates a network's topology and gather its layers and nodes.
  Args:
    inputs: List of input tensors.
    outputs: List of outputs tensors.
  Returns:
    A tuple `(nodes, nodes_by_depth, layers, layers_by_depth)`.
    - nodes: list of Node instances.
    - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
    - layers: list of Layer instances.
    - layers_by_depth: dict mapping ints (depth) to lists of layer instances.
  Raises:
    ValueError: In case the network is not valid (e.g. disconnected graph).
  """
  nodes_in_decreasing_depth, layer_indices = _build_map(outputs)
  network_nodes = {
      _make_node_key(node.layer.name, node.layer._inbound_nodes.index(node))
      for node in nodes_in_decreasing_depth
  }
  for node in reversed(nodes_in_decreasing_depth):
    depth = nodes_depths.setdefault(node, 0)
    previous_depth = layers_depths.get(node.layer, 0)
    depth = max(depth, previous_depth)
    layers_depths[node.layer] = depth
    nodes_depths[node] = depth
    for node_dep in node.parent_nodes:
      previous_depth = nodes_depths.get(node_dep, 0)
      nodes_depths[node_dep] = max(depth + 1, previous_depth)
  for input_t in inputs:
    input_layer = input_t._keras_history[0]
    if input_layer not in layers_depths:
      layers_depths[input_layer] = 0
      layer_indices[input_layer] = -1
      nodes_depths[input_layer._inbound_nodes[0]] = 0
      network_nodes.add(_make_node_key(input_layer.name, 0))
  nodes_by_depth = collections.defaultdict(list)
  for node, depth in nodes_depths.items():
    nodes_by_depth[depth].append(node)
  layers_by_depth = collections.defaultdict(list)
  for layer, depth in layers_depths.items():
    layers_by_depth[depth].append(layer)
  depth_keys = list(layers_by_depth.keys())
  depth_keys.sort(reverse=True)
  layers = []
  for depth in depth_keys:
    layers_for_depth = layers_by_depth[depth]
    layers_for_depth.sort(key=lambda x: layer_indices[x])
    layers.extend(layers_for_depth)
  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)
  computable_tensors = set()
  for x in inputs:
    computable_tensors.add(id(x))
  for depth in depth_keys:
    for node in nodes_by_depth[depth]:
      layer = node.layer
      if layer and not node.is_input:
        for x in nest.flatten(node.keras_inputs):
          if id(x) not in computable_tensors:
            raise ValueError('Graph disconnected: '
                             'cannot obtain value for tensor ' + str(x) +
                             ' at layer "' + layer.name + '". '
                             'The following previous layers '
                             'were accessed without issue: ' +
                             str(layers_with_complete_input))
        for x in nest.flatten(node.outputs):
          computable_tensors.add(id(x))
        layers_with_complete_input.append(layer.name)
  all_names = [layer.name for layer in layers]
  for name in all_names:
    if all_names.count(name) != 1:
      raise ValueError('The name "' + name + '" is used ' +
                       str(all_names.count(name)) + ' times in the model. '
                       'All layer names should be unique.')
  return network_nodes, nodes_by_depth, layers, layers_by_depth
def _build_map(outputs):
  """This method topologically sorts nodes in order from inputs to outputs.
  It uses a depth-first search to topologically sort nodes that appear in the
  _keras_history connectivity metadata of `outputs`.
  Args:
    outputs: the output tensors whose _keras_history metadata should be walked.
    This may be an arbitrary nested structure.
  Returns:
    A tuple like (ordered_nodes, layer_to_first_traversal_index)
    ordered_nodes: list of nodes appearing in the keras history, topologically
      sorted from original inputs to the `outputs`.
      (If outputs have different sets of ancestors, the inputs to one output
      may appear after a different output).
    layer_to_first_traversal_index:
      A dict mapping layer to the traversal index in the DFS where it is
      seen. Note: if a layer is shared by several nodes, the dict will only
      store the index corresponding to the *first* time the layer seen.
  """
  finished_nodes = set()
  nodes_in_progress = set()
  for output in nest.flatten(outputs):
    _build_map_helper(output, finished_nodes, nodes_in_progress,
                      nodes_in_decreasing_depth, layer_indices)
  return nodes_in_decreasing_depth, layer_indices
def _build_map_helper(tensor, finished_nodes, nodes_in_progress,
                      nodes_in_decreasing_depth, layer_indices):
  if node in finished_nodes:
    return
  if node in nodes_in_progress:
    raise ValueError('The tensor ' + str(tensor) + ' at layer "' + layer.name +
                     '" is part of a cycle.')
  if layer not in layer_indices:
    layer_indices[layer] = len(layer_indices)
  nodes_in_progress.add(node)
  if not node.is_input:
    for tensor in node.keras_inputs:
      _build_map_helper(tensor, finished_nodes, nodes_in_progress,
                        nodes_in_decreasing_depth, layer_indices)
  finished_nodes.add(node)
  nodes_in_progress.remove(node)
  nodes_in_decreasing_depth.append(node)
def _map_subgraph_network(inputs, outputs):
  if not ops.executing_eagerly_outside_functions():
    base_layer_utils.create_keras_history(outputs)
  _, nodes_by_depth, layers, _ = _map_graph_network(inputs, outputs)
  return nest.flatten([nodes for nodes in nodes_by_depth.values()]), layers
def _should_skip_first_node(layer):
  if layer._self_tracked_trackables:
    return (isinstance(layer, Functional) and
            isinstance(layer._self_tracked_trackables[0],
                       input_layer_module.InputLayer))
  else:
    return isinstance(layer, Functional)
def connect_ancillary_layers(model, created_layers):
  ancillary_layers = [
      layer for layer in created_layers.values() if layer not in model.layers
  ]
  if ancillary_layers:
    relevant_nodes = nest.flatten([
        layer.inbound_nodes[1:]
        if _should_skip_first_node(layer) else layer.inbound_nodes
        for layer in created_layers.values()
    ])
    model._insert_layers(ancillary_layers, relevant_nodes)
  return model
def reconstruct_from_config(config, custom_objects=None, created_layers=None):
  """Reconstructs graph from config object.
  Args:
    config: Dictionary returned from Network.get_config()
    custom_objects: Optional dictionary mapping names (strings) to custom
      classes or functions to be considered during deserialization.
    created_layers: Optional dictionary mapping names to Layer objects. Any
      layer not in this dictionary will be created and added to the dict.
      This function will add new nodes to all layers (excluding InputLayers),
      instead of re-using pre-existing nodes in the layers.
  Returns:
    Tuple of (input tensors, output tensors, dictionary of created layers)
  """
  created_layers = created_layers or collections.OrderedDict()
  node_index_map = {}
  node_count_by_layer = {}
  unprocessed_nodes = {}
  def add_unprocessed_node(layer, node_data):
    if layer not in unprocessed_nodes:
      unprocessed_nodes[layer] = [node_data]
    else:
      unprocessed_nodes[layer].append(node_data)
  def get_node_index(layer, config_node_index):
    if isinstance(layer, input_layer_module.InputLayer):
      return 0
    return node_index_map.get((layer.name, config_node_index), None)
  def _deserialize_keras_tensors(kwargs, layer_map):
    def _deserialize_keras_tensor(t):
      if isinstance(t, tf_utils.ListWrapper):
        t = t.as_list()
        layer_name = t[0]
        node_index = t[1]
        tensor_index = t[2]
        layer = layer_map[layer_name]
        new_node_index = get_node_index(layer, node_index)
        if new_node_index is None:
          raise IndexError
        node = layer._inbound_nodes[new_node_index]
        return nest.flatten(node.outputs)[tensor_index]
      return t
    kwargs = tf_utils.convert_inner_node_data(kwargs, wrap=True)
    return nest.map_structure(_deserialize_keras_tensor, kwargs)
  def process_node(layer, node_data):
    input_tensors = []
    for input_data in nest.flatten(node_data):
      input_data = input_data.as_list()
      inbound_layer_name = input_data[0]
      inbound_node_index = input_data[1]
      inbound_tensor_index = input_data[2]
      if len(input_data) == 3:
        kwargs = {}
      elif len(input_data) == 4:
        kwargs = input_data[3]
        try:
          kwargs = _deserialize_keras_tensors(kwargs, created_layers)
        except IndexError:
          add_unprocessed_node(layer, node_data)
          return
      else:
        raise ValueError('Improperly formatted model config.')
      if inbound_layer_name != node_module._CONSTANT_VALUE:
        inbound_layer = created_layers[inbound_layer_name]
        inbound_node_index = get_node_index(inbound_layer, inbound_node_index)
        if inbound_node_index is None:
          add_unprocessed_node(layer, node_data)
          return
        inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
        input_tensors.append(
            nest.flatten(inbound_node.outputs)[inbound_tensor_index])
      else:
        input_tensors.append(inbound_tensor_index)
    input_tensors = nest.pack_sequence_as(node_data, input_tensors)
    if input_tensors is not None:
      if not layer._preserve_input_structure_in_config:
        input_tensors = (
            base_layer_utils.unnest_if_single_tensor(input_tensors))
      output_tensors = layer(input_tensors, **kwargs)
      output_index = nest.flatten(output_tensors)[0]._keras_history.node_index
      node_index_map[(layer.name, node_count_by_layer[layer])] = output_index
      node_count_by_layer[layer] += 1
  def process_layer(layer_data):
    layer_name = layer_data['name']
    if layer_name in created_layers:
      layer = created_layers[layer_name]
    else:
      layer = deserialize_layer(layer_data, custom_objects=custom_objects)
      created_layers[layer_name] = layer
    node_count_by_layer[layer] = int(_should_skip_first_node(layer))
    inbound_nodes_data = layer_data['inbound_nodes']
    inbound_nodes_data = tf_utils.convert_inner_node_data(
        inbound_nodes_data, wrap=True)
    for node_data in inbound_nodes_data:
      add_unprocessed_node(layer, node_data)
  for layer_data in config['layers']:
    process_layer(layer_data)
  while unprocessed_nodes:
    for layer_data in config['layers']:
      layer = created_layers[layer_data['name']]
      if layer in unprocessed_nodes:
        for node_data in unprocessed_nodes.pop(layer):
          process_node(layer, node_data)
  input_tensors = []
  output_tensors = []
  input_layers = tf_utils.convert_inner_node_data(
      config['input_layers'], wrap=True)
  for layer_data in nest.flatten(input_layers):
    layer_name, node_index, tensor_index = layer_data.as_list()
    assert layer_name in created_layers
    layer = created_layers[layer_name]
    node_index = get_node_index(layer, node_index)
    layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
    input_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])
  output_layers = tf_utils.convert_inner_node_data(
      config['output_layers'], wrap=True)
  for layer_data in nest.flatten(output_layers):
    layer_name, node_index, tensor_index = layer_data.as_list()
    assert layer_name in created_layers
    layer = created_layers[layer_name]
    node_index = get_node_index(layer, node_index)
    layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
    output_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])
  input_tensors = nest.pack_sequence_as(input_layers, input_tensors)
  output_tensors = nest.pack_sequence_as(output_layers, output_tensors)
  return input_tensors, output_tensors, created_layers
def get_network_config(network, serialize_layer_fn=None):
  serialize_layer_fn = (
      serialize_layer_fn or generic_utils.serialize_keras_object)
  config = {
      'name': network.name,
  }
  node_conversion_map = {}
  for layer in network.layers:
    kept_nodes = 1 if _should_skip_first_node(layer) else 0
    for original_node_index, node in enumerate(layer._inbound_nodes):
      node_key = _make_node_key(layer.name, original_node_index)
      if node_key in network._network_nodes:
        node_conversion_map[node_key] = kept_nodes
        kept_nodes += 1
  layer_configs = []
  with generic_utils.SharedObjectSavingScope():
      filtered_inbound_nodes = []
      for original_node_index, node in enumerate(layer._inbound_nodes):
        node_key = _make_node_key(layer.name, original_node_index)
        if node_key in network._network_nodes and not node.is_input:
          node_data = node.serialize(_make_node_key, node_conversion_map)
          filtered_inbound_nodes.append(node_data)
      layer_config = serialize_layer_fn(layer)
      layer_config['name'] = layer.name
      layer_config['inbound_nodes'] = filtered_inbound_nodes
      layer_configs.append(layer_config)
    config['layers'] = layer_configs
  model_inputs = []
  for i in range(len(network._input_layers)):
    layer, node_index, tensor_index = network._input_coordinates[i]
    node_key = _make_node_key(layer.name, node_index)
    if node_key not in network._network_nodes:
      continue
    new_node_index = node_conversion_map[node_key]
    model_inputs.append(
        tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
  model_inputs = nest.pack_sequence_as(network._nested_inputs, model_inputs)
  if not nest.is_nested(model_inputs):
    model_inputs = [model_inputs]
  model_inputs = tf_utils.convert_inner_node_data(model_inputs)
  config['input_layers'] = model_inputs
  model_outputs = []
  for i in range(len(network._output_layers)):
    layer, node_index, tensor_index = network._output_coordinates[i]
    node_key = _make_node_key(layer.name, node_index)
    if node_key not in network._network_nodes:
      continue
    new_node_index = node_conversion_map[node_key]
    model_outputs.append(
        tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
  model_outputs = nest.pack_sequence_as(network._nested_outputs, model_outputs)
  if not nest.is_nested(model_outputs):
    model_outputs = [model_outputs]
  model_outputs = tf_utils.convert_inner_node_data(model_outputs)
  config['output_layers'] = model_outputs
  return config
def shape_with_no_batch_size(x):
  if x.shape.rank is None:
    return None
  shape = x.shape.as_list()
  if shape:
    shape[0] = None
  return shape
class ModuleWrapper(base_layer.Layer):
  def __init__(self, module, method_name=None, **kwargs):
    """Initializes the wrapper Layer for this module.
    Args:
      module: The `tf.Module` instance to be wrapped.
      method_name: (Optional) str. The name of the method to use as the forward
        pass of the module. If not set, defaults to '__call__' if defined, or
        'call'.
      **kwargs: Additional keywrod arguments. See `tf.keras.layers.Layer`.
    Raises:
      ValueError: If `method` is not defined on `module`.
    """
    super(ModuleWrapper, self).__init__(**kwargs)
    if method_name is None:
      if hasattr(module, '__call__'):
        method_name = '__call__'
      elif hasattr(module, 'call'):
        method_name = 'call'
    if method_name is None or not hasattr(module, method_name):
      raise ValueError('{} is not defined on object {}'.format(
          method_name, module))
    self._module = module
    self._method_name = method_name
    method = getattr(module, method_name)
    method_arg_spec = tf_inspect.getfullargspec(method)
    self._expects_training_arg = ('training' in method_arg_spec.args or
                                  method_arg_spec.varkw is not None)
    self._expects_mask_arg = ('mask' in method_arg_spec.args or
                              method_arg_spec.varkw is not None)
  def call(self, *args, **kwargs):
    if 'training' in kwargs and not self._expects_training_arg:
      kwargs.pop('training')
    if 'mask' in kwargs and not self._expects_mask_arg:
      kwargs.pop('mask')
    return getattr(self._module, self._method_name)(*args, **kwargs)
