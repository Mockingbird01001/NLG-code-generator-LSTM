
import contextlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _IsControlInput(input_name):
  return input_name.startswith('^')
def _ParseTensorName(tensor_name):
  """Parses a tensor name into an operation name and output index.
  This function will canonicalize tensor names as follows:
  * "foo:0"       -> ("foo", 0)
  * "foo:7"       -> ("foo", 7)
  * "foo"         -> ("foo", 0)
  * "foo:bar:baz" -> ValueError
  Args:
    tensor_name: The name of a tensor.
  Returns:
    A tuple containing the operation name, and the output index.
  Raises:
    ValueError: If `tensor_name' cannot be interpreted as the name of a tensor.
  """
  components = tensor_name.split(':')
  if len(components) == 2:
    try:
      output_index = int(components[1])
    except ValueError:
      raise ValueError(f'Cannot convert {tensor_name!r} to a tensor name. '
                       'Second component of the name following the `:` should '
                       f'be an int. Got {components[1]}.')
    return components[0], output_index
  elif len(components) == 1:
    return components[0], 0
  else:
    raise ValueError(f"Cannot convert '{tensor_name}' to a tensor name. Tensor "
                     'names should not contain more than 1 `:`. Obtained '
                     f'{len(components) - 1}')
@contextlib.contextmanager
def _MaybeDevice(device):
  if device:
    with ops.device(device):
      yield
  else:
    yield
def _ProcessGraphDefParam(graph_def):
  if not isinstance(graph_def, graph_pb2.GraphDef):
    try:
      old_graph_def = graph_def
      graph_def = graph_pb2.GraphDef()
      graph_def.MergeFrom(old_graph_def)
    except TypeError:
      raise TypeError('Argument `graph_def` must be a GraphDef proto.')
  else:
    for node in graph_def.node:
      op_def = op_def_registry.get(node.op)
      if op_def is None:
        continue
      _SetDefaultAttrValues(node, op_def)
  return graph_def
def _ProcessInputMapParam(input_map):
  if input_map is None:
    input_map = {}
  else:
    if not isinstance(input_map, dict):
      raise TypeError('Argument `input_map` must be a dictionary. Obtained '
                      f'{type(input_map).__name__}')
    if not all(
        isinstance(k, compat.bytes_or_text_types) for k in input_map.keys()):
      raise TypeError('All keys for argument `input_map` must be strings. '
                      f'Obtained keys: {list(input_map.keys())}')
  return input_map
def _ProcessReturnElementsParam(return_elements):
  if return_elements is None:
    return None
  if not all(
      isinstance(x, compat.bytes_or_text_types) for x in return_elements):
    raise TypeError('Argument `return_elements` must be a list of strings. '
                    f'Obtained {return_elements}.')
  return tuple(compat.as_str(x) for x in return_elements)
def _FindAttrInOpDef(attr_name, op_def):
  for attr_def in op_def.attr:
    if attr_name == attr_def.name:
      return attr_def
  return None
def _RemoveDefaultAttrs(producer_op_list, graph_def):
  """Removes unknown default attrs according to `producer_op_list`.
  Removes any unknown attrs in `graph_def` (i.e. attrs that do not appear in
  registered OpDefs) that have a default value in `producer_op_list`.
  Args:
    producer_op_list: OpList proto.
    graph_def: GraphDef proto
  """
  producer_op_dict = {op.name: op for op in producer_op_list.op}
  for node in graph_def.node:
    if node.op in producer_op_dict:
      op_def = op_def_registry.get(node.op)
      if op_def is None:
        continue
      producer_op_def = producer_op_dict[node.op]
      for key in list(node.attr):
        if _FindAttrInOpDef(key, op_def) is None:
          attr_def = _FindAttrInOpDef(key, producer_op_def)
          if (attr_def and attr_def.HasField('default_value') and
              node.attr[key] == attr_def.default_value):
            del node.attr[key]
def _ConvertInputMapValues(name, input_map):
  if not all(isinstance(v, ops.Tensor) for v in input_map.values()):
      raise ValueError(
          'tf.import_graph_def() requires a non-empty `name` if `input_map` '
          'contains non-Tensor values. Try calling tf.convert_to_tensor() on '
          '`input_map` values before calling tf.import_graph_def().')
    with ops.name_scope('_inputs'):
      input_map = {k: ops.convert_to_tensor(v) for k, v in input_map.items()}
  return input_map
def _PopulateTFImportGraphDefOptions(options, prefix, input_map,
                                     return_elements,
                                     validate_colocation_constraints):
  c_api.TF_ImportGraphDefOptionsSetPrefix(options, prefix)
  c_api.TF_ImportGraphDefOptionsSetUniquifyNames(options, True)
  for input_src, input_dst in input_map.items():
    input_src = compat.as_str(input_src)
    if input_src.startswith('^'):
      src_name = compat.as_str(input_src[1:])
      c_api.TF_ImportGraphDefOptionsRemapControlDependency(
          options, src_name, dst_op)
    else:
      src_name, src_idx = _ParseTensorName(input_src)
      src_name = compat.as_str(src_name)
      c_api.TF_ImportGraphDefOptionsAddInputMapping(options, src_name, src_idx,
                                                    dst_output)
  for name in return_elements or []:
    if ':' in name:
      op_name, index = _ParseTensorName(name)
      op_name = compat.as_str(op_name)
      c_api.TF_ImportGraphDefOptionsAddReturnOutput(options, op_name, index)
    else:
      c_api.TF_ImportGraphDefOptionsAddReturnOperation(options,
                                                       compat.as_str(name))
  c_api.TF_ImportGraphDefOptionsSetValidateColocationConstraints(
      options, validate_colocation_constraints)
def _ProcessNewOps(graph):
  colocation_pairs = {}
    original_device = new_op.device
    colocation_names = _GetColocationNames(new_op)
    if colocation_names:
      colocation_pairs[new_op] = colocation_names
    else:
      with _MaybeDevice(original_device):
  for op, coloc_op_list in colocation_pairs.items():
    coloc_device = None
    for coloc_op_name in coloc_op_list:
      try:
      except KeyError:
        if tf2.enabled() or control_flow_util.EnableControlFlowV2(graph):
          continue
        raise ValueError(f'Specified colocation to an op: {coloc_op_name} that '
                         f'does not exist during import for op: {op.name}')
      if coloc_op.device:
        coloc_device = pydev.DeviceSpec.from_string(coloc_op.device)
        break
    if coloc_device:
def _GetColocationNames(op):
  colocation_names = []
  try:
    class_values = op.get_attr('_class')
  except ValueError:
    return
  for val in class_values:
    val = compat.as_str(val)
    if val.startswith('loc:@'):
      colocation_node_name = val[len('loc:@'):]
      if colocation_node_name != op.name:
        colocation_names.append(colocation_node_name)
  return colocation_names
def _GatherReturnElements(requested_return_elements, graph, results):
  return_outputs = c_api.TF_ImportGraphDefResultsReturnOutputs(results)
  return_opers = c_api.TF_ImportGraphDefResultsReturnOperations(results)
  combined_return_elements = []
  outputs_idx = 0
  opers_idx = 0
  for name in requested_return_elements:
    if ':' in name:
      combined_return_elements.append(
      outputs_idx += 1
    else:
      combined_return_elements.append(
      opers_idx += 1
  return combined_return_elements
def _SetDefaultAttrValues(node_def, op_def):
  assert node_def.op == op_def.name
  for attr_def in op_def.attr:
    key = attr_def.name
    if attr_def.HasField('default_value'):
      value = node_def.attr[key]
      if value is None or value.WhichOneof('value') is None:
        node_def.attr[key].CopyFrom(attr_def.default_value)
@tf_export('graph_util.import_graph_def', 'import_graph_def')
@deprecated_args(None, 'Please file an issue at '
                 'https://github.com/tensorflow/tensorflow/issues if you depend'
                 ' on this feature.', 'op_dict')
def import_graph_def(graph_def,
                     input_map=None,
                     return_elements=None,
                     name=None,
                     op_dict=None,
                     producer_op_list=None):
  """Imports the graph from `graph_def` into the current default `Graph`.
  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  `tf.Tensor` and `tf.Operation` objects. Once extracted,
  these objects are placed into the current default `Graph`. See
  `tf.Graph.as_graph_def` for a way to create a `GraphDef`
  proto.
  Args:
    graph_def: A `GraphDef` proto containing operations to be imported into
      the default graph.
    input_map: A dictionary mapping input names (as strings) in `graph_def`
      to `Tensor` objects. The values of the named input tensors in the
      imported graph will be re-mapped to the respective `Tensor` values.
    return_elements: A list of strings containing operation names in
      `graph_def` that will be returned as `Operation` objects; and/or
      tensor names in `graph_def` that will be returned as `Tensor` objects.
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Note that this does not apply to imported function names.
      Defaults to `"import"`.
    op_dict: (Optional.) Deprecated, do not use.
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided,
      unrecognized attrs for ops in `graph_def` that have their default value
      according to `producer_op_list` will be removed. This will allow some more
      `GraphDef`s produced by later binaries to be accepted by earlier binaries.
  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`,
    and None if `returns_elements` is None.
  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
  del op_dict
  return _import_graph_def_internal(
      graph_def,
      input_map=input_map,
      return_elements=return_elements,
      name=name,
      producer_op_list=producer_op_list)
    graph_def, name=None):
  return _import_graph_def_internal(
      graph_def, validate_colocation_constraints=False, name=name)
    graph_def,
    input_map=None,
    return_elements=None,
    validate_colocation_constraints=True,
    name=None,
    producer_op_list=None):
  """Imports the graph from `graph_def` into the current default `Graph`.
  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  `tf.Tensor` and `tf.Operation` objects. Once extracted,
  these objects are placed into the current default `Graph`. See
  `tf.Graph.as_graph_def` for a way to create a `GraphDef`
  proto.
  Args:
    graph_def: A `GraphDef` proto containing operations to be imported into the
      default graph.
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    return_elements: A list of strings containing operation names in `graph_def`
      that will be returned as `Operation` objects; and/or tensor names in
      `graph_def` that will be returned as `Tensor` objects.
    validate_colocation_constraints: Whether to validate colocation constraints.
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Note that this does not apply to imported function names.
      Defaults to `"import"`.
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided,
      unrecognized attrs for ops in `graph_def` that have their default value
      according to `producer_op_list` will be removed. This will allow some more
      `GraphDef`s produced by later binaries to be accepted by earlier binaries.
  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`,
    and None if `returns_elements` is None.
  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
  graph_def = _ProcessGraphDefParam(graph_def)
  input_map = _ProcessInputMapParam(input_map)
  return_elements = _ProcessReturnElementsParam(return_elements)
  if producer_op_list is not None:
    _RemoveDefaultAttrs(producer_op_list, graph_def)
  graph = ops.get_default_graph()
  with ops.name_scope(name, 'import', input_map.values()) as scope:
    if scope:
      assert scope.endswith('/')
      prefix = scope[:-1]
    else:
      prefix = ''
    input_map = _ConvertInputMapValues(name, input_map)
  scoped_options = c_api_util.ScopedTFImportGraphDefOptions()
  options = scoped_options.options
  _PopulateTFImportGraphDefOptions(options, prefix, input_map, return_elements,
                                   validate_colocation_constraints)
    with c_api_util.tf_buffer(graph_def.SerializeToString()) as serialized:
      try:
        results = c_api.TF_GraphImportGraphDefWithResults(
        results = c_api_util.ScopedTFImportGraphDefResults(results)
      except errors.InvalidArgumentError as e:
        raise ValueError(str(e))
    _ProcessNewOps(graph)
  if graph_def.library and graph_def.library.function:
    functions = function.from_library(graph_def.library)
    for f in functions:
      f.add_to_graph(graph)
  missing_unused_input_keys = (
      c_api.TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(
          results.results))
  if missing_unused_input_keys:
    missing_unused_input_keys = [
        compat.as_str(s) for s in missing_unused_input_keys
    ]
    missing_keys = ', '.join(missing_unused_input_keys)
    raise ValueError(
        'Attempted to map inputs that were not found in graph_def: '
        f'[{missing_keys}]')
  if return_elements is None:
    return None
  else:
    return _GatherReturnElements(return_elements, graph, results.results)
