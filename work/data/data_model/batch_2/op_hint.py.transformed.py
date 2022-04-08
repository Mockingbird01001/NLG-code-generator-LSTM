
"""Define tflite op hints (intrinsic operations).
This essentially allows defining a TensorFlow API for tflite operations in
Python with hints on how they are represented in TensorFlow Lite. This basically
is a form of tflite intrinsic. It wraps a subpart of a TensorFlow execution
graph and is useful for LSTMs and other complicated TensorFlow constructions
that are difficult to pattern match in TOCO, but are represented by a single
accelerated tflite op.
Example:
  def tflite_cool_activation(input):
    custom = tf.lite.OpHint("cool_activation")
    input, = custom.add_inputs(input)
    output = tf.sigmoid(input) * input
    output, = custom.add_outputs(output)
    return output
  image = tf.compat.v1.placeholder(tf.float32, (1, 16, 16, 1))
  output = tf.identity(tflite_cool_activation(image))
  session = tf.compat.v1.Session()
  graphdef_to_convert = tf.lite.experimental.convert_op_hints_to_stubs(session)
  tflite_graph = tf.compat.v1.lite.toco_convert(
      graphdef_to_convert, [image], [output], allow_custom_ops=True)
  with open("/tmp/graph.fb", "wb") as fp:
    fp.write(tflite_graph)
How does it work?:
OpHint is a helper that you use when defining a vanilla python function.
It allows you to wrap arguments with tf.identities with some custom attributes.
These attributes allow you to find the original block of ops that was created.
For example, if you use cool_activation above you essentially get:
a_input = tf.identity()
result = tf.multiply(tf.sigmoid(a_input), a_input)
output = tf.identity()
a_input, output are identities that have parameters representing
what argument they are, what the name of the function they should turn into
in tf lite as well as a guid that uniquely identifies a particular invocation.
Once you have built your whole tensorflow graph, you can run it and train it
as usual, but after you have done that, you need to convert the graph into
a form that replaces these subgraphs wrapped in identities to stub ops. These
ops don't actually exist in the normal TensorFlow runtime, but will be
understood by toco later. The generated TensorFlow Lite flatbuffer file will
contain a custom operator called "cool_activation". Developer needs to implement
and register this operator in TensorFlow Lite in order to do inference.
"""
import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
import six as _six
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
@_tf_export(v1=["lite.OpHint"])
@_deprecation.deprecated(
    None,
    "Please follow instructions under "
    "https://www.tensorflow.org/lite/convert/operation_fusion for operation"
    "fusion in tflite."
)
class OpHint(object):
  FUNCTION_NAME_ATTR = "_tflite_function_name"
  FUNCTION_UUID_ATTR = "_tflite_function_uuid"
  FUNCTION_INPUT_INDEX_ATTR = "_tflite_function_input_index"
  FUNCTION_OUTPUT_INDEX_ATTR = "_tflite_function_output_index"
  FUNCTION_SORT_INDEX_ATTR = "_tflite_function_sort_index"
  FUNCTION_AGGREGATE_ATTR = "_tflite_function_aggregate"
  TFLITE_INPUT_INDICES = "_tflite_input_indices"
  FUNCTION_LEVEL_ATTR = "_tflite_ophint_level"
  CHILDREN_INPUTS_MAPPINGS = "_tflite_children_ophint_inputs_mapping"
  AGGREGATE_STACK = "stack"
  AGGREGATE_FIRST = "first"
  AGGREGATE_LAST = "last"
  class OpHintArgumentTracker(object):
    def __init__(self,
                 function_name,
                 unique_function_id,
                 node_name_prefix,
                 attr_name,
                 level=1,
                 children_inputs_mappings=None):
      self._function_name = function_name
      self._unique_function_id = unique_function_id
      self._used_global_indices = set()
      self._node_name_prefix = node_name_prefix
      self._attr_name = attr_name
      self._level = level
      self._children_inputs_mappings = children_inputs_mappings
    def _get_new_global_index(self, index_override):
      if index_override is None:
        global_index = self._next_global_index
      else:
        if index_override in self._used_global_indices:
          raise ValueError("Index %d was already used by another call to add")
        global_index = index_override
      self._used_global_indices.add(global_index)
      while self._next_global_index in self._used_global_indices:
        self._next_global_index += 1
      return global_index
    def add(self, arg, tag=None, name=None, aggregate=None,
            index_override=None):
      """Return a wrapped tensor of an input tensor as an argument.
      Args:
        arg: A TensorFlow tensor that should be considered an argument.
        tag: String tag to identify arguments that should be packed.
        name: Name of argument. This is included in the Identity hint op names.
        aggregate: Strategy to aggregate.
        Acceptable values are OpHint.AGGREGATE_FIRST, OpHint.AGGREGATE_LAST,
          and OpHint.AGGREGATE_STACK.
          Note, aggregate is only valid if tag is specified.
        index_override: Specify what input/output index should this be in the
          final stub. i.e. add(arg0, index=1); add(arg1, index=0) will make the
          final stub be as stub_func(inputs[arg1, arg0], outputs=[]) rather than
          the default call order based ordering.
      Returns:
        A tensor representing the wrapped argument.
      Raises:
        ValueError: When indices are not consistent.
      """
      if tag is None:
        if aggregate is not None:
          raise ValueError("You must specify `tag` if using aggregate.")
        global_index = self._get_new_global_index(index_override)
        sort_index = None
      else:
        if aggregate is None:
          raise ValueError("You must specify `aggregate` if using tag.")
        if tag not in self._tag_to_global_index:
          self._tag_to_global_index[tag] = (
              self._get_new_global_index(index_override))
          self._tag_to_next_sort_index[tag] = 0
        elif (index_override and
              index_override != self._tag_to_global_index[tag]):
          raise ValueError(
              "Tag %r was called with two indices %r and %r" %
              (tag, index_override, self._tag_to_global_index[tag]))
        global_index = self._tag_to_global_index[tag]
        sort_index = self._tag_to_next_sort_index[tag]
        self._tag_to_next_sort_index[tag] += 1
      uuid = self._unique_function_id
      name = "%s-%s-%s-%r-%r-%s" % (self._node_name_prefix, self._function_name,
                                    uuid, global_index, sort_index, name)
      identity_op = _array_ops.identity(arg, name=name)
      identity_op.op._set_attr(
          OpHint.FUNCTION_NAME_ATTR,
          _attr_value_pb2.AttrValue(
              s=_compat.as_bytes(self._function_name)))
      identity_op.op._set_attr(
          OpHint.FUNCTION_UUID_ATTR,
          _attr_value_pb2.AttrValue(
              s=_compat.as_bytes(self._unique_function_id)))
      identity_op.op._set_attr(
          self._attr_name, _attr_value_pb2.AttrValue(i=global_index))
      identity_op.op._set_attr(OpHint.FUNCTION_LEVEL_ATTR,
                               _attr_value_pb2.AttrValue(i=self._level))
      if self._children_inputs_mappings:
        identity_op.op._set_attr(
            OpHint.CHILDREN_INPUTS_MAPPINGS,
            _attr_value_pb2.AttrValue(
                s=_compat.as_bytes(_json.dumps(
                    self._children_inputs_mappings))))
      if sort_index is not None:
        identity_op.op._set_attr(
            OpHint.FUNCTION_SORT_INDEX_ATTR,
            _attr_value_pb2.AttrValue(i=sort_index))
      if aggregate is not None:
        identity_op.op._set_attr(
            OpHint.FUNCTION_AGGREGATE_ATTR,
            _attr_value_pb2.AttrValue(s=_compat.as_bytes((aggregate))))
      return identity_op
  def __init__(self,
               function_name,
               level=1,
               children_inputs_mappings=None,
               **kwargs):
    """Create a OpHint.
    Args:
      function_name: Name of the function (the custom op name in tflite)
      level: OpHint level.
      children_inputs_mappings: Children OpHint inputs/outputs mapping.
        children_inputs_mappings should like below:
        "parent_first_child_input":
            [{"parent_input_index": num, "child_input_index": num}, ...]
        "parent_last_child_output":
            [{"parent_output_index": num, "child_output_index": num}, ...]
        "internal_children_input_output":
            [{"child_input_index": num, "child_output_index": num}, ...]
      **kwargs: Keyword arguments of any constant attributes for the function.
    """
    self._function_name = function_name
    self._level = level
    if self._level == 1:
      assert children_inputs_mappings is None
    else:
      assert isinstance(children_inputs_mappings, dict)
    self._children_inputs_mappings = children_inputs_mappings
    if self._children_inputs_mappings is not None:
      self._validate_children_inputs_mappings(self._children_inputs_mappings)
    self._unique_function_id = _uuid.uuid1().hex
    self._attrs_to_store_later = kwargs
    self._stored_attrs = False
    self._inputs = OpHint.OpHintArgumentTracker(
        self._function_name, self._unique_function_id, "InputHint",
        OpHint.FUNCTION_INPUT_INDEX_ATTR, level, self._children_inputs_mappings)
    self._outputs = OpHint.OpHintArgumentTracker(
        self._function_name, self._unique_function_id, "OutputHint",
        OpHint.FUNCTION_OUTPUT_INDEX_ATTR, level,
        self._children_inputs_mappings)
  def _validate_children_inputs_mappings(self, children_inputs_mappings):
    assert isinstance(children_inputs_mappings, dict)
    assert "parent_first_child_input" in children_inputs_mappings
    assert "parent_last_child_output" in children_inputs_mappings
    assert "internal_children_input_output" in children_inputs_mappings
    def assert_dictlist_has_keys(dictlist, keys):
      for dikt in dictlist:
        assert isinstance(dikt, dict)
        for key in keys:
          assert key in dikt
    assert_dictlist_has_keys(
        children_inputs_mappings["parent_first_child_input"],
        ["parent_ophint_input_index", "first_child_ophint_input_index"])
    assert_dictlist_has_keys(
        children_inputs_mappings["parent_last_child_output"],
        ["parent_output_index", "child_output_index"])
    assert_dictlist_has_keys(
        children_inputs_mappings["internal_children_input_output"],
        ["child_input_index", "child_output_index"])
  def _setattr(self, dest_op, name, value):
    tensor_value = _ops.convert_to_tensor(value)
    dest_op.op._set_attr(name, _attr_value_pb2.AttrValue(
        tensor=tensor_value.op.node_def.attr["value"].tensor))
  def add_input(self, *args, **kwargs):
    return self._inputs.add(*args, **kwargs)
  def add_output(self, *args, **kwargs):
    return self._outputs.add(*args, **kwargs)
  def add_inputs(self, *args, **kwargs):
    """Add a sequence of inputs to the function invocation.
    Args:
      *args: List of inputs to be converted (should be Tf.Tensor).
      **kwargs: This allows 'names' which should be a list of names.
    Returns:
      Wrapped inputs (identity standins that have additional metadata). These
      are also are also tf.Tensor's.
    """
    if "names" in kwargs:
      return [
          self._inputs.add(arg, name=name)
          for arg, name in zip(args, kwargs["names"])
      ]
    else:
      return [self._inputs.add(arg) for arg in args]
  def add_outputs(self, *args, **kwargs):
    """Add a sequence of outputs to the function invocation.
    Args:
      *args: List of outputs to be converted (should be tf.Tensor).
      **kwargs: See
    Returns:
      Wrapped outputs (identity standins that have additional metadata). These
      are also tf.Tensor's.
    """
    if "names" in kwargs:
      return [
          self._outputs.add(arg, name=name)
          for arg, name in zip(args, kwargs["names"])
      ]
    else:
      return [self._outputs.add(arg) for arg in args]
class _LiteOperand(object):
  def aggregate_and_return_name_for_input(self, out_graphdef):
    """This adds the node(s) to out_graphdef and returns the input node name.
    Args:
      out_graphdef: A graphdef that is ready to have this input added.
    Returns:
      The output that the stub should use as an input for this operand.
    Raises:
      RuntimeError: if the method is not implemented.
    """
    del out_graphdef
    raise RuntimeError("Unimplemented abstract method.")
  def aggregate_and_return_name_for_output(self, fused_op_name, output_index,
                                           out_graphdef):
    """Add node(s) to graph representing output operands and returns type.
    Args:
      fused_op_name: name of the fused op stub name.
      output_index: Output index that we are currently processing from stub.
      out_graphdef: The destination graphdef we are currently building up.
    Returns:
      The datatype of this identity.
    Raises:
      RuntimeError: if the method is not implemented.
    """
    del fused_op_name, output_index, out_graphdef
    raise RuntimeError("Unimplemented abstract method.")
class _LiteSingleOperand(_LiteOperand):
  def __init__(self, node):
    _LiteOperand.__init__(self)
    self.node = node
    self.name = _tensor_name_base(node.name)
  def flatten(self):
    return [self.name]
  def aggregate_and_return_name_for_input(self, out_graphdef):
    return self.name
  def aggregate_and_return_name_for_output(self, fused_op_name, index,
                                           out_graphdef):
    output_node = _copy.deepcopy(self.node)
    del output_node.input[:]
    output_node.input.append(_tensorflow_output_name(fused_op_name, index))
    out_graphdef.node.extend([output_node])
    return self.node.attr["type"].i
  def __str__(self):
    return str(self.name)
class _LiteAggregateOperand(_LiteOperand):
  def __init__(self, aggregation):
    _LiteOperand.__init__(self)
    self.aggregation = aggregation
    self.names = {}
    self.nodes = {}
    self.flattened = None
  def add(self, sort, node):
    self.names[sort] = _tensor_name_base(node.name)
    self.nodes[sort] = node
  def flatten_nodes(self):
    if not self.flattened:
      self.flattened = [None] * len(self.nodes)
      for idx, node in _six.iteritems(self.nodes):
        self.flattened[idx] = node
      for n in self.nodes:
        if n is None:
          raise RuntimeError("Aggregate was missing argument.")
      if self.aggregation == OpHint.AGGREGATE_FIRST:
        self.flattened = self.flattened[:1]
      elif self.aggregation == OpHint.AGGREGATE_LAST:
        self.flattened = self.flattened[-1:]
      elif self.aggregation == OpHint.AGGREGATE_STACK:
        pass
      else:
        raise ValueError("Invalid aggregation type %r specified" %
                         self.aggregation)
    return self.flattened
  def flatten(self):
    return [_tensor_name_base(x.name) for x in self.flatten_nodes()]
  def aggregate_and_return_name_for_input(self, out_graphdef):
    flattened = self.flatten_nodes()
    if (self.aggregation == OpHint.AGGREGATE_FIRST) or (
        self.aggregation == OpHint.AGGREGATE_LAST):
      assert len(flattened) == 1
    if len(flattened) == 1 and self.aggregation != OpHint.AGGREGATE_STACK:
      return _tensor_name_base(flattened[0].name)
    else:
      new_node = _node_def_pb2.NodeDef()
      new_node.op = "Pack"
      new_node.name = "OpHintStack-%s" % flattened[0].name
      new_node.attr["N"].i = len(flattened)
      new_node.attr["T"].type = flattened[0].attr["T"].type
      for discrete in flattened:
        new_node.input.append(_tensor_name_base(discrete.name))
      out_graphdef.node.extend([new_node])
      return new_node.name
  def aggregate_and_return_name_for_output(self, fused_op_name, output_index,
                                           out_graphdef):
    """This adds to `out_graphdef` all the unaggregated outputs.
    I.e. we are outputting from a fused stub, but we need to make it compatible
    with the unfused original graph so we insert an unpack. Ideally in a later
    stage the unpack -> pack sequences will be removed.
    Args:
      fused_op_name: The name of the stub we are in the process of fusing.
      output_index: The output output_index this object represents.
      out_graphdef: The graphdef we are in the process of buildings
    Returns:
      The type of the aggregated output (so we can finish building the stub
      op).
    """
    flattened = self.flatten_nodes()
    if (self.aggregation == OpHint.AGGREGATE_FIRST) or (
        self.aggregation == OpHint.AGGREGATE_LAST):
      assert len(flattened) == 1
    if len(flattened) == 1 and self.aggregation != OpHint.AGGREGATE_STACK:
      temp_op = _LiteSingleOperand(flattened[0])
      return temp_op.aggregate_and_return_name_for_output(
          fused_op_name, output_index, out_graphdef)
    else:
      stack_node = _node_def_pb2.NodeDef()
      stack_node.op = "Unpack"
      stack_node.name = "OpHintUnstack-%s" % flattened[0].name
      stack_node.attr["num"].i = len(flattened)
      output_type = flattened[0].attr["T"].type
      stack_node.attr["T"].type = output_type
      stack_node.input.append(
          _tensorflow_output_name(fused_op_name, output_index))
      out_graphdef.node.extend([stack_node])
      for idx, discrete in enumerate(flattened):
        output_node = _copy.deepcopy(discrete)
        del output_node.input[:]
        output_node.input.append(_tensorflow_output_name(stack_node.name, idx))
        out_graphdef.node.extend([output_node])
      return output_type
  def __str__(self):
    s = "\t\t\tAGGREGATE %s\n" % self.aggregation
    for sort, val in self.names.iteritems():
      s += "\t\t\t%d: %s\n" % (sort, val)
    return s
class _LiteFuncCall(object):
  """Represent a TensorFlow Lite custom function.
  This is uses to accumulate found hints in the graphdef into a single
  conceptual unit.
  Attributes:
    function_name: the tflite custom op name to use
    uuid: a unique call id for this particular call  (i.e. multiple function
      calls would have the same function_name but different uuids.
    params: A param name to key value for op constant data. I.e. for axis on a
      reduction, strides on a convolution, etc.
    level: Level of the OpHint.
    children_inputs_mappings: If the Ophint has children, children inputs
      mappings indicate how their inputs & outputs are mapped.
  """
  def __init__(self):
    self.inputs = {}
    self.outputs = {}
    self.function_name = None
    self.uuid = None
    self.params = {}
    self.level = -1
    self.children_inputs_mappings = {}
  def flattened_inputs_and_outputs(self):
    """Return a list of inputs and outputs in a flattened format.
    Returns:
      Tuple of (inputs, outputs). where input and output i a list of names.
    """
    def _flatten(input_or_output_dict):
      flattened_items = []
      for item in input_or_output_dict.values():
        flattened_items.extend(item.flatten())
      return flattened_items
    return _flatten(self.inputs), _flatten(self.outputs)
  def __str__(self):
    def format_args(items):
      s = ""
      for idx, item in items.iteritems():
        s += ("\t\t%d:\n" % idx) + str(item)
      return s
    inputs_str = "\tInputs\n" + format_args(self.inputs)
    outputs_str = "\tOutputs\n" + format_args(self.outputs)
    return (
        "tflite function %s call %s level %d "
        "\n\tinputs:\n\t\t%s\n\toutputs:\n\t\t%s" %
        (self.function_name, self.uuid, self.level, inputs_str, outputs_str))
def _find_all_hints_in_nodes(nodes):
  func_calls = _collections.defaultdict(_LiteFuncCall)
  for node in nodes:
    attr = node.attr
    if (OpHint.FUNCTION_UUID_ATTR not in attr or
        not attr[OpHint.FUNCTION_UUID_ATTR].s):
      continue
    uuid = attr[OpHint.FUNCTION_UUID_ATTR].s
    call_def = func_calls[uuid]
    call_def.uuid = uuid
    call_def.function_name = attr[OpHint.FUNCTION_NAME_ATTR].s
    call_def.level = attr[OpHint.FUNCTION_LEVEL_ATTR].i
    sort = (
        attr[OpHint.FUNCTION_SORT_INDEX_ATTR].i
        if OpHint.FUNCTION_SORT_INDEX_ATTR in attr else None)
    if sort == -1:
      sort = None
    aggregation = None
    if OpHint.FUNCTION_AGGREGATE_ATTR in attr:
      aggregation = _compat.as_text(attr[OpHint.FUNCTION_AGGREGATE_ATTR].s)
    if OpHint.CHILDREN_INPUTS_MAPPINGS in attr:
      call_def.children_inputs_mappings = _json.loads(
          _compat.as_text(attr[OpHint.CHILDREN_INPUTS_MAPPINGS].s))
    def put_operand(stuff, index, sort, operand, aggregation):
      if sort is None:
        stuff[index] = _LiteSingleOperand(operand)
      else:
        if index not in stuff:
          stuff[index] = _LiteAggregateOperand(aggregation)
        stuff[index].add(sort, operand)
    if OpHint.FUNCTION_INPUT_INDEX_ATTR in attr:
      put_operand(call_def.inputs, attr[OpHint.FUNCTION_INPUT_INDEX_ATTR].i,
                  sort, node, aggregation)
    if OpHint.FUNCTION_OUTPUT_INDEX_ATTR in attr:
      put_operand(call_def.outputs, attr[OpHint.FUNCTION_OUTPUT_INDEX_ATTR].i,
                  sort, node, aggregation)
    for a in attr:
      if a.startswith("_tflite_attr_"):
        call_def.params[a.replace("_tflite_attr_,", "")] = attr[a].tensor
  return func_calls
def _extract_topology_sequence_mapping(nodes):
  return dict(
      (_tensor_name_base(node.name), idx) for idx, node in enumerate(nodes))
def _find_children_hints_in_while_loop(function_def, nodes_mapping):
  new_nodes = []
  for node in function_def.node_def:
    for i, _ in enumerate(node.input):
      if node.input[i] in nodes_mapping:
        node.input[i] = nodes_mapping[node.input[i]]
    new_nodes.append(_copy.deepcopy(node))
  name_to_seq_num = _extract_topology_sequence_mapping(function_def.node_def)
  children_hints = _find_all_hints_in_nodes(new_nodes)
  children_hints_q = []
  for hint in _six.itervalues(children_hints):
    _, output_names = hint.flattened_inputs_and_outputs()
    seq = name_to_seq_num[output_names[0]]
    for output_name in output_names:
      seq = min(seq, name_to_seq_num[output_name])
    children_hints_q.append((seq, hint))
  children_hints_q.sort(key=lambda tup: tup[0])
  ordered_children_hints = [x[1] for x in children_hints_q]
  return ordered_children_hints, new_nodes
def _find_children_hints(call, graph_def):
  """Find all children hints.
  For a given OpHint, we find all children hints inside it, we also copy all the
  nodes inside function defs (if applicable) to the original graph_def, they are
  returned in a list as well.
  Args:
    call: Parent OpHint that contains children ophints.
    graph_def: Original graph def.
  Returns:
    Ordered children hints inside the parent ophint; new graph def that contains
    nodes inside function defs (if applicable); nodes inside function defs.
  """
  name_to_input_name, _, _ = _extract_graph_summary(graph_def)
  input_names, output_names = call.flattened_inputs_and_outputs()
  reachable_by_input = _bfs_for_reachable_nodes(input_names, name_to_input_name)
  reachable_by_output = _bfs_for_reachable_nodes(output_names,
                                                 name_to_input_name)
  output_nodes_set = set(output_names)
  children_hints = []
  out = _graph_pb2.GraphDef()
  out.library.CopyFrom(graph_def.library)
  out.versions.CopyFrom(graph_def.versions)
  function_def_nodes = set()
  for node in graph_def.node:
    out.node.extend([_copy.deepcopy(node)])
    n = _tensor_name_base(node.name)
    if n in reachable_by_output:
      if n not in reachable_by_input and n not in output_nodes_set:
        if node.op == "While" or node.op == "StatelessWhile":
          body_name = node.attr["body"].func.name
          inputs_outside_loop = node.input
          for function_def in graph_def.library.function:
            if function_def.signature.name == body_name:
              function_inputs = function_def.signature.input_arg
              assert len(inputs_outside_loop) == len(function_inputs)
              nodes_mapping = {}
              for i, function_input in enumerate(function_inputs):
                nodes_mapping[function_input.name] = inputs_outside_loop[i]
              (children_hints_in_loop,
               new_nodes) = _find_children_hints_in_while_loop(
                   function_def, nodes_mapping)
              function_def_nodes.update([x.name for x in new_nodes])
              children_hints.extend(children_hints_in_loop)
              out.node.extend(new_nodes)
  return children_hints, out, function_def_nodes
def _tensor_name_base(full_tensor_name):
  """Removes the device assignment code from a tensor.
  e.g. _tensor_name_base("foo:3") => "foo"
  Args:
    full_tensor_name: A tensor name that is annotated with a device placement
      (this is what tensor flow introspection gives).
  Returns:
    A name without any device assignment.
  """
  if full_tensor_name.startswith("^"):
    return full_tensor_name[1:]
  return full_tensor_name.split(":")[0]
def _tensorflow_output_name(tensor_name, output_index):
  return tensor_name if output_index == 0 else "%s:%d" % (tensor_name,
                                                          output_index)
def _check_subgraph_closed(n, reachable_by_input, input_nodes_set,
                           name_to_input_name):
  next_to_visit = [n]
  visited = set()
  while next_to_visit:
    current_node = next_to_visit.pop()
    visited.add(current_node)
    if (current_node in reachable_by_input and
        current_node not in input_nodes_set):
      raise TypeError("Node %s uses input %s not in input_nodes." %
                      (n, current_node))
    if current_node not in input_nodes_set:
      next_to_visit += [
          input_node for input_node in name_to_input_name[current_node]
          if input_node not in visited
      ]
def _convert_single_op_hint_to_stub(call,
                                    graph_def,
                                    function_def_nodes=None,
                                    is_last_run=True):
  """Given a graph_def, converts `call` into a stub and returns a new graph_def.
  Args:
    call: A single function call to be converted.
    graph_def: A graph_def to use as input (that has call obviously).
    function_def_nodes: Nodes inside the function def those are not connected to
      the graph.
    is_last_run: Whether it is the last run for a given pass (for OpHint has
      children).
  Returns:
    A new transformed graph-def that has call as a stub (single op).
  Note: after this process, the graph_def can no longer be loaded into
      the tensorflow runtime, so all future manipulations are done in graph_def
      level.
  """
  if function_def_nodes is None:
    function_def_nodes = set()
  name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(
      graph_def)
  input_names, output_names = call.flattened_inputs_and_outputs()
  reachable_by_input = _bfs_for_reachable_nodes(input_names, name_to_input_name)
  reachable_by_output = _bfs_for_reachable_nodes(output_names,
                                                 name_to_input_name)
  output_nodes_set = set(output_names)
  nodes_after_fuse = []
  nodes_deleted_by_fuse = set()
  for node in graph_def.node:
    n = _tensor_name_base(node.name)
    if n in reachable_by_output:
      if n not in reachable_by_input and n not in output_nodes_set:
        nodes_deleted_by_fuse.add(n)
    elif n not in reachable_by_input and n not in function_def_nodes:
      nodes_after_fuse.append(n)
    else:
      if not is_last_run:
        nodes_after_fuse.append(n)
  out = _graph_pb2.GraphDef()
  reachable_by_input_sorted = sorted(
      list(reachable_by_input), key=lambda n: name_to_seq_num[n])
  for node in reachable_by_input_sorted:
    out.node.extend([_copy.deepcopy(name_to_node[node])])
  sorted_input_indices = list(call.inputs.keys())
  sorted_input_indices.sort()
  sorted_output_indices = list(call.outputs.keys())
  sorted_output_indices.sort()
  new_node = _node_def_pb2.NodeDef()
  optional_input_node = _node_def_pb2.NodeDef()
  optional_input_node.name = "Const" + str(_uuid.uuid1().hex)
  optional_input_node.op = "Const"
  optional_input_node.attr["dtype"].CopyFrom(
      _attr_value_pb2.AttrValue(type=_dtypes.float32.as_datatype_enum))
  optional_input_node.attr["value"].CopyFrom(
      _attr_value_pb2.AttrValue(
          tensor=_tensor_util.make_tensor_proto([-1], _dtypes.float32, [1])))
  out.node.extend([optional_input_node])
  max_index = max(sorted_input_indices) + 1
  for cur_index in range(max_index):
    if cur_index in sorted_input_indices:
      inputs = call.inputs[cur_index]
      input_name = inputs.aggregate_and_return_name_for_input(out)
      new_node.input.append(input_name)
    else:
      new_node.input.append(optional_input_node.name)
  new_node.attr[OpHint.TFLITE_INPUT_INDICES].list.i.extend(sorted_input_indices)
  new_node.op = call.function_name
  new_node.name = call.uuid
  out.node.extend([new_node])
  output_dtypes = []
  max_output_index = max(sorted_output_indices) + 1
  for cur_index in range(max_output_index):
    if cur_index in sorted_output_indices:
      output = call.outputs[cur_index]
      output_dtype = (
          output.aggregate_and_return_name_for_output(new_node.name, cur_index,
                                                      out))
    else:
      output_dtype = optional_input_node.attr["type"].i
    output_dtypes.append(output_dtype)
  new_node.attr["_output_types"].list.type[:] = output_dtypes
  new_node.attr["_output_quantized"].b = False
  for n in nodes_after_fuse:
    should_keep = True
    for input_name in name_to_input_name[n]:
      if input_name in nodes_deleted_by_fuse:
        should_keep = False
    if should_keep:
      out.node.extend([_copy.deepcopy(name_to_node[n])])
  out.library.CopyFrom(graph_def.library)
  out.versions.CopyFrom(graph_def.versions)
  return out
def _remove_one_redundant_stack_unstack(in_graph_def):
  """Removes a stack->unstack pattern from in_graph_def in a returned graph.
  Args:
    in_graph_def: Graph def to use as input.
  Returns:
    Simplified tuple (graph_def, changed_something) where changed_something
    is true if anything was done.
  """
  name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(
      in_graph_def)
  del name_to_seq_num
  do_generic_pack_unpack = True
  out = _graph_pb2.GraphDef()
  out.library.CopyFrom(in_graph_def.library)
  out.versions.CopyFrom(in_graph_def.versions)
  for n in in_graph_def.node:
    node_name = _tensor_name_base(n.name)
    if not node_name.startswith("OpHintStack") and not n.op.startswith("Pack"):
      continue
    next_to_visit = [node_name]
    visited = set()
    unpack_nodes = set()
    pack_node = node_name
    matches_pattern = True
    is_hint_created_stack = False
    while next_to_visit:
      current_node_name = next_to_visit[0]
      visited.add(current_node_name)
      del next_to_visit[0]
      node = name_to_node[current_node_name]
      is_op_hint_stack = node.name.startswith("OpHintStack")
      is_op_hint_unstack = node.name.startswith("OpHintUnstack")
      if (node.op == "Identity" or is_op_hint_stack or
          (do_generic_pack_unpack and node.op == "Pack")):
        is_hint_created_stack |= is_op_hint_stack
        next_to_visit += [
            input_node for input_node in name_to_input_name[current_node_name]
            if input_node not in visited
        ]
      elif (is_op_hint_unstack or
            (do_generic_pack_unpack and node.op == "Unpack")):
        unpack_nodes.add(node.name)
        is_hint_created_stack &= is_op_hint_unstack
      else:
        matches_pattern = False
        break
      visited.add(node.name)
    if matches_pattern and len(unpack_nodes) == 1:
      pack_node = node_name
      no_external_dependency = True
      for other_n in in_graph_def.node:
        if other_n.name in visited:
          continue
        for input_tensor in name_to_input_name[other_n.name]:
          input_op = _tensor_name_base(input_tensor)
          if input_op in visited and input_op != pack_node:
            no_external_dependency = False
      if is_hint_created_stack or no_external_dependency:
        end = unpack_nodes.pop()
        end_input = name_to_node[end].input[0]
        for other_n in in_graph_def.node:
          node_name = _tensor_name_base(other_n.name)
          if node_name not in visited:
            new_node = _copy.deepcopy(other_n)
            new_node.input[:] = [
                (end_input if stripped == pack_node else non_stripped)
                for stripped, non_stripped in zip(name_to_input_name[node_name],
                                                  new_node.input[:])
            ]
            out.node.extend([new_node])
        return out, True
  return in_graph_def, False
def _remove_redundant_stack_unstack(graph_def):
  curr = graph_def
  del graph_def
  changed_stuff = True
  while changed_stuff:
    curr, changed_stuff = _remove_one_redundant_stack_unstack(curr)
  return curr
def _get_correct_mapping(original_index, nodes):
  if original_index == -1:
    node_indices = nodes.keys()
    node_indices = sorted(node_indices)
    return node_indices[-1]
  return original_index
def _convert_op_hints_to_stubs_helper(
    graph_def, write_callback=lambda sess, graph_def: None):
  """Converts a graph_def to a new graph_def where all op hints are stubbed.
  Args:
    graph_def: A graph def that we should convert.
    write_callback: A function pointer that can be used to write intermediate
      steps of graph transformation (optional).
  Returns:
    A new stubbed graph_def.
  """
  hints = _find_all_hints_in_nodes(graph_def.node)
  hints_q = []
  for hint in _six.itervalues(hints):
    hints_q.append((hint.level, hint.uuid))
  hints_q.sort(key=lambda tup: tup[0])
  for i in range(len(hints_q) - 1, -1, -1):
    level, hint_uuid = hints_q[i]
  curr_graph_def = graph_def
  for i in range(len(hints_q) - 1, -1, -1):
    level, hint_uuid = hints_q[i]
    if level >= 2:
      children_hints, curr_graph_def, function_def_nodes = _find_children_hints(
          hints[hint_uuid], curr_graph_def)
      children_inputs_mappings = hints[hint_uuid].children_inputs_mappings
      for j, child_hint in enumerate(children_hints):
        if j == 0:
          for mapping in children_inputs_mappings["parent_first_child_input"]:
            parent_input_index = _get_correct_mapping(
                mapping["parent_ophint_input_index"], hints[hint_uuid].inputs)
            child_input_index = _get_correct_mapping(
                mapping["first_child_ophint_input_index"], child_hint.inputs)
            child_hint.inputs[child_input_index] = hints[hint_uuid].inputs[
                parent_input_index]
        else:
          for mapping in children_inputs_mappings[
              "internal_children_input_output"]:
            input_index = _get_correct_mapping(mapping["child_input_index"],
                                               child_hint.inputs)
            output_index = _get_correct_mapping(mapping["child_output_index"],
                                                children_hints[j - 1].outputs)
            child_hint.inputs[input_index] = children_hints[
                j - 1].outputs[output_index]
        if j == len(children_hints) - 1:
          for mapping in children_inputs_mappings["parent_last_child_output"]:
            parent_output_index = _get_correct_mapping(
                mapping["parent_output_index"], hints[hint_uuid].outputs)
            child_output_index = _get_correct_mapping(
                mapping["child_output_index"], child_hint.outputs)
            child_hint.outputs[child_output_index] = hints[hint_uuid].outputs[
                parent_output_index]
      for j, child_hint in enumerate(children_hints):
        curr_graph_def = _convert_single_op_hint_to_stub(
            child_hint, curr_graph_def, function_def_nodes,
            j == len(children_hints) - 1)
    else:
      curr_graph_def = _convert_single_op_hint_to_stub(hints[hint_uuid],
                                                       curr_graph_def)
      write_callback(curr_graph_def, "initial")
  curr_graph_def = _remove_redundant_stack_unstack(curr_graph_def)
  return curr_graph_def
def find_all_hinted_output_nodes(session=None, graph_def=None):
  """Find all Ophints output nodes in the graph.
  This is used to get all the output nodes those are ophinted, it is important
  for operation like convert_variables_to_constants keep all ophints structure.
  Note: only one of session or graph_def should be used, not both.
  Why this can be useful? Some TensorFlow ops (e.g. bidirectional rnn), can
  generate multiple outputs for unfused subgraph. If not all output nodes are
  consumed, graph optimization can potentially drop the unused nodes and cause
  ophints in an invalid states (due to missing ophinted output nodes). So it's
  important for us to find all those hinted output nodes and make sure they're
  not discarded away.
  Args:
    session: A TensorFlow session that contains the graph to convert.
    graph_def: A graph def that we should convert.
  Returns:
    A list of OpHints output nodes.
  Raises:
    ValueError: If both session and graph_def are provided.
  """
  if session is not None and graph_def is not None:
    raise ValueError("Provide only one of session and graph_def.")
  hinted_outputs_nodes = []
  if session is not None:
    hints = _find_all_hints_in_nodes(session.graph_def.node)
  elif graph_def is not None:
    hints = _find_all_hints_in_nodes(graph_def.node)
  for hint in _six.itervalues(hints):
    _, output_nodes = hint.flattened_inputs_and_outputs()
    hinted_outputs_nodes.extend(output_nodes)
  return hinted_outputs_nodes
def is_ophint_converted(graph_def):
  if graph_def is None:
    raise ValueError("Must provide the graph_def.")
  ophint_converted = False
  for node in graph_def.node:
    attr = node.attr
    if OpHint.FUNCTION_INPUT_INDEX_ATTR in attr:
      ophint_converted = True
      break
  return ophint_converted
@_tf_export(v1=["lite.experimental.convert_op_hints_to_stubs"])
@_deprecation.deprecated(
    None,
    "Please follow instructions under "
    "https://www.tensorflow.org/lite/convert/operation_fusion for operation"
    "fusion in tflite."
)
def convert_op_hints_to_stubs(session=None,
                              graph_def=None,
                              write_callback=lambda graph_def, comments: None):
  """Converts a graphdef with LiteOp hints into stub operations.
  This is used to prepare for toco conversion of complex intrinsic usages.
  Note: only one of session or graph_def should be used, not both.
  Args:
    session: A TensorFlow session that contains the graph to convert.
    graph_def: A graph def that we should convert.
    write_callback: A function pointer that can be used to write intermediate
      steps of graph transformation (optional).
  Returns:
    A new graphdef with all ops contained in OpHints being replaced by
    a single op call with the right parameters.
  Raises:
    ValueError: If both session and graph_def are provided.
  """
  if session is not None and graph_def is not None:
    raise ValueError("Provide only one of session and graph_def.")
  if session is not None:
    return _convert_op_hints_to_stubs_helper(session.graph_def, write_callback)
  elif graph_def is not None:
    return _convert_op_hints_to_stubs_helper(graph_def, write_callback)
  else:
    raise ValueError("Must specify session or graph_def as input.")
_allowed_symbols = [
    "OpHint",
    "convert_op_hints_to_stubs",
    "convert_op_hints_to_stubs_new",
    "find_all_hinted_output_nodes",
    "is_ophint_converted",
]
remove_undocumented(__name__, _allowed_symbols)
