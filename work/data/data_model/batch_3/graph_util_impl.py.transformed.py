
import copy
import re
import six
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util.tf_export import tf_export
tf_export(v1=["GraphDef"])(graph_pb2.GraphDef)
convert_to_constants = lazy_loader.LazyLoader(
    "convert_to_constants", globals(),
    "tensorflow.python.framework.convert_to_constants")
_VARIABLE_OPS = {
    "Assign",
    "AssignAdd",
    "AssignSub",
    "Queue",
    "ScatterAdd",
    "ScatterSub",
    "ScatterUpdate",
    "TruncatedNormal",
    "Variable",
    "VariableV2",
}
_CONTROL_FLOW_OP_NAMES_OR_IDENTITY = [
    "Switch",
    "Enter",
    "Exit",
    "Identity",
    "Merge",
    "NextIteration",
]
def _is_variable_op(op):
  return op in _VARIABLE_OPS
graph_pb2.GraphDef.__doc__ = """\
A protobuf containing the graph of operations.
@compatibility(TF2)
This API is not available in TensorFlow 2.x.
You should not need to use `GraphDef`s directly in TF2. To load `GraphDef`s in
TF2, use SavedModel. The SavedModel contains the `GraphDef`.
Before:
```python
with tf.io.gfile.GFile('/tmp/graph.pb', 'rb') as f:
  graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(f.read())
```
After:
```python
tf.saved_model.load('/tmp/saved_model')
```
If you would like to create a `GraphDef` in TF2, use `tf.function` and
`get_concrete_function`.
>>> @tf.function
>>> def f(x):
>>>   return x
>>>
>>> graph_def = f.get_concrete_function(1.).graph.as_graph_def()
>>> print(graph_def)
@end_compatibility
"""
@deprecation.deprecated(
    date=None,
    instructions="Use `tf.compat.v1.graph_util.must_run_on_cpu`")
@tf_export(v1=["graph_util.must_run_on_cpu"])
def must_run_on_cpu(node, pin_variables_on_cpu=False):
  if isinstance(node, ops.Operation):
    node_def = node.node_def
  else:
    assert isinstance(node, node_def_pb2.NodeDef)
    node_def = node
  if pin_variables_on_cpu and _is_variable_op(node_def.op):
    return True
  if node_def.op == "Const":
    dtype = node_def.attr["dtype"].type
    if dtype == dtypes.string or dtype == dtypes.int32:
      return True
  if node_def.op in ["DynamicStitch", "ParallelDynamicStitch"]:
    dtype = node_def.attr["T"].type
    if dtype == dtypes.int32:
      return True
  if node_def.op in ["Cast"]:
    dtype = node_def.attr["SrcT"].type
    if dtype == dtypes.int32:
      return True
  return False
def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]
def _get_colocated_node_name(colocated_node_name):
  colocated_node_decoded = colocated_node_name.decode("utf-8")
  if colocated_node_decoded.startswith("loc:@"):
    return colocated_node_decoded[5:]
  return colocated_node_decoded
def _extract_graph_summary(graph_def):
  seq = 0
  for node in graph_def.node:
    n = _node_name(node.name)
    name_to_node[n] = node
    name_to_input_name[n] = [_node_name(x) for x in node.input]
    if "_class" in node.attr:
      for colocated_node_name in node.attr["_class"].list.s:
        name_to_input_name[n].append(
            _get_colocated_node_name(colocated_node_name))
    name_to_seq_num[n] = seq
    seq += 1
  return name_to_input_name, name_to_node, name_to_seq_num
def _assert_nodes_are_present(name_to_node, nodes):
  for d in nodes:
    assert d in name_to_node, "%s is not in graph" % d
def _bfs_for_reachable_nodes(target_nodes, name_to_input_name):
  nodes_to_keep = set()
  next_to_visit = list(target_nodes)
  while next_to_visit:
    node = next_to_visit[0]
    del next_to_visit[0]
    if node in nodes_to_keep:
      continue
    nodes_to_keep.add(node)
    if node in name_to_input_name:
      next_to_visit += name_to_input_name[node]
  return nodes_to_keep
@deprecation.deprecated(
    date=None,
    instructions="Use `tf.compat.v1.graph_util.extract_sub_graph`")
@tf_export(v1=["graph_util.extract_sub_graph"])
def extract_sub_graph(graph_def, dest_nodes):
  if not isinstance(graph_def, graph_pb2.GraphDef):
    raise TypeError("graph_def must be a graph_pb2.GraphDef proto, but got "
                    f"type {type(graph_def)}.")
  if isinstance(dest_nodes, six.string_types):
    raise TypeError("dest_nodes must be an iterable of strings, but got "
                    f"type {type(dest_nodes)}.")
  name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(
      graph_def)
  _assert_nodes_are_present(name_to_node, dest_nodes)
  nodes_to_keep = _bfs_for_reachable_nodes(dest_nodes, name_to_input_name)
  nodes_to_keep_list = sorted(
      list(nodes_to_keep), key=lambda n: name_to_seq_num[n])
  out = graph_pb2.GraphDef()
  for n in nodes_to_keep_list:
    out.node.extend([copy.deepcopy(name_to_node[n])])
  out.library.CopyFrom(graph_def.library)
  out.versions.CopyFrom(graph_def.versions)
  return out
@deprecation.deprecated(
    date=None,
    instructions="Use `tf.compat.v1.graph_util.tensor_shape_from_node_def_name`"
)
@tf_export(v1=["graph_util.tensor_shape_from_node_def_name"])
def tensor_shape_from_node_def_name(graph, input_name):
  if ":" not in input_name:
    canonical_name = input_name + ":0"
  else:
    canonical_name = input_name
  tensor = graph.get_tensor_by_name(canonical_name)
  shape = tensor.get_shape()
  return shape
@deprecation.deprecated(
    date=None,
    instructions="Use `tf.compat.v1.graph_util.convert_variables_to_constants`")
@tf_export(v1=["graph_util.convert_variables_to_constants"])
def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None):
  """Replaces all the variables in a graph with constants of the same values.
  If you have a trained graph containing Variable ops, it can be convenient to
  convert them all to Const ops holding the same values. This makes it possible
  to describe the network fully with a single GraphDef file, and allows the
  removal of a lot of ops related to loading and saving the variables.
  Args:
    sess: Active TensorFlow session containing the variables.
    input_graph_def: GraphDef object holding the network.
    output_node_names: List of name strings for the result nodes of the graph.
    variable_names_whitelist: The set of variable names to convert (by default,
                              all variables are converted).
    variable_names_blacklist: The set of variable names to omit converting
                              to constants.
  Returns:
    GraphDef containing a simplified version of the original.
  Raises:
    RuntimeError: if a DT_RESOURCE op is found whose ancestor Variables are both
      denylisted AND whitelisted for freezing.
  """
  ret = convert_to_constants.convert_variables_to_constants_from_session_graph(
      session=sess,
      graph_def=input_graph_def,
      output_node_names=output_node_names,
      variable_names_allowlist=variable_names_whitelist,
      variable_names_denylist=variable_names_blacklist)
  ret.versions.Clear()
  return ret
@deprecation.deprecated(
    date=None,
    instructions="Use `tf.compat.v1.graph_util.remove_training_nodes`")
@tf_export(v1=["graph_util.remove_training_nodes"])
def remove_training_nodes(input_graph, protected_nodes=None):
  if not protected_nodes:
    protected_nodes = []
  types_to_remove = {"CheckNumerics": True}
  input_nodes = input_graph.node
  names_to_remove = {}
  for node in input_nodes:
    if node.op in types_to_remove and node.name not in protected_nodes:
      names_to_remove[node.name] = True
  nodes_after_removal = []
  for node in input_nodes:
    if node.name in names_to_remove:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      if input_name in names_to_remove:
        continue
      new_node.input.append(full_input_name)
    nodes_after_removal.append(new_node)
  types_to_splice = {"Identity": True}
  control_input_names = set()
  node_names_with_control_input = set()
  for node in nodes_after_removal:
    for node_input in node.input:
      if "^" in node_input:
        control_input_names.add(node_input.replace("^", ""))
        node_names_with_control_input.add(node.name)
  names_to_splice = {}
  for node in nodes_after_removal:
    if node.op in types_to_splice and node.name not in protected_nodes:
      if node.name not in node_names_with_control_input:
        names_to_splice[node.name] = node.input[0]
  names_to_splice = {name: value for name, value in names_to_splice.items()
                     if name not in control_input_names}
  nodes_after_splicing = []
  for node in nodes_after_removal:
    if node.name in names_to_splice:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      while input_name in names_to_splice:
        full_input_name = names_to_splice[input_name]
        input_name = re.sub(r"^\^", "", full_input_name)
      new_node.input.append(full_input_name)
    nodes_after_splicing.append(new_node)
  output_graph = graph_pb2.GraphDef()
  output_graph.node.extend(nodes_after_splicing)
  return output_graph
@tf_export("__internal__.graph_util.graph_defs_equal", v1=[])
def graph_defs_equal(graph_def_1: graph_pb2.GraphDef,
                     graph_def_2: graph_pb2.GraphDef,
                     treat_nan_as_equal: bool = False) -> bool:
  """Returns True iff the graph def arguments are structurally equivalent.
  The notion of equivalence encoded here checks that the set of NodeDefs in
  the GraphDef's function library and main graph body are identical.
  Additionally, it checks that the functions in the function library are equal
  as sets.
  Example usage:
  ```
  with tf.Graph().as_default() as g1:
    tf.constant(1)
  with tf.Graph().as_default() as g2:
    tf.constant(2)
  with tf.Graph().as_default() as g3:
    tf.constant(1)
  assert tf.__internal__.graph_util.graph_defs_equal(g1.as_graph_def(),
                                                     g3.as_graph_def())
  assert not tf.__internal__.graph_util.graph_defs_equal(g1.as_graph_def(),
                                                         g2.as_graph_def())
  ```
  Args:
    graph_def_1: Instance of `graph_pb2.GraphDef` to compare.
    graph_def_2: Instance of `graph_pb2.GraphDef` to compare.
    treat_nan_as_equal: Boolean indicating whether or not to treat nan
      floating-point values as equal. This is crucial for any equivalence
      relation defined over GraphDefs, to ensure symmetry.
  Returns:
    Boolean indicating structural equivalence as described above.
  Raises:
    TypeError: If either of the GraphDefs are not instances of
      `graph_pb2.GraphDef`.
  """
  if not isinstance(graph_def_1, graph_pb2.GraphDef):
    raise TypeError("graph_def_1 must be a graph_pb2.GraphDef proto, but got "
                    f"type {type(graph_def_1)}.")
  if not isinstance(graph_def_2, graph_pb2.GraphDef):
    raise TypeError("graph_def_2 must be a graph_pb2.GraphDef proto, but got "
                    f"type {type(graph_def_2)}.")
  options = _proto_comparators.ProtoComparisonOptions(treat_nan_as_equal)
  return _proto_comparators.EqualsGraphDef(graph_def_1.SerializeToString(),
                                           graph_def_2.SerializeToString(),
                                           options)
