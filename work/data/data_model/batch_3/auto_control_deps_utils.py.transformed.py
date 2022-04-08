
from tensorflow.python.framework import dtypes
from tensorflow.python.util import object_identity
READ_ONLY_RESOURCE_INPUTS_ATTR = "_read_only_resource_inputs"
RESOURCE_READ_OPS = set()
COLLECTIVE_MANAGER_IDS = "_collective_manager_ids"
def register_read_only_resource_op(op_type):
  RESOURCE_READ_OPS.add(op_type)
def get_read_only_resource_input_indices_graph(func_graph):
  result = []
  op_read_only_resource_inputs = {}
  for input_index, t in enumerate(func_graph.inputs):
    if t.dtype != dtypes.resource:
      continue
    read_only = True
    for op in t.consumers():
      if op in op_read_only_resource_inputs:
        if t not in op_read_only_resource_inputs[op]:
          read_only = False
          break
      else:
        indices = _get_read_only_resource_input_indices_op(op)
        op_read_only_resource_inputs[op] = object_identity.ObjectIdentitySet(
            [op.inputs[i] for i in indices])
        if t not in op_read_only_resource_inputs[op]:
          read_only = False
          break
    if read_only:
      result.append(input_index)
  return result
def _get_read_only_resource_input_indices_op(op):
  if op.type in RESOURCE_READ_OPS:
    return [i for i, t in enumerate(op.inputs) if t.dtype == dtypes.resource]
  try:
    read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
  except ValueError:
    return []
  read_only_index = 0
  result = []
  for i, t in enumerate(op.inputs):
    if read_only_index >= len(read_only_input_indices):
      break
    if op.inputs[i].dtype != dtypes.resource:
      continue
    if (read_only_index < len(read_only_input_indices) and
        i == read_only_input_indices[read_only_index]):
      result.append(i)
      read_only_index += 1
  return result
def get_read_write_resource_inputs(op):
  reads = object_identity.ObjectIdentitySet()
  writes = object_identity.ObjectIdentitySet()
  if op.type in RESOURCE_READ_OPS:
    reads.update(t for t in op.inputs if t.dtype == dtypes.resource)
    return (reads, writes)
  try:
    read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
  except ValueError:
    writes.update(t for t in op.inputs if t.dtype == dtypes.resource)
    return (reads, writes)
  read_only_index = 0
  for i, t in enumerate(op.inputs):
    if op.inputs[i].dtype != dtypes.resource:
      continue
    if (read_only_index < len(read_only_input_indices) and
        i == read_only_input_indices[read_only_index]):
      reads.add(op.inputs[i])
      read_only_index += 1
    else:
      writes.add(op.inputs[i])
  return (reads, writes)
def _op_writes_to_resource(handle, op):
  if op.type in RESOURCE_READ_OPS:
    return False
  input_index = _input_index(op, handle)
  try:
    read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
  except ValueError:
    return True
  return input_index not in read_only_input_indices
def _input_index(op, handle):
  for i, t in enumerate(op.inputs):
    if handle is t:
      return i
  raise ValueError(f"{handle!s} not in list of inputs for op: {op!r}")
