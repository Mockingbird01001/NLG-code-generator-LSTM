
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
def auto_shard_dataset(dataset, num_shards, index, num_replicas_in_sync=None):
  if (dataset.options().experimental_distribute.auto_shard_policy !=
      AutoShardPolicy.OFF):
    if num_replicas_in_sync is None:
      num_replicas_in_sync = 1
    if isinstance(dataset, dataset_ops.DatasetV1):
      return distribute._AutoShardDatasetV1(dataset, num_shards, index,
                                            num_replicas_in_sync)
    else:
      return distribute._AutoShardDataset(dataset, num_shards, index,
                                          num_replicas_in_sync)
  else:
    return dataset
def _clone_dataset(dataset):
  variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(dataset)
  remap_dict = _clone_helper(dataset._variant_tensor.op, variant_tensor_ops)
  new_variant_tensor = remap_dict[dataset._variant_tensor.op].outputs[0]
  return dataset_ops._VariantDataset(new_variant_tensor, dataset.element_spec)
def _get_op_def(op):
  return op.op_def or op_def_registry.get(op.type)
def _clone_helper(op_to_clone, variant_tensor_ops):
  remap_dict = {}
  for input_tensor in op_to_clone.inputs:
    input_tensor_op = input_tensor.op
    if input_tensor_op in variant_tensor_ops:
      recursive_map = _clone_helper(input_tensor_op, variant_tensor_ops)
      remap_dict.update(recursive_map)
  inputs_list = []
  for input_tensor in op_to_clone.inputs:
    input_tensor_op = input_tensor.op
    if input_tensor_op in remap_dict:
      remapped_input = remap_dict[input_tensor_op].outputs[0]
      inputs_list.append(remapped_input)
    else:
      inputs_list.append(input_tensor_op.outputs[input_tensor.value_index])
  g = ops.get_default_graph()
  new_op = g.create_op(
      op_to_clone.type,
      inputs_list, [o.dtype for o in op_to_clone.outputs],
      name=op_to_clone.name,
      attrs=op_to_clone.node_def.attr,
      op_def=_get_op_def(op_to_clone))
  remap_dict[op_to_clone] = new_op
  return remap_dict
