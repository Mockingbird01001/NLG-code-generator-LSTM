
from tensorflow.python.framework import dtypes
OP_TYPES_ALLOWLIST = ["DummyIterationCounter"]
TENSOR_TYPES_ALLOWLIST = [dtypes.variant]
def _traverse(dataset, op_filter_fn):
  result = []
  bfs_q = Queue.Queue()
  visited = []
  while not bfs_q.empty():
    op = bfs_q.get()
    visited.append(op)
    if op_filter_fn(op):
      result.append(op)
    for i in op.inputs:
      input_op = i.op
      if input_op not in visited:
        bfs_q.put(input_op)
  return result
def obtain_capture_by_value_ops(dataset):
  def capture_by_value(op):
    return (op.outputs[0].dtype in TENSOR_TYPES_ALLOWLIST or
            op.type in OP_TYPES_ALLOWLIST)
  return _traverse(dataset, capture_by_value)
def obtain_all_variant_tensor_ops(dataset):
  return _traverse(dataset, lambda op: op.outputs[0].dtype == dtypes.variant)
