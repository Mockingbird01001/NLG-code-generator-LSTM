
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
ops.NotDifferentiable("TensorArray")
ops.NotDifferentiable("TensorArrayGrad")
ops.NotDifferentiable("TensorArraySize")
ops.NotDifferentiable("TensorArrayClose")
ops.NotDifferentiable("TensorArrayV2")
ops.NotDifferentiable("TensorArrayGradV2")
ops.NotDifferentiable("TensorArraySizeV2")
ops.NotDifferentiable("TensorArrayCloseV2")
ops.NotDifferentiable("TensorArrayV3")
ops.NotDifferentiable("TensorArrayGradV3")
ops.NotDifferentiable("TensorArrayGradWithShape")
ops.NotDifferentiable("TensorArraySizeV3")
ops.NotDifferentiable("TensorArrayCloseV3")
def _GetGradSource(op_or_tensor):
  name_tokens = op_or_tensor.name.split("/")
  grad_pos = [i for i, x in enumerate(name_tokens) if x.startswith("gradients")]
  if not grad_pos:
    raise ValueError(
        "Expected op/tensor name to start with gradients (excluding scope)"
        f", got: {op_or_tensor.name}. This means that a tf.gradients op with "
        "this op in its dependency path has a custom name that does not start "
        "with 'gradients'. Please make sure all calls to tf.gradients that "
        "have non-empty `name` arguments use names that start with "
        "'gradients'.")
  return "/".join(name_tokens[:grad_pos[-1] + 1])
@ops.RegisterGradient("TensorArrayRead")
@ops.RegisterGradient("TensorArrayReadV2")
@ops.RegisterGradient("TensorArrayReadV3")
def _TensorArrayReadGrad(op, grad):
  handle = op.inputs[0]
  index = op.inputs[1]
  flow = op.inputs[2]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  w_g = g.write(index, grad)
  return [None, None, w_g.flow]
@ops.RegisterGradient("TensorArrayWrite")
@ops.RegisterGradient("TensorArrayWriteV2")
@ops.RegisterGradient("TensorArrayWriteV3")
def _TensorArrayWriteGrad(op, flow):
  handle = op.inputs[0]
  index = op.inputs[1]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  flow_out = array_ops.identity(op.outputs[0], "flow_out")
  with ops.control_dependencies([flow_out]):
    flow = array_ops.identity(flow, "write_barrier")
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  grad = g.read(index)
  return [None, None, grad, flow]
@ops.RegisterGradient("TensorArrayGather")
@ops.RegisterGradient("TensorArrayGatherV2")
@ops.RegisterGradient("TensorArrayGatherV3")
def _TensorArrayGatherGrad(op, grad):
  handle = op.inputs[0]
  indices = op.inputs[1]
  flow = op.inputs[2]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  u_g = g.scatter(indices, grad)
  return [None, None, u_g.flow]
@ops.RegisterGradient("TensorArrayScatter")
@ops.RegisterGradient("TensorArrayScatterV2")
@ops.RegisterGradient("TensorArrayScatterV3")
def _TensorArrayScatterGrad(op, flow):
  handle = op.inputs[0]
  indices = op.inputs[1]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  flow_out = array_ops.identity(op.outputs[0], "flow_out")
  with ops.control_dependencies([flow_out]):
    flow = array_ops.identity(flow, "write_barrier")
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  grad = g.gather(indices)
  return [None, None, grad, flow]
@ops.RegisterGradient("TensorArrayConcat")
@ops.RegisterGradient("TensorArrayConcatV2")
@ops.RegisterGradient("TensorArrayConcatV3")
def _TensorArrayConcatGrad(op, grad, unused_lengths_grad):
  handle = op.inputs[0]
  flow = op.inputs[1]
  lengths = op.outputs[1]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  u_g = g.split(grad, lengths=lengths)
  return [None, u_g.flow]
@ops.RegisterGradient("TensorArraySplit")
@ops.RegisterGradient("TensorArraySplitV2")
@ops.RegisterGradient("TensorArraySplitV3")
def _TensorArraySplitGrad(op, flow):
  handle = op.inputs[0]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  flow_out = array_ops.identity(op.outputs[0], "flow_out")
  with ops.control_dependencies([flow_out]):
    flow = array_ops.identity(flow, "write_barrier")
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  grad = g.concat()
  return [None, grad, None, flow]
