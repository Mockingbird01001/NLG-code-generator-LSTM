
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.util import nest
def rewrite_grad_indexed_slices(grads, body_grad_graph, loop_vars,
                                forward_inputs):
  """Handles special case of IndexedSlices returned from while gradient.
  Some gradient functions return IndexedSlices instead of a Tensor (e.g. the
  gradient of Gather ops). When this happens in the gradient of a while body,
  the resulting gradient body function will have mismatched inputs and outputs,
  since the input is a single Tensor, but the IndexedSlices gets unnested into
  three output Tensors.
  This function fixes this by rewriting the gradient body to have three inputs
  to match the three outputs, i.e., it effectively converts the input Tensor
  into an input IndexedSlices. It also returns new `loop_vars` to reflect the
  new inputs.
  Args:
    grads: the input gradient Tensors to the while gradient computation.
    body_grad_graph: _WhileBodyGradFuncGraph.
    loop_vars: list of Tensors. The inputs to body_grad_graph.
    forward_inputs: list of Tensors. The (flat) inputs to the forward-pass While
      op.
  Returns:
    The new loop_vars to pass to body_grad_graph.
  """
  inputs_with_grads = [
      t for g, t in zip(grads, forward_inputs) if g is not None
  ]
  structured_outputs = body_grad_graph.structured_outputs[3:]
  for forward_input, output in zip(inputs_with_grads, structured_outputs):
    if not isinstance(output, indexed_slices.IndexedSlices):
      continue
    if forward_input.dtype == dtypes.resource:
      loop_vars = _rewrite_input_as_indexed_slices(body_grad_graph, output,
                                                   forward_input, loop_vars)
    else:
      _rewrite_output_as_tensor(body_grad_graph, output)
  return loop_vars
def _get_tensor_index_in_iterable(iterable, t):
  for i, elem in enumerate(iterable):
    if t is elem:
      return i
  raise ValueError(f"Element `{t!r}` is not found in iterable `{iterable!r}`.")
def _rewrite_output_as_tensor(body_grad_graph, grad_output_slices):
  with body_grad_graph.as_default():
    new_output = ops.convert_to_tensor_v2(grad_output_slices)
  idx = _get_tensor_index_in_iterable(body_grad_graph.structured_outputs,
                                      grad_output_slices)
  body_grad_graph.structured_outputs[idx] = new_output
  body_grad_graph.outputs = func_graph.flatten(
      body_grad_graph.structured_outputs)
def _rewrite_input_as_indexed_slices(body_grad_graph, grad_output_slices,
                                     forward_input, loop_vars):
  init_slices = _create_grad_indexed_slices_init(grad_output_slices,
                                                 forward_input)
  with body_grad_graph.as_default():
    input_slices = indexed_slices.IndexedSlices(
        values=body_grad_graph.capture(init_slices.values, allowlisted=True),
        indices=body_grad_graph.capture(init_slices.indices, allowlisted=True),
        dense_shape=body_grad_graph.capture(
            init_slices.dense_shape, allowlisted=True))
    for t in _flatten(init_slices):
      captured_t = body_grad_graph.captures.pop(t)
      body_grad_graph.inputs.remove(captured_t)
    new_output_slices = _rewrite_grad_indexed_slices_output(
        grad_output_slices, input_slices)
  return _update_indexed_slices_param(body_grad_graph, loop_vars, init_slices,
                                      input_slices, new_output_slices,
                                      grad_output_slices)
def _create_grad_indexed_slices_init(grad_output_slices, forward_input):
  assert isinstance(grad_output_slices, indexed_slices.IndexedSlices)
  assert isinstance(forward_input, ops.Tensor)
  values_out = grad_output_slices.values
  indices_out = grad_output_slices.indices
  if values_out.shape.is_fully_defined():
    values_shape = tensor_shape.TensorShape([0] +
                                            values_out.shape.as_list()[1:])
    values = array_ops.zeros(
        values_shape, dtype=values_out.dtype, name="values_init")
  else:
    if forward_input.dtype == dtypes.resource:
      forward_shape = gen_resource_variable_ops.variable_shape(forward_input)
    else:
      forward_shape = array_ops.shape(forward_input)
    values_shape = array_ops.concat([[0], forward_shape[1:]], 0)
    values = array_ops.zeros(
        values_shape, dtype=values_out.dtype, name="values_init")
  indices = constant_op.constant([], indices_out.dtype, name="indices_init")
  if forward_input.dtype == dtypes.resource:
    shape = gen_resource_variable_ops.variable_shape(
        forward_input, name="shape_init")
  else:
    shape = array_ops.shape(forward_input, name="shape_init")
  return indexed_slices.IndexedSlices(
      values=values, indices=indices, dense_shape=shape)
def _rewrite_grad_indexed_slices_output(old_output_slices, new_input_slices):
  def rewrite(old_output, new_input):
    assert old_output.type == "Identity"
    concat_op = old_output.inputs[0].op
    assert concat_op.type == "ConcatV2"
    old_concat_args = concat_op.inputs[:-1]
    return array_ops.concat([new_input] + old_concat_args[1:], 0)
  values = rewrite(old_output_slices.values.op, new_input_slices.values)
  indices = rewrite(old_output_slices.indices.op, new_input_slices.indices)
  return indexed_slices.IndexedSlices(
      values=values, indices=indices, dense_shape=new_input_slices.dense_shape)
def _update_indexed_slices_param(graph, loop_vars, init_slices, input_slices,
                                 output_slices, old_output_slices):
  structured_idx = _get_tensor_index_in_iterable(graph.structured_outputs,
                                                 old_output_slices)
  flat_idx = _get_tensor_index_in_iterable(
      graph.outputs,
      func_graph.flatten(old_output_slices)[0])
  graph.structured_outputs[structured_idx] = output_slices
  graph.outputs = func_graph.flatten(graph.structured_outputs)
  graph.inputs = (
      graph.inputs[:flat_idx] + _flatten(input_slices) +
      graph.inputs[flat_idx + 1:])
  return loop_vars[:flat_idx] + _flatten(init_slices) + loop_vars[flat_idx + 1:]
def _flatten(arg):
  return nest.flatten(arg, expand_composites=True)
