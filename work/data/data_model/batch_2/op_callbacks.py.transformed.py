
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
def add_op_callback(callback_fn):
  r"""Add a thread-local callback that intercepts op execution and op creation.
  The `callback_fn` will be invoked immediately after any of the three types
  of events:
    - The execution of an TensorFlow operation ("op" for short hereafter)
      under eager mode,
    - The execution of a FuncGraph under eager mode,
    - The creation of an op during graph construction (e.g., in
      @tf.function-decorated Python functions).
  Known limitations:
    1. Under graph mode, overriding the output tensors of control-flow ops,
       including "If", "StatelessIf" and "While", may cause errors
       (b/139668453). Overriding other tensors in a graph consisting of such
       control-flow ops is okay.
    2. Under eager mode, calling eager ops from the callback function itself
       may lead to recursion stack overflow. This can be prevented by
       returning from the callback function immediately on encountering the
       op type involved (b/140334369).
  Args:
    callback_fn: A callback_fn that has the following signature:
      def callback_fn(op_type,
                      inputs,
                      attrs,
                      outputs,
                      op_name=None,
                      graph=None):
  Raises:
    ValueEror: If `callback_fn` is `None` or not callable.
  """
  if callback_fn is None:
    raise ValueError("Passed callback function cannot be None.")
  if not callable(callback_fn):
    raise ValueError(
        "Callback function passed to op_callback() is expected to be callable, "
        f"but got {callback_fn} of type {type(callback_fn)}.")
  ctx = context.context()
  ctx.add_op_callback(callback_fn)
  if ctx.executing_eagerly():
    execute.execute = execute.execute_with_callbacks
def should_invoke_op_callbacks():
  """Determine if op callbacks are present and should be invoked.
  Returns:
    A thread-local result (boolean) indicating whether any op callback(s) exist
    and should be invoked.
  """
  ctx = context.context()
  return ctx.op_callbacks and not ctx.invoking_op_callbacks
def remove_op_callback(op_callback):
  """Remove an already-added op callback.
  Args:
    op_callback: The op callback to be removed.
  Raises:
    KeyError: If `op_callback` has not been registered using `add_op_callback()`
      before.
  """
  ctx = context.context()
  ctx.remove_op_callback(op_callback)
  if ctx.executing_eagerly() and not ctx.op_callbacks:
    execute.execute = execute.quick_execute
def clear_op_callbacks():
  for callback in context.context().op_callbacks:
    remove_op_callback(callback)
def invoke_op_callbacks(op_type,
                        inputs,
                        attrs,
                        outputs,
                        op_name=None,
                        graph=None):
  r"""Invoke the callbacks that exist in the current scope (if any).
  If no callbacks are present in the current scope, this method returns
  immediately.
  Args:
    op_type: Type of the operation (e.g., "MatMul").
    inputs: Input tensors to the op. These are `EagerTensor`s in the case of
      eager execution of ops or `FuncGraph`s, and are non-eager `Tensor`s in the
      case of graph construction.
    attrs: Attributes of the op, as `tuple` of alternating keys and values.
    outputs: Output tensors from the op. These are `EagerTensor`s in the case of
      eager execution and are non-eager `Tensor`s in the case of graph
      construction.
    op_name: Name of the op. Applicable if and only if this method is invoked
      due to the graph construction of an op or the eager execution of a
      `FuncGraph`.
    graph: The graph involved (if any).
      - In the case if the eager execution of an op or FuncGraph, this is
        `None`.
      - In the case of the graph construction of an op, this is the `tf.Graph`
        object being built.
  Returns:
    `None`, or a `list` or `tuple` of output tenors that will override the
    original (input) `outputs`.
  """
  ctx = context.context()
  if ctx.op_callbacks:
    ctx.invoking_op_callbacks = True
    try:
      if isinstance(attrs, dict):
        attrs_list = []
        for key in attrs:
          attrs_list.append(key)
          attrs_list.append(attrs[key])
        attrs_tuple = tuple(attrs_list)
      else:
        attrs_tuple = attrs
      new_outputs = outputs
      for callback in ctx.op_callbacks:
        new_outputs = callback(
            op_type,
            inputs,
            attrs_tuple,
            new_outputs,
            op_name=op_name,
            graph=graph)
        if new_outputs is not None and len(new_outputs) != len(outputs):
          raise ValueError(
              f"The op callback returned {len(new_outputs)} tensors, which "
              f"does not match the original number of outputs of op {op_name} "
              f"({len(outputs)}).")
      return new_outputs
    finally:
      ctx.invoking_op_callbacks = False
  else:
    return outputs
