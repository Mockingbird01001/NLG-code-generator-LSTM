
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
def _GetMaxSizeFromNestedMaximumIterations(value, while_ctxt):
  value_name = value.name
  curr_ctxt_name = curr_ctxt.name if curr_ctxt is not None else ""
  max_size = constant_op.constant(1)
  while while_ctxt not in (None, curr_ctxt):
    max_iter = while_ctxt.maximum_iterations
    if max_iter is None:
      raise ValueError(
          "Cannot create a gradient accumulator for tensor '%s' inside "
          "XLA while_loop because maximum_iterations was not passed to "
          "the tf.while_loop call ('%s')." % (value_name, while_ctxt.name))
    max_iter_ctxt = max_iter.op._get_control_flow_context()
    if util.IsContainingContext(curr_ctxt, max_iter_ctxt):
      max_size *= max_iter
    else:
      const_max_iter = tensor_util.constant_value(max_iter)
      if const_max_iter is None:
        raise ValueError(
            "Cannot create a gradient accumulator for tensor '%s' inside XLA "
            "while_loop. maximum_iterations tensor '%s' for while_loop context "
            "'%s' must be statically known (e.g. a constant value or known "
            "shape dimension), or be defined at or outside the while loop "
            "context '%s' (currently defined in '%s')." %
            (value_name, max_iter.name, while_ctxt.name, curr_ctxt_name,
             max_iter_ctxt.name))
      max_size *= const_max_iter
    while_ctxt = util.GetContainingWhileContext(
        while_ctxt.outer_context, stop_ctxt=curr_ctxt)
  return max_size
class _GradLoopState:
  """The state used for constructing the gradient graph for a while loop.
  We create a _GradLoopState for each while loop in forward and its
  corresponding while loop in backprop. This gives us access to both
  the forward and the backprop WhileContexts.
  During the construction of gradient graph, any time when we detect
  a forward value that is needed for backprop, we create a history
  accumulator and add it to `history_map`. Any time when we backprop
  a loop switch op (in _SwitchGrad), we add the grad merge op in
  `switch_map`.
  """
  def __init__(self, forward_ctxt, outer_grad_state):
    self._outer_grad_state = None
    self._forward_context = None
    self._forward_index = None
    self._forward_sync = None
    self._grad_context = None
    self._grad_index = None
    self._grad_sync = None
    self._history_map = {}
    self._switch_map = {}
    self._unused_exits = []
    self._deferred_exits = []
    self._forward_loop_exits = list(forward_ctxt.loop_exits)
    self._pending_exits_count = len(forward_ctxt.loop_exits)
    self._outer_grad_state = outer_grad_state
    if outer_grad_state:
      outer_forward_ctxt = outer_grad_state.forward_context
    else:
      if not hasattr(forward_ctxt, "outer_context"):
        raise ValueError("Failed to call gradients on a while loop without"
                         "properly serializing graph via MetaGraphDef")
      outer_forward_ctxt = forward_ctxt.outer_context
      if outer_forward_ctxt:
        outer_forward_ctxt.Enter()
      cnt, forward_index = forward_ctxt.AddForwardLoopCounter(outer_grad_state)
      if outer_forward_ctxt:
        outer_forward_ctxt.Exit()
    self._forward_context = forward_ctxt
    self._forward_index = forward_index
    if outer_grad_state:
      outer_forward_ctxt.AddName(cnt.name)
      history_cnt = outer_grad_state.AddForwardAccumulator(cnt)
      outer_grad_ctxt = outer_grad_state.grad_context
      outer_grad_ctxt.Enter()
      self._grad_context = control_flow_ops.WhileContext(
          maximum_iterations=forward_ctxt.maximum_iterations,
          parallel_iterations=forward_ctxt.parallel_iterations,
          back_prop=forward_ctxt.back_prop,
          swap_memory=forward_ctxt.swap_memory,
          name=forward_ctxt.name,
          grad_state=self)
      real_cnt = outer_grad_state.AddBackpropAccumulatedValue(history_cnt, cnt)
      self._grad_index = self._grad_context.AddBackpropLoopCounter(
          real_cnt, outer_grad_state)
      outer_grad_ctxt.Exit()
    else:
      if outer_forward_ctxt:
        outer_forward_ctxt.Enter()
      self._grad_context = control_flow_ops.WhileContext(
          maximum_iterations=forward_ctxt.maximum_iterations,
          parallel_iterations=forward_ctxt.parallel_iterations,
          back_prop=forward_ctxt.back_prop,
          swap_memory=forward_ctxt.swap_memory,
          name=forward_ctxt.name,
          grad_state=self)
      self._grad_index = self._grad_context.AddBackpropLoopCounter(
          cnt, outer_grad_state)
      if outer_forward_ctxt:
        outer_forward_ctxt.Exit()
  @property
  def outer_grad_state(self):
    return self._outer_grad_state
  @property
  def forward_context(self):
    return self._forward_context
  @property
  def forward_index(self):
    return self._forward_index
  @property
  def forward_sync(self):
    if self._forward_sync is None:
      with ops.control_dependencies(None):
        self._forward_sync = control_flow_ops.control_trigger(name="f_sync")
      self._forward_sync._set_control_flow_context(self._forward_context)
      self._forward_index.op._add_control_input(self._forward_sync)
    return self._forward_sync
  @property
  def grad_context(self):
    return self._grad_context
  @property
  def grad_index(self):
    return self._grad_index
  @property
  def grad_sync(self):
    if self._grad_sync is None:
      with ops.control_dependencies(None):
        self._grad_sync = control_flow_ops.control_trigger(name="b_sync")
      self._grad_sync._set_control_flow_context(self._grad_context)
      self._grad_index.op._add_control_input(self._grad_sync)
      if self._grad_context.outer_context:
        self._grad_context.outer_context.AddInnerOp(self._grad_sync)
    return self._grad_sync
  @property
  def history_map(self):
    return self._history_map
  @property
  def switch_map(self):
    return self._switch_map
  @property
  def unused_exits(self):
    return self._unused_exits
  @property
  def deferred_exits(self):
    return self._deferred_exits
  @property
  def forward_loop_exits(self):
    return self._forward_loop_exits
  @property
  def pending_exits_count(self):
    return self._pending_exits_count
  @pending_exits_count.setter
  def pending_exits_count(self, cnt):
    self._pending_exits_count = cnt
  def AddForwardAccumulator(self, value, dead_branch=False):
    """Add an accumulator for each forward tensor that is needed in backprop.
    This is added to the forward loop at the first time when a tensor
    in the forward loop is used by backprop gradient computation loop.
    We create an accumulator that accumulates the value of tensor at each
    iteration. Called in the control flow context where gradients() is called.
    The pseudocode is:
    ```
      acc = stack();
      while (_pivot) {
        acc = stack_push(acc, value);
      }
    ```
    We make sure that the stack push op in one iteration is executed before
    next iteration. This is achieved by adding a control edge from
    `forward_index.op.inputs[0].op` to the push op, and another control
    edge from the push op to either `forward_index.op` or `forward_sync`.
    Args:
      value: The source tensor in forward that is to be accumulated.
      dead_branch: True iff the tensor is on a dead branch of a cond.
    Returns:
      The stack that contains the accumulated history of the tensor.
    Raises:
      TypeError: For internal errors involving the value condition context.
      ValueError: If `value` is inside a XLA scope and a valid max size
        for the stack can't be found.
    """
    with self._forward_index.graph.as_default():
      with ops.control_dependencies(None):
        if curr_ctxt:
          curr_ctxt.Enter()
        with ops.colocate_with(value):
          if not util.IsInXLAContext(value.op):
            max_size = constant_op.constant(-1, dtypes.int32)
          else:
            max_size = _GetMaxSizeFromNestedMaximumIterations(
                value, self.forward_context)
          acc = gen_data_flow_ops.stack_v2(
              max_size=max_size, elem_type=value.dtype.base_dtype, name="f_acc")
        if curr_ctxt:
          curr_ctxt.Exit()
        enter_acc = self.forward_context.AddValue(acc)
        swap_enabled = self.forward_context.swap_memory
        value_ctxt = util.GetOutputContext(value.op)
        if value_ctxt == self.forward_context:
          self.forward_context.Enter()
          push = gen_data_flow_ops.stack_push_v2(
              enter_acc, value, swap_memory=swap_enabled)
          self.forward_context.Exit()
          self.forward_index.op._add_control_input(push.op)
        else:
          if not isinstance(value_ctxt, control_flow_ops.CondContext):
            raise TypeError("value_ctxt is not a CondContext: %s" % value_ctxt)
          if dead_branch:
            value_ctxt.outer_context.Enter()
            push = gen_data_flow_ops.stack_push_v2(
                enter_acc, value, swap_memory=swap_enabled)
            value_ctxt.outer_context.Exit()
            push.op._set_control_flow_context(value_ctxt)
          else:
            value_ctxt.Enter()
            push = gen_data_flow_ops.stack_push_v2(
                enter_acc, value, swap_memory=swap_enabled)
            value_ctxt.Exit()
          self.forward_sync._add_control_input(push.op)
        add_op = self.forward_index.op.inputs[0].op
        push.op._add_control_input(add_op)
        return acc
  def AddBackpropAccumulatedValue(self, history_value, value,
                                  dead_branch=False):
    """Add the getter for an accumulated value in the grad context.
    This is added to the backprop loop. Called in the grad context to
    get the value of an accumulated value. The stack pop op must be guarded
    by the pred of the controlling cond.
    Args:
      history_value: The history (a stack) of a value.
      value: The value that is pushed onto the stack.
      dead_branch: True iff the tensor is on a dead branch of a cond.
    Returns:
      The current value (the top of the stack).
    """
    history_ctxt = history_value.op._get_control_flow_context()
    cond_ctxt = None
    value_ctxt = value.op._get_control_flow_context()
    while value_ctxt and value_ctxt != history_ctxt:
      if isinstance(value_ctxt, control_flow_ops.CondContext):
        cond_ctxt = value_ctxt
        break
      value_ctxt = value_ctxt.outer_context
    with ops.control_dependencies(None):
      self.grad_context.Enter()
      if cond_ctxt:
        grad_state = self
        pred = None
        while pred is None and grad_state:
          pred = grad_state.history_map.get(cond_ctxt.pred.name)
          grad_state = grad_state.outer_grad_state
        if pred is None:
          pred = cond_ctxt.pred
        branch = (1 - cond_ctxt.branch) if dead_branch else cond_ctxt.branch
        history_value = control_flow_ops._SwitchRefOrTensor(
            history_value, pred)[branch]
      pop = gen_data_flow_ops.stack_pop_v2(history_value,
                                           value.dtype.base_dtype)
      pop.set_shape(value.get_shape())
      self.grad_context.Exit()
    parallel_iterations = self.grad_context.parallel_iterations
    if parallel_iterations > 1:
      self.grad_sync._add_control_input(pop.op)
    return pop
  def GetRealValue(self, value):
    assert value.op.type not in ["Variable", "VariableV2"]
    real_value = self._history_map.get(value.name)
    if real_value is None:
      cur_value = value
      cur_grad_state = self
      while True:
        enter_op = util.GetLoopConstantEnter(cur_value)
        if enter_op:
          cur_value = enter_op.inputs[0]
          cur_grad_state = cur_grad_state.outer_grad_state
          if cur_grad_state is None:
            real_value = self._grad_context.AddValue(cur_value)
            break
        elif constant_op.is_constant(cur_value):
          real_value = constant_op.constant(
              tensor_util.constant_value(cur_value), dtype=cur_value.dtype)
          break
        else:
          self._grad_context.Exit()
          history_value = cur_grad_state.AddForwardAccumulator(cur_value)
          self._grad_context.Enter()
          break
      if real_value is None:
        real_value = cur_grad_state.AddBackpropAccumulatedValue(
            history_value, cur_value)
        if cur_grad_state != self:
          real_value = self._grad_context.AddValue(real_value)
      self._history_map[value.name] = real_value
    return real_value
class _ControlFlowState:
  def __init__(self):
  def GetGradState(self, op, before):
    if before and util.IsLoopExit(op):
      forward_ctxt = forward_ctxt.outer_context
      if forward_ctxt:
        forward_ctxt = forward_ctxt.GetWhileContext()
    else:
      forward_ctxt = util.GetWhileContext(op)
    if forward_ctxt:
      return self._map.get(forward_ctxt)
    return None
  def ProcessUnusedLoopExits(self, pending_count, to_ops_set):
    """Process all the "unused" loop exits.
    The "unused" exits of the loops are added to `unused_exits`. An exit is
    unused if its pending_count is 0. If there is an exit with real gradient,
    all these deferred exits will enter the backprop loop with zero gradient.
    Otherwise, they will enter the backprop loop with None. As an example,
    people often write:
    ```python
    v1, _ = tf.while_loop(p, b, [x1, x2])
    result = gradients(v1, x1)
    ```
    The exit node for x2 is not included by the betweenness analysis. But we
    need to backprop x2 if x2 is involved in computing v1.
    Args:
      pending_count: The number of backprop inputs for every op.
      to_ops_set: The set of ops for ys in gradients(ys, xs)
    Returns:
      The set of unused loop exits that we know at this point we need
      to backprop.
    """
    loop_exits = []
    for grad_state in self._map.values():
      for y in grad_state.forward_loop_exits:
        if pending_count[y.op] == 0:
          grad_state.pending_exits_count -= 1
          if y.op not in to_ops_set:
            grad_state.unused_exits.append(y)
          if grad_state.pending_exits_count == 0:
            loop_exits.extend(grad_state.unused_exits)
      for y in grad_state.forward_context.loop_enters:
        if pending_count[y.op] == 0:
          pending_count[y.op] = 1
    return loop_exits
  def EnterGradWhileContext(self, op, before):
    grad_state = self.GetGradState(op, before)
    if grad_state:
      grad_state.grad_context.Enter()
  def ExitGradWhileContext(self, op, before):
    grad_state = self.GetGradState(op, before)
    if grad_state:
      grad_state.grad_context.Exit()
  def AddWhileContext(self, op, between_op_list, between_ops):
    """Add the grad state for the while loop that op belongs to.
    Note that op is an Exit, and this method must be called in
    the control flow context where gradients() is called.
    Note that this method modifies `between_op_list` and `between_ops`.
    """
    forward_ctxt = util.GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      outer_forward_ctxt = forward_ctxt.outer_context
      if outer_forward_ctxt:
        outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
      outer_grad_state = None
      if outer_forward_ctxt:
        outer_grad_state = self._map.get(outer_forward_ctxt)
      grad_state = _GradLoopState(forward_ctxt, outer_grad_state)
      self._map[forward_ctxt] = grad_state
      for loop_exit in grad_state.forward_loop_exits:
        if loop_exit.op not in between_ops:
          between_ops.add(loop_exit.op)
          between_op_list.append(loop_exit.op)
  def ZerosLikeForExit(self, val):
    """Create zeros_like gradient for a loop exit.
    If the result of a loop variable is not used but is involved in
    computing the result of some needed loop variable, we create a
    zero-valued tensor that is fed as gradient for the Exit node of that
    loop variable. Note that val.op is an Exit, and this method must be
    called in the control flow context where gradients() is called.
    Args:
      val: The output tensor of an Exit op.
    Returns:
      A zero tensor of the same shape of val.
    """
    val_shape = val.get_shape()
    forward_ctxt = val.op._get_control_flow_context()
    outer_forward_ctxt = forward_ctxt.outer_context
    if outer_forward_ctxt:
      outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
    outer_grad_state = None
    if outer_forward_ctxt:
      outer_grad_state = self._map.get(outer_forward_ctxt)
    if outer_grad_state:
      if val_shape.is_fully_defined():
        outer_grad_state.grad_context.Enter()
        result = array_ops.zeros(val_shape.dims, val.dtype)
        outer_grad_state.grad_context.Exit()
      else:
        forward_ctxt.outer_context.Enter()
        shape = array_ops.shape_internal(val, optimize=False)
        forward_ctxt.outer_context.Exit()
        history_shape = outer_grad_state.AddForwardAccumulator(shape)
        outer_grad_ctxt = outer_grad_state.grad_context
        outer_grad_ctxt.Enter()
        real_shape = outer_grad_state.AddBackpropAccumulatedValue(
            history_shape, shape)
        result = array_ops.zeros(real_shape, val.dtype)
        outer_grad_ctxt.Exit()
    else:
      if val_shape.is_fully_defined():
        result = array_ops.zeros(val_shape.dims, val.dtype)
      else:
        result = array_ops.zeros_like(val, optimize=False)
    return result
  def ZerosLikeV1WhileLoop(self, op, index):
    """Create zeros_like for the specified output of an op.
    If op is in a while loop that is part of gradients(), this method
    must be called in its grad loop context.
    Args:
      op: A tensorflow operation.
      index: the index for a specific output of the op.
    Returns:
      A zero tensor of the same shape of op.outputs[index].
    """
    if util.IsLoopSwitch(op):
      return None
    if op.graph.building_function:
      return array_ops.zeros_like(op.outputs[index])
    dead_branch = util.IsSwitch(op)
    forward_ctxt = util.GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      return ZerosLike(op, index)
    op_ctxt = op._get_control_flow_context()
    val = ops.convert_to_tensor(op.outputs[index], name="tensor")
    shape = val.get_shape()
    if shape.is_fully_defined():
      if val.dtype == dtypes.resource:
        result = array_ops.zeros(
            resource_variable_ops.variable_shape(val),
            dtype=default_gradient.get_zeros_dtype(val))
      else:
        result = constant_op.constant(0, shape=shape.dims, dtype=val.dtype)
      if dead_branch:
        pred = grad_state.history_map.get(op_ctxt.pred.name)
        branch = op_ctxt.branch
        result = control_flow_ops._SwitchRefOrTensor(result, pred)[1 - branch]
    else:
      if dead_branch:
        pred = op_ctxt.pred
        branch = op_ctxt.branch
        op_ctxt.outer_context.Enter()
        val = control_flow_ops._SwitchRefOrTensor(op.inputs[0],
                                                  pred)[1 - branch]
        zeros_shape = array_ops.shape_internal(val, optimize=False)
        op_ctxt.outer_context.Exit()
        val.op._set_control_flow_context(op_ctxt)
        zeros_shape.op._set_control_flow_context(op_ctxt)
      else:
        op_ctxt.Enter()
        zeros_shape = array_ops.shape_internal(val, optimize=False)
        op_ctxt.Exit()
      grad_state.grad_context.Exit()
      history_zeros_shape = grad_state.AddForwardAccumulator(
          zeros_shape, dead_branch=dead_branch)
      grad_state.grad_context.Enter()
      shape = grad_state.AddBackpropAccumulatedValue(history_zeros_shape,
                                                     zeros_shape, dead_branch)
      result = array_ops.zeros(shape, val.dtype)
    return result
  def PostProcessing(self):
    """Perform postprocessing at the end of gradients().
    We have created the gradient graph at this point. So this function
    can be used to perform any postprocessing on the gradient graph.
    We currently perform the following postprocessing:
      1. Patch the gradient graph if the output of a loop variable
         doesn't depend on its input.
    """
    for _, grad_state in self._map.items():
      for _, b_merge in grad_state.switch_map.items():
        if b_merge.op.inputs[0] == b_merge.op.inputs[1]:
          dtype = b_merge.op.inputs[0].dtype
          shape = b_merge.op.inputs[0].get_shape()
          if shape.is_fully_defined():
            grad_state.grad_context.Enter()
            grad_val = constant_op.constant(0, dtype=dtype, shape=shape)
            next_grad_val = control_flow_ops._NextIteration(grad_val)
            grad_state.grad_context.Exit()
          else:
            outer_grad_ctxt = grad_state.grad_context.outer_context
            if outer_grad_ctxt:
              outer_grad_ctxt.Enter()
            enter_grad_op = b_merge.op.inputs[0].op
            enter_grad = enter_grad_op.inputs[0]
            grad_shape = array_ops.shape_internal(enter_grad, optimize=False)
            grad_val = array_ops.zeros(grad_shape)
            if outer_grad_ctxt:
              outer_grad_ctxt.Exit()
            grad_state.grad_context.Enter()
            next_grad_val = control_flow_ops._NextIteration(grad_val)
            grad_state.grad_context.Exit()
          b_merge.op._update_input(1, next_grad_val)
def MaybeCreateControlFlowState(between_op_list, between_ops,
                                colocate_gradients_with_ops):
  """Create the state for all the while loops involved in one gradients().
  We create a _ControlFlowState when there are while loops involved in
  gradients(). In gradients(), control flow logic is only invoked when
  the _ControlFlowState is not None.
  Note that this method modifies `between_op_list` and `between_ops`.
  """
  loop_state = None
  for op in between_op_list:
    if util.IsLoopExit(op):
      if loop_state is None:
        loop_state = _ControlFlowState()
      if colocate_gradients_with_ops:
        with ops.colocate_with(op):
          loop_state.AddWhileContext(op, between_op_list, between_ops)
      else:
        loop_state.AddWhileContext(op, between_op_list, between_ops)
  return loop_state
def _ZerosLikeV1(op, index):
  val = op.outputs[index]
  if op_ctxt:
    pred = op_ctxt.pred
    branch = op_ctxt.branch
    switch_val = control_flow_ops.switch(op.inputs[0], pred)[1 - branch]
    pivot = array_ops.identity(switch_val)
    if val.dtype == dtypes.resource:
      with ops.control_dependencies([pivot]):
        return array_ops.zeros(
            gen_resource_variable_ops.variable_shape(switch_val),
            dtype=default_gradient.get_zeros_dtype(val))
    zeros_shape = array_ops.shape_internal(switch_val, optimize=False)
    with ops.control_dependencies([pivot]):
      return array_ops.zeros(zeros_shape, dtype=val.dtype)
  else:
    return array_ops.zeros_like(val, optimize=False)
def _ZerosLikeV2(op, index):
  val = op.outputs[index]
  if val.dtype == dtypes.resource:
    return array_ops.zeros(
        gen_resource_variable_ops.variable_shape(val),
        dtype=default_gradient.get_zeros_dtype(val))
  if (isinstance(val.op.graph, control_flow_v2_func_graphs.WhileBodyFuncGraph)
      and val.dtype != dtypes.variant):
    if val.shape.is_fully_defined():
      return constant_op.constant(0, shape=val.shape.dims, dtype=val.dtype)
    else:
      zeros_shape = array_ops.shape_internal(val, optimize=False)
      return array_ops.zeros(zeros_shape, val.dtype)
  else:
    return array_ops.zeros_like(val, optimize=False)
def ZerosLike(op, index):
  if not util.IsSwitch(op):
    return _ZerosLikeV2(op, index)
  else:
    return _ZerosLikeV1(op, index)
