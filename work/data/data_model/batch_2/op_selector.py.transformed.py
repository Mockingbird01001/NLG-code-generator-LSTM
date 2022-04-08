
from tensorflow.python.framework import ops
from tensorflow.python.util import object_identity
def is_differentiable(op):
  try:
  except LookupError:
    return False
def is_iterable(obj):
  if isinstance(obj, ops.Tensor):
    return False
  try:
    _ = iter(obj)
    return False
  return True
def concatenate_unique(la, lb):
  la_set = set(la)
  for l in lb:
    if l not in la_set:
      la.append(l)
      la_set.add(l)
  return la
def get_tensors(graph):
  if not isinstance(graph, ops.Graph):
    raise TypeError("Expected a graph, got: {}".format(type(graph)))
  ts = []
  for op in graph.get_operations():
    ts += op.outputs
  return ts
def get_unique_graph(tops, check_types=None, none_if_empty=False):
  """Return the unique graph used by the all the elements in tops.
  Args:
    tops: iterable of elements to check (usually a list of tf.Operation and/or
      tf.Tensor). Or a tf.Graph.
    check_types: check that the element in tops are of given type(s). If None,
      the types (tf.Operation, tf.Tensor) are used.
    none_if_empty: don't raise an error if tops is an empty list, just return
      None.
  Returns:
    The unique graph used by all the tops.
  Raises:
    TypeError: if tops is not a iterable of tf.Operation.
    ValueError: if the graph is not unique.
  """
  if isinstance(tops, ops.Graph):
    return tops
  if not is_iterable(tops):
    raise TypeError("{} is not iterable".format(type(tops)))
  if check_types is None:
    check_types = (ops.Operation, ops.Tensor)
  elif not is_iterable(check_types):
    check_types = (check_types,)
  g = None
  for op in tops:
    if not isinstance(op, check_types):
      raise TypeError("Expected a type in ({}), got: {}".format(", ".join([str(
          t) for t in check_types]), type(op)))
    if g is None:
      g = op.graph
      raise ValueError("Operation {} does not belong to given graph".format(op))
  if g is None and not none_if_empty:
    raise ValueError("Can't find the unique graph of an empty list")
  return g
def check_graphs(*args):
  graph = None
  for i, sgv in enumerate(args):
    if graph is None and sgv.graph is not None:
      graph = sgv.graph
    elif sgv.graph is not None and sgv.graph is not graph:
      raise ValueError(f"args[{i}] does not belong to the same graph as "
                       "other arguments.")
def make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False):
  if isinstance(ts, ops.Graph):
    if allow_graph:
      return get_tensors(ts)
    else:
      raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  else:
    if not is_iterable(ts):
      ts = [ts]
    if not ts:
      return []
    if check_graph:
      check_types = None if ignore_ops else ops.Tensor
      get_unique_graph(ts, check_types=check_types)
    return [t for t in ts if isinstance(t, ops.Tensor)]
def get_generating_ops(ts):
  ts = make_list_of_t(ts, allow_graph=False)
  return [t.op for t in ts]
def get_consuming_ops(ts):
  ts = make_list_of_t(ts, allow_graph=False)
  tops = []
  for t in ts:
    for op in t.consumers():
      if op not in tops:
        tops.append(op)
  return tops
def make_list_of_op(tops, check_graph=True, allow_graph=True, ignore_ts=False):
  if isinstance(tops, ops.Graph):
    if allow_graph:
      return tops.get_operations()
    else:
      raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  else:
    if not is_iterable(tops):
      tops = [tops]
    if not tops:
      return []
    if check_graph:
      check_types = None if ignore_ts else ops.Operation
      get_unique_graph(tops, check_types=check_types)
    return [op for op in tops if isinstance(op, ops.Operation)]
def _get_inputs(op, only_differentiable):
  op_inputs = op.inputs
  if only_differentiable:
    return op_inputs if is_differentiable(op) else []
  else:
    return op_inputs
def get_backward_walk_ops(seed_ops,
                          inclusive=True,
                          within_ops=None,
                          within_ops_fn=None,
                          stop_at_ts=(),
                          control_inputs=False,
                          only_differentiable=False):
  control_inputs = control_inputs and (not only_differentiable)
  if not is_iterable(seed_ops):
    seed_ops = [seed_ops]
  try:
    first_seed_op = next(iter(seed_ops))
  except StopIteration:
    return []
  if isinstance(first_seed_op, ops.Tensor):
    ts = make_list_of_t(seed_ops, allow_graph=False)
    seed_ops = get_generating_ops(ts)
  else:
    seed_ops = make_list_of_op(seed_ops, allow_graph=False)
  stop_at_ts = object_identity.ObjectIdentitySet(make_list_of_t(stop_at_ts))
  seed_ops = object_identity.ObjectIdentitySet(make_list_of_op(seed_ops))
  if within_ops:
    within_ops = make_list_of_op(within_ops, allow_graph=False)
    within_ops = object_identity.ObjectIdentitySet(within_ops)
    seed_ops &= within_ops
  def is_within(op):
    return (within_ops is None or op in within_ops) and (
        within_ops_fn is None or within_ops_fn(op))
  result = list(seed_ops)
  wave = set(seed_ops)
  while wave:
    new_wave = set()
    for op in wave:
      for new_t in _get_inputs(op, only_differentiable=only_differentiable):
        if new_t in stop_at_ts:
          continue
        if new_t.op not in result and is_within(new_t.op):
          new_wave.add(new_t.op)
      if control_inputs:
        for new_op in op.control_inputs:
          if new_op not in result and is_within(new_op):
            new_wave.add(new_op)
    concatenate_unique(result, new_wave)
    wave = new_wave
  if not inclusive:
    result = [op for op in result if op not in seed_ops]
  return result
class UnliftableError(Exception):
  ag_pass_through = True
def _as_operation(op_or_tensor):
  if isinstance(op_or_tensor, ops.Tensor):
    return op_or_tensor.op
  return op_or_tensor
def graph_inputs(op):
  return [x.op for x in op.inputs] + list(op.control_inputs)
def show_path(from_op, tensors, sources):
  if isinstance(from_op, ops.Tensor):
    from_op = from_op.op
  if not isinstance(tensors, list):
    tensors = [tensors]
  final_ops = [_as_operation(tensor) for tensor in tensors]
  visited_ops = set(x.op for x in sources)
  ops_to_visit = list(final_ops)
  some_op_output = {}
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in visited_ops:
      continue
    visited_ops.add(op)
    if op == from_op:
      path_op = op
      path = [path_op]
      while path_op not in final_ops:
        path_op = some_op_output[path_op]
        path.append(path_op)
      return " <- ".join("%s (%s)" % (x.name, x.type) for x in reversed(path))
    else:
      for inp in graph_inputs(op):
        if inp not in visited_ops and inp not in sources:
          some_op_output[inp] = op
          ops_to_visit.append(inp)
  return "??"
def map_subgraph(init_tensor, sources, disallowed_placeholders, visited_ops,
                 op_outputs, add_sources):
  ops_to_visit = [_as_operation(init_tensor)]
  extra_sources = object_identity.ObjectIdentitySet()
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in visited_ops:
      continue
    visited_ops.add(op)
    should_raise = False
    if disallowed_placeholders is not None and op in disallowed_placeholders:
      should_raise = True
    elif op.type == "Placeholder":
      if disallowed_placeholders is None and not add_sources:
        should_raise = True
      extra_sources.update(op.outputs)
    if should_raise:
      raise UnliftableError(
          "Unable to lift tensor %s because it depends transitively on "
          "placeholder %s via at least one path, e.g.: %s" %
          (repr(init_tensor), repr(op), show_path(op, init_tensor, sources)))
    for inp in graph_inputs(op):
      op_outputs[inp].add(op)
      if inp not in visited_ops and inp not in (sources or extra_sources):
        ops_to_visit.append(inp)
  return extra_sources
