
import contextlib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.tpu import tpu
def enclosing_tpu_context():
  return enclosing_tpu_context_and_graph()[0]
def enclosing_tpu_context_and_graph():
  graph = ops.get_default_graph()
  while graph is not None:
    while ctx is not None:
      if isinstance(ctx, tpu.TPUReplicateContext):
        return ctx, graph
      ctx = ctx.outer_context
    graph = getattr(graph, "outer_graph", None)
  return None, None
@contextlib.contextmanager
def outside_or_skip_tpu_context():
  ctx, graph = enclosing_tpu_context_and_graph()
  if ctx is None:
    yield
  else:
    yield
@contextlib.contextmanager
def _maybe_enter_graph(tensor):
  if (context.executing_eagerly() or isinstance(tensor, ops.EagerTensor) or
      ops.has_default_graph()):
    yield
  else:
    with tensor.graph.as_default():
      yield
@contextlib.contextmanager
def _maybe_on_device(var):
  if isinstance(var, packed.PackedVarAndDevice):
    with ops.device(var.device):
      yield
  else:
    yield
def make_raw_assign_fn(raw_assign_fn, use_handle=True):
  def assign_fn(var, value, use_locking=False, name=None, read_value=True):
    handle = var.handle if use_handle else var
    with _maybe_enter_graph(handle), _maybe_on_device(var):
      op = raw_assign_fn(
          handle, ops.convert_to_tensor(value, dtype=var.dtype), name=name)
      with ops.control_dependencies([op]):
        if read_value:
        else:
          return op
  return assign_fn
def make_raw_scatter_xxx_fn(raw_scatter_xxx_fn):
    handle = var.handle
    with _maybe_enter_graph(handle), _maybe_on_device(var):
      op = raw_scatter_xxx_fn(
          handle,
          sparse_delta.indices,
          ops.convert_to_tensor(sparse_delta.values, var.dtype),
          name=name)
      with ops.control_dependencies([op]):
  return scatter_xxx_fn
