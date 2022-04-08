
from collections import abc
import contextlib
import threading
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
def regroup(values, wrap_class=values_lib.PerReplica, always_wrap=False):
  v0 = values[0]
  if isinstance(v0, list):
    for v in values[1:]:
      assert isinstance(v, list)
      assert len(v) == len(v0), ("len(v) == %d, len(v0) == %d, v: %s, v0: %s" %
                                 (len(v), len(v0), v, v0))
    return [
        regroup(tuple(v[i] for v in values), wrap_class, always_wrap)
        for i in range(len(v0))
    ]
  if isinstance(v0, tuple):
    for v in values[1:]:
      assert isinstance(v, tuple)
      assert len(v) == len(v0), ("Values to regroup had different lengths: "
                                 f"len(v) == {len(v)}, len(v0) == {len(v0)}, "
                                 f"v: {v}, v0: {v0}")
    regrouped_tuple = tuple(
        regroup(tuple(v[i] for v in values), wrap_class, always_wrap)
        for i in range(len(v0)))
    if hasattr(v0, "_fields"):
      assert hasattr(v0, "_make")
      return v0._make(regrouped_tuple)
    else:
      return regrouped_tuple
  if isinstance(v0, abc.Mapping):
    v0keys = v0.keys()
    for v in values[1:]:
      assert isinstance(v, abc.Mapping), ("v[0]: %r  v[i]: %r" % (v0, v))
      assert set(v.keys()) == set(v0keys), ("v[0].keys: %s  v[i].keys: %s" %
                                            (set(v0keys), set(v.keys())))
    return type(v0)({
        key: regroup(tuple(v[key] for v in values),
                     wrap_class, always_wrap)
        for key in v0keys
    })
  same_id = True
  for v in values[1:]:
    if v is not v0:
      same_id = False
      break
  if same_id and isinstance(v0, values_lib.DistributedVariable):
    return v0
  if same_id and not always_wrap and not hasattr(v0, "_distributed_container"):
    return v0
  if hasattr(v0, "_distributed_container"):
    assert not isinstance(v0, values_lib.MirroredVariable), (
        "ids = %s, values = %s" % ([id(v) for v in values], values))
    distributed_container = v0._distributed_container()
    assert distributed_container is not None
    for v in values[1:]:
      assert distributed_container is v._distributed_container()
    return distributed_container
  return wrap_class(values)
def select_replica(replica_id, structured):
  def _get(x):
    if (isinstance(x, values_lib.DistributedVariable) or
        not isinstance(x, values_lib.DistributedValues)):
      return x
    else:
      return x.values[replica_id]
  return nest.map_structure(_get, structured)
def select_replica_mirrored(replica_id, structured):
  assert_mirrored(structured)
  return select_replica(replica_id, structured)
def assert_mirrored(structured):
  def _assert_mirrored(x):
    if isinstance(x, values_lib.DistributedValues) and not is_mirrored(x):
      raise TypeError(
          "Expected value to be mirrored across replicas: %s in %s." %
          (x, structured))
  nest.map_structure(_assert_mirrored, structured)
def update_regroup(extended, updates, group):
  if not group:
    regrouped = regroup(updates, values_lib.Mirrored)
  def _make_grouped_mirrored(values):
    if len(values) == 1:
      return values_lib.Mirrored(values)
    g = control_flow_ops.group(values)
    if not all(tensor_util.is_tf_type(v) for v in values):
      return g
    with_dep = []
    for v in values:
      with ops.device(v.device), ops.control_dependencies([g]):
        with_dep.append(array_ops.identity(v))
    return values_lib.Mirrored(with_dep)
  return regroup(updates, _make_grouped_mirrored)
def value_container(val):
  """Returns the container that this per-replica `value` belongs to.
  Args:
    val: A value returned by `call_for_each_replica()` or a variable created in
      `scope()`.
  Returns:
    A container that `value` belongs to.
    If value does not belong to any container (including the case of
    container having been destroyed), returns the value itself.
  """
  if (hasattr(val, "_distributed_container") and
      not isinstance(val, values_lib.DistributedVariable)):
    if container is not None:
      return container
  return val
def is_distributed_variable(v):
  return getattr(v, "is_distributed_variable", False)
def is_distributed_table(v):
  return getattr(v, "is_distributed_table", False)
def _validate_colocate_extended(v, extended):
  if variable_strategy.extended is not extended:
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not %s created in scope: %s" %
        (v, variable_strategy))
def validate_colocate_distributed_variable(v, extended):
  if not isinstance(v, values_lib.DistributedVariable):
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not: %r" % (v,))
  _validate_colocate_extended(v, extended)
def validate_colocate(v, extended):
  if not hasattr(v, "_distribute_strategy"):
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not: %r" % (v,))
  _validate_colocate_extended(v, extended)
def _validate_synchronization(kwargs):
  synchronization = kwargs.get("synchronization",
                               vs.VariableSynchronization.AUTO)
  if synchronization == vs.VariableSynchronization.NONE:
    raise ValueError(
        "`NONE` variable synchronization mode is not supported with "
        "tf.distribute strategy. Please change the `synchronization` for "
        "variable: " + str(kwargs["name"]))
  if synchronization not in (vs.VariableSynchronization.ON_READ,
                             vs.VariableSynchronization.ON_WRITE,
                             vs.VariableSynchronization.AUTO):
    raise ValueError(
        "Invalid variable synchronization mode: %s for variable: %s" %
        (synchronization, kwargs["name"]))
  if synchronization == vs.VariableSynchronization.AUTO:
    return vs.VariableSynchronization.ON_WRITE
  return synchronization
def _validate_aggregation(kwargs):
  aggregation = kwargs.get("aggregation", vs.VariableAggregation.NONE)
  if aggregation not in (vs.VariableAggregation.NONE,
                         vs.VariableAggregation.SUM,
                         vs.VariableAggregation.MEAN,
                         vs.VariableAggregation.ONLY_FIRST_REPLICA):
    raise ValueError("Invalid variable aggregation mode: %s for variable: %s" %
                     (aggregation, kwargs["name"]))
  return aggregation
def create_mirrored_variable(strategy, real_mirrored_creator, class_mapping,
                             policy_mapping, **kwargs):
  var_collections = kwargs.pop("collections", None)
  if var_collections is None:
    var_collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  kwargs["collections"] = []
  synchronization = _validate_synchronization(kwargs)
  kwargs["synchronization"] = synchronization
  aggregation = _validate_aggregation(kwargs)
  use_var_policy = getattr(strategy.extended, "_use_var_policy", False)
  kwargs.pop("caching_device", None)
  with tape.stop_recording():
    value_list = real_mirrored_creator(**kwargs)
    for v in value_list:
      if hasattr(v, "_initializer_op") and v._initializer_op is None:
        v._initializer_op = control_flow_ops.no_op()
    if use_var_policy:
      var_policy_cls = policy_mapping.get(synchronization)
      var_policy = var_policy_cls(aggregation=aggregation)
      var_cls = class_mapping.get("VariableClass")
      result = var_cls(strategy, value_list, aggregation, var_policy=var_policy)
    else:
      var_cls = class_mapping.get(synchronization)
      result = var_cls(strategy, value_list, aggregation)
  if not context.executing_eagerly():
    g = ops.get_default_graph()
    if kwargs.get("trainable", True):
      var_collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
      l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
      for value in value_list:
        for i, trainable_variable in enumerate(l):
          if value is trainable_variable:
            del l[i]
            break
    g.add_to_collections(var_collections, result)
  elif ops.GraphKeys.GLOBAL_STEP in var_collections:
    ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, result)
  return result
def is_mirrored(val):
  if isinstance(val, values_lib.DistributedVariable):
  return isinstance(val, values_lib.Mirrored)
def is_sync_on_read(val):
  if isinstance(val, values_lib.DistributedVariable):
  return not isinstance(val, values_lib.Mirrored)
class CachingScopeLocal(threading.local):
  def __init__(self):
    super(CachingScopeLocal, self).__init__()
    self.new_cache_scope_count = 0
    self.cache_scope_exited_count = 0
  def enter_scope(self):
    self.new_cache_scope_count += 1
  def exit_scope(self):
    self.cache_scope_exited_count += 1
  def in_caching_scope(self):
    return self.new_cache_scope_count > self.cache_scope_exited_count
caching_scope_local = CachingScopeLocal()
@contextlib.contextmanager
def cache_variable_reads():
  """Scope for caching variable reads for AggregatingVariable.
  The variable reads for AggregatingVariable inside this scope are cached. i.e.
  the first read of variable reads the value from possibly remote handle, but
  subsequent reads are returned using local cached value.
  For example:
  strategy = ParameterServerStrategy...
  with strategy.scope():
    v = tf.Variable(1.0)
  with distribute_utils.cache_variable_reads():
    t1 = v.read_value()
  Notes about cache_variable_reads scope:
  1. Nesting of scope cache_variable_reads() is not supported
  2. And when caching scope is enabled, the thread enabling the cache and
    mirrored_run._MirroredReplicaThread threads spawned from it will have
    caching enabled.
  Yields:
    A context for caching variables.
  """
  try:
    if caching_scope_local.in_caching_scope():
      raise ValueError("cache_variable_reads scope cannot be nested")
    caching_scope_local.enter_scope()
    yield
  finally:
    caching_scope_local.exit_scope()
VARIABLE_POLICY_MAPPING = {
    vs.VariableSynchronization.ON_WRITE: values_lib.OnWritePolicy,
    vs.VariableSynchronization.ON_READ: values_lib.OnReadPolicy,
}
VARIABLE_CLASS_MAPPING = {
    "VariableClass": values_lib.DistributedVariable,
    vs.VariableSynchronization.ON_WRITE: values_lib.MirroredVariable,
    vs.VariableSynchronization.ON_READ: values_lib.SyncOnReadVariable,
}
TPU_VARIABLE_POLICY_MAPPING = {
    vs.VariableSynchronization.ON_WRITE: tpu_values_lib.TPUOnWritePolicy,
    vs.VariableSynchronization.ON_READ: tpu_values_lib.TPUOnReadPolicy,
}
TPU_VARIABLE_CLASS_MAPPING = {
    "VariableClass": tpu_values_lib.TPUDistributedVariable,
    vs.VariableSynchronization.ON_WRITE: tpu_values_lib.TPUMirroredVariable,
    vs.VariableSynchronization.ON_READ: tpu_values_lib.TPUSyncOnReadVariable,
}
