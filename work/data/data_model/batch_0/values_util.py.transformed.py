
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.training.saving import saveable_object
def write_object_proto(var, proto, options):
  """Update a SavedObject proto for the caller.
  If a DistributedVariable object supports this method, it will be called when
  saving with a pre-built `SavedObject` proto representing the object, plus an
  instance of `SaveOptions`. This method is then free to modify that proto
  instance.
  `DistributedVariable` with `AUTO` or `ON_WRITE` synchronization optionally
   write out information about their components to the
   `experimental_distributed_variable_components` field of a
   `SavedVariable` (depending on the `SaveOptions` variable policy).
  Args:
    var: The DistributedVariable object.
    proto: A pre-built `SavedObject` proto for this object. It is assumed this
      will be a `SavedVariable` instance.
    options: A `SaveOptions` instance.
  """
  ):
    for var in var.values:
      var_proto = (
          proto.variable.experimental_distributed_variable_components.add())
      var_proto.name = var.name.split(":")[0]
      var_proto.device = var.device
def get_on_write_saveable(var, primary_var, name):
  def tensor():
    if context.executing_eagerly() and not primary_var.is_initialized():
      return None
    strategy = var.distribute_strategy
    return strategy.extended.read_var(var)
  spec = saveable_object.SaveSpec(
      tensor=tensor,
      slice_spec="",
      name=name,
      dtype=var.dtype,
      device=primary_var.device)
  return tensor, [spec]
def get_on_write_restore_ops(var, tensor):
  if packed_var is not None:
    return control_flow_ops.group(
        tuple(
            assign_on_device(d, packed_var, tensor)
            for d in packed_var.devices))
  return control_flow_ops.group(
      tuple(
          assign_on_device(v.device, v, tensor)
          for v in var.values))
def get_on_read_saveable(var, primary_var, name):
  def tensor():
  spec = saveable_object.SaveSpec(
      tensor=tensor,
      slice_spec="",
      name=name,
      dtype=var.dtype,
      device=primary_var.device)
  return tensor, [spec]
def get_on_read_restore_ops(var, tensor, aggregation):
  if aggregation == vs.VariableAggregation.SUM:
    strategy = var.distribute_strategy
    tensor = math_ops.cast(tensor / strategy.num_replicas_in_sync,
                           var.dtype)
  return control_flow_ops.group(
      tuple(
          assign_on_device(v.device, v, tensor)
          for v in var.values))
def in_replica_update_context():
  return distribute_lib.get_update_replica_id() is not None
def on_write_assign(var, value, use_locking=False, name=None, read_value=True):
  assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
      update_fn=assign_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)
def on_write_assign_add(var, value, use_locking=False, name=None,
                        read_value=True):
  assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
      update_fn=assign_add_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)
def on_write_assign_sub(var, value, use_locking=False, name=None,
                        read_value=True):
  assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
      update_fn=assign_sub_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)
def assign_on_each_device(var, assign_func, value, read_value):
    update = control_flow_ops.group(
        tuple(
  else:
    update = control_flow_ops.group(
  if not read_value:
    return update
  with ops.control_dependencies([update] if update else []):
    return var.read_value()
def on_read_assign_sub_cross_replica(var, value, read_value=True):
  with ds_context.enter_or_assert_strategy(var.distribute_strategy):
    if ds_context.in_cross_replica_context():
      if var.aggregation == vs.VariableAggregation.SUM:
        raise ValueError(
            "SyncOnReadVariable does not support `assign_sub` in "
            "cross-replica context when aggregation is set to "
            "`tf.VariableAggregation.SUM`.")
      return assign_on_each_device(var, assign_sub_on_device,
                                   value, read_value)
def on_read_assign_add_cross_replica(var, value, read_value=True):
  with ds_context.enter_or_assert_strategy(var.distribute_strategy):
    if ds_context.in_cross_replica_context():
      if var.aggregation == vs.VariableAggregation.SUM:
        raise ValueError(
            "SyncOnReadVariable does not support `assign_add` in "
            "cross-replica context when aggregation is set to "
            "`tf.VariableAggregation.SUM`.")
      return assign_on_each_device(var, assign_add_on_device,
                                   value, read_value)
def on_read_assign_cross_replica(var, value, read_value=True):
  with ds_context.enter_or_assert_strategy(var.distribute_strategy):
    if ds_context.in_cross_replica_context():
      tensor = value
      if var.aggregation == vs.VariableAggregation.SUM:
        tensor = math_ops.cast(tensor / strategy.num_replicas_in_sync,
                               var.dtype)
      return assign_on_each_device(var, assign_on_device, tensor,
                                   read_value)
def scatter_sub(var, sparse_delta, use_locking=False, name=None):
  scatter_sub_fn = lambda var, *a, **kw: var.scatter_sub(*a, **kw)
      update_fn=scatter_sub_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)
def scatter_add(var, sparse_delta, use_locking=False, name=None):
  scatter_add_fn = lambda var, *a, **kw: var.scatter_add(*a, **kw)
      update_fn=scatter_add_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)
def scatter_mul(var, sparse_delta, use_locking=False, name=None):
  scatter_mul_fn = lambda var, *a, **kw: var.scatter_mul(*a, **kw)
      update_fn=scatter_mul_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)
def scatter_div(var, sparse_delta, use_locking=False, name=None):
  scatter_div_fn = lambda var, *a, **kw: var.scatter_div(*a, **kw)
      update_fn=scatter_div_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)
def scatter_min(var, sparse_delta, use_locking=False, name=None):
  scatter_min_fn = lambda var, *a, **kw: var.scatter_min(*a, **kw)
      update_fn=scatter_min_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)
def scatter_max(var, sparse_delta, use_locking=False, name=None):
  scatter_max_fn = lambda var, *a, **kw: var.scatter_max(*a, **kw)
      update_fn=scatter_max_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)
def scatter_update(var, sparse_delta, use_locking=False, name=None):
  scatter_update_fn = lambda var, *a, **kw: var.scatter_update(*a, **kw)
      update_fn=scatter_update_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)
def get_current_replica_id_as_int():
  replica_context = ds_context.get_replica_context()
  if replica_context:
    if not isinstance(replica_id, int):
      replica_id = tensor_util.constant_value(replica_id)
  else:
    replica_id = distribute_lib.get_update_replica_id()
  return replica_id
def assign_on_device(device, variable, tensor):
  with ops.device(device):
    return variable.assign(tensor)
def assign_add_on_device(device, variable, tensor):
  with ops.device(device):
    return variable.assign_add(tensor)
def assign_sub_on_device(device, variable, tensor):
  with ops.device(device):
    return variable.assign_sub(tensor)
def assert_replica_context(strategy):
  replica_context = ds_context.get_replica_context()
  if not replica_context:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")
  if replica_context.strategy is not strategy:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")
def apply_aggregation(strategy, value, aggregation, destinations):
  if aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
    return strategy.extended.broadcast_to(
        strategy.experimental_local_results(value)[0],
        destinations=destinations)
  reduce_op = reduce_util.ReduceOp.from_variable_aggregation(aggregation)
  return strategy.extended.reduce_to(reduce_op, value, destinations)
aggregation_error_msg = (
    "You must specify an aggregation method to update a "
    "{variable_type} in Replica Context. You can do so by passing "
    "an explicit value for argument `aggregation` to tf.Variable(..)."
    "e.g. `tf.Variable(..., aggregation=tf.VariableAggregation.SUM)`"
    "`tf.VariableAggregation` lists the possible aggregation methods."
    "This is required because {variable_type} should always be "
    "kept in sync. When updating them or assigning to them in a "
    "replica context, we automatically try to aggregate the values "
    "before updating the variable. For this aggregation, we need to "
    "know the aggregation method. "
    "Another alternative is to not try to update such "
    "{variable_type} in replica context, but in cross replica "
    "context. You can enter cross replica context by calling "
    "`tf.distribute.get_replica_context().merge_call(merge_fn, ..)`."
    "Inside `merge_fn`, you can then update the {variable_type} "
    "using `tf.distribute.StrategyExtended.update()`.")
scatter_error_msg = ("{op_name} is only supported for mirrored "
                     "variable (variable created within certain "
                     "`tf.distribute.Strategy` scope) with NONE or "
                     "`ONLY_FIRST_REPLICA` aggregation, got: {aggregation}.")
def is_saving_non_distributed():
  if not save_context.in_save_context():
    return False
  options = save_context.get_save_options()
  return (options.experimental_variable_policy !=
          save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES)
def mark_as_unsaveable():
  if ops.inside_function() and not save_context.in_save_context():
    ops.get_default_graph().mark_as_unsaveable("""
ConcreteFunction that uses distributed variables in certain way cannot be saved.
If you're saving with
tf.saved_model.save(..., signatures=f.get_concrete_function())
do
@tf.function(input_signature=...)
def f_with_input_signature():
  ...
tf.saved_model.save(..., signatures=f_with_input_signature)`
instead.""")
