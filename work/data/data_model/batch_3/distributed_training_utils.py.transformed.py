
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.keras import backend
from tensorflow.python.ops import variables
def global_batch_size_supported(distribution_strategy):
def call_replica_local_fn(fn, *args, **kwargs):
  strategy = None
  if 'strategy' in kwargs:
    strategy = kwargs.pop('strategy')
  else:
    if ds_context.has_strategy():
      strategy = ds_context.get_strategy()
  is_tpu = backend.is_tpu_strategy(strategy)
  if ((not is_tpu) and strategy and ds_context.in_cross_replica_context()):
    with strategy.scope():
      return strategy.extended.call_for_each_replica(fn, args, kwargs)
  return fn(*args, **kwargs)
def is_distributed_variable(v):
  return (isinstance(v, values_lib.DistributedValues) and
          isinstance(v, variables.Variable))
