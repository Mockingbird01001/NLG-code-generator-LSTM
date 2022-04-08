
import copy
import enum
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export("distribute.experimental.CommunicationImplementation",
           "distribute.experimental.CollectiveCommunication")
class CommunicationImplementation(enum.Enum):
  AUTO = "AUTO"
  RING = "RING"
  NCCL = "NCCL"
CollectiveCommunication = CommunicationImplementation
@tf_export("distribute.experimental.CommunicationOptions")
class _OptionsExported(object):
  """Options for cross device communications like All-reduce.
  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.
  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.
  Examples:
  ```python
  options = tf.distribute.experimental.CommunicationOptions(
      bytes_per_pack=50 * 1024 * 1024,
      timeout_seconds=120.0,
      implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
  )
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, options=options)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```
  """
  def __new__(cls, *args, **kwargs):
    return Options(*args, **kwargs)
  def __init__(self,
               bytes_per_pack=0,
               timeout_seconds=None,
               implementation=CommunicationImplementation.AUTO):
    pass
class Options(object):
  def __init__(self,
               bytes_per_pack=0,
               timeout_seconds=None,
               implementation=CommunicationImplementation.AUTO):
    if bytes_per_pack < 0:
      raise ValueError(
          f"Argument `bytes_per_pack` must be >=0, Received {bytes_per_pack}.")
    if isinstance(implementation, str):
      implementation = CommunicationImplementation(implementation.upper())
    if not isinstance(implementation, CommunicationImplementation):
      raise ValueError(
          "Argument `implementation` must be instance of "
          "`tf.distribute.experimental.CommunicationImplementation`.")
    self.bytes_per_pack = bytes_per_pack
    self.timeout_seconds = timeout_seconds
    self.implementation = implementation
  __init__.__doc__ = _OptionsExported.__init__.__doc__
  def merge(self, options):
    merged = copy.deepcopy(self)
    if options is None:
      return merged
    if options.bytes_per_pack != 0:
      merged.bytes_per_pack = options.bytes_per_pack
    if options.timeout_seconds is not None:
      merged.timeout_seconds = options.timeout_seconds
    if options.implementation != CommunicationImplementation.AUTO:
      merged.implementation = options.implementation
    return merged
  def __str__(self):
    return (f"Options(bytes_per_pack={self.bytes_per_pack},"
            f"timeout_seconds={self.timeout_seconds}, "
            f"implementation={self.implementation})")
@tf_export("distribute.experimental.CollectiveHints")
class Hints(object):
  """Hints for collective operations like AllReduce.
  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.
  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.
  Examples:
  - bytes_per_pack
  ```python
  hints = tf.distribute.experimental.CollectiveHints(
      bytes_per_pack=50 * 1024 * 1024)
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, experimental_hints=hints)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```
  - timeout_seconds
  ```python
  strategy = tf.distribute.MirroredStrategy()
  hints = tf.distribute.experimental.CollectiveHints(
      timeout_seconds=120.0)
  try:
    strategy.reduce("sum", v, axis=None, experimental_hints=hints)
  except tf.errors.DeadlineExceededError:
    do_something()
  ```
  """
  @deprecation.deprecated(
      None, "use distribute.experimental.CommunicationOptions instead")
  def __new__(cls, bytes_per_pack=0, timeout_seconds=None):
    return Options(
        bytes_per_pack=bytes_per_pack, timeout_seconds=timeout_seconds)
  def __init__(self, bytes_per_pack=0, timeout_seconds=None):
    pass
