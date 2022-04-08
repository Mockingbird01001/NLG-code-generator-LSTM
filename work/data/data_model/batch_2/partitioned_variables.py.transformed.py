
"""Helper functions for creating partitioned variables.
This is a convenient abstraction to partition a large variable across
multiple smaller variables that can be assigned to different devices.
The full variable can be reconstructed by concatenating the smaller variables.
Using partitioned variables instead of a single variable is mostly a
performance choice.  It however also has an impact on:
1. Random initialization, as the random number generator is called once per
   slice
2. Updates, as they happen in parallel across slices
A key design goal is to allow a different graph to repartition a variable
with the same name but different slicings, including possibly no partitions.
TODO(touts): If an initializer provides a seed, the seed must be changed
deterministically for each slice, maybe by adding one to it, otherwise each
slice will use the same values.  Maybe this can be done by passing the
slice offsets to the initializer functions.
Typical usage:
```python
vs = create_partitioned_variables(
    <shape>, <slicing>, <initializer>, name=<optional-name>)
y = embedding_lookup(vs, ids, partition_strategy="div")
z = matmul(x, concat(slice_dim, vs))
```
"""
import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
__all__ = [
    "create_partitioned_variables",
    "variable_axis_size_partitioner",
    "min_max_variable_partitioner",
    "fixed_size_partitioner",
]
@tf_export(v1=["variable_axis_size_partitioner"])
def variable_axis_size_partitioner(
    max_shard_bytes, axis=0, bytes_per_string_element=16, max_shards=None):
  """Get a partitioner for VariableScope to keep shards below `max_shard_bytes`.
  This partitioner will shard a Variable along one axis, attempting to keep
  the maximum shard size below `max_shard_bytes`.  In practice, this is not
  always possible when sharding along only one axis.  When this happens,
  this axis is sharded as much as possible (i.e., every dimension becomes
  a separate shard).
  If the partitioner hits the `max_shards` limit, then each shard may end up
  larger than `max_shard_bytes`. By default `max_shards` equals `None` and no
  limit on the number of shards is enforced.
  One reasonable value for `max_shard_bytes` is `(64 << 20) - 1`, or almost
  `64MB`, to keep below the protobuf byte limit.
  Args:
    max_shard_bytes: The maximum size any given shard is allowed to be.
    axis: The axis to partition along.  Default: outermost axis.
    bytes_per_string_element: If the `Variable` is of type string, this provides
      an estimate of how large each scalar in the `Variable` is.
    max_shards: The maximum number of shards in int created taking precedence
      over `max_shard_bytes`.
  Returns:
    A partition function usable as the `partitioner` argument to
    `variable_scope` and `get_variable`.
  Raises:
    ValueError: If any of the byte counts are non-positive.
  """
  if max_shard_bytes < 1 or bytes_per_string_element < 1:
    raise ValueError(
        "Both max_shard_bytes and bytes_per_string_element must be positive. "
        f"Currently, max_shard_bytes is {max_shard_bytes} and"
        f"bytes_per_string_element is {bytes_per_string_element}")
  if max_shards and max_shards < 1:
    raise ValueError(
        "max_shards must be positive.")
  def _partitioner(shape, dtype):
    if not isinstance(shape, tensor_shape.TensorShape):
      raise ValueError(f"shape is not a TensorShape: {shape}")
    if not shape.is_fully_defined():
      raise ValueError(f"shape is not fully defined: {shape}")
    if not isinstance(dtype, dtypes.DType):
      raise ValueError(f"dtype is not a DType: {dtype}")
    if dtype.base_dtype == dtypes.string:
      element_size = bytes_per_string_element
    else:
      element_size = dtype.size
    partitions = [1] * shape.ndims
    bytes_per_slice = 1.0 * (
        shape.num_elements() / shape.dims[axis].value) * element_size
    slices_per_shard = max(1, math.floor(max_shard_bytes / bytes_per_slice))
    axis_shards = int(math.ceil(
        1.0 * shape.dims[axis].value / slices_per_shard))
    if max_shards:
      axis_shards = min(max_shards, axis_shards)
    partitions[axis] = axis_shards
    return partitions
  return _partitioner
@tf_export(v1=["min_max_variable_partitioner"])
def min_max_variable_partitioner(max_partitions=1, axis=0,
                                 min_slice_size=256 << 10,
                                 bytes_per_string_element=16):
  """Partitioner to allocate minimum size per slice.
  Returns a partitioner that partitions the variable of given shape and dtype
  such that each partition has a minimum of `min_slice_size` slice of the
  variable. The maximum number of such partitions (upper bound) is given by
  `max_partitions`.
  Args:
    max_partitions: Upper bound on the number of partitions. Defaults to 1.
    axis: Axis along which to partition the variable. Defaults to 0.
    min_slice_size: Minimum size of the variable slice per partition. Defaults
      to 256K.
    bytes_per_string_element: If the `Variable` is of type string, this provides
      an estimate of how large each scalar in the `Variable` is.
  Returns:
    A partition function usable as the `partitioner` argument to
    `variable_scope` and `get_variable`.
  """
  def _partitioner(shape, dtype):
    """Partitioner that partitions list for a variable of given shape and type.
    Ex: Consider partitioning a variable of type float32 with
      shape=[1024, 1024].
      If `max_partitions` >= 16, this function would return
        [(1024 * 1024 * 4) / (256 * 1024), 1] = [16, 1].
      If `max_partitions` < 16, this function would return
        [`max_partitions`, 1].
    Args:
      shape: Shape of the variable.
      dtype: Type of the variable.
    Returns:
      List of partitions for each axis (currently only one axis can be
      partitioned).
    Raises:
      ValueError: If axis to partition along does not exist for the variable.
    """
    if axis >= len(shape):
      raise ValueError(
          f"Cannot partition variable along axis {axis} when shape is "
          f"only {shape}")
    if dtype.base_dtype == dtypes.string:
      bytes_per_element = bytes_per_string_element
    else:
      bytes_per_element = dtype.size
    total_size_bytes = shape.num_elements() * bytes_per_element
    partitions = total_size_bytes / min_slice_size
    partitions_list = [1] * len(shape)
    partitions_list[axis] = max(1, min(shape.dims[axis].value,
                                       max_partitions,
                                       int(math.ceil(partitions))))
    return partitions_list
  return _partitioner
@tf_export(v1=["fixed_size_partitioner"])
def fixed_size_partitioner(num_shards, axis=0):
  """Partitioner to specify a fixed number of shards along given axis.
  @compatibility(TF2)
  This API is deprecated in TF2. In TF2, partitioner is no longer part of
  the variable declaration via `tf.Variable`.
  [ParameterServer Training]
  (https://www.tensorflow.org/tutorials/distribute/parameter_server_training)
  handles partitioning of variables. The corresponding TF2 partitioner class of
  `fixed_size_partitioner` is
  `tf.distribute.experimental.partitioners.FixedShardsPartitioner`.
  Check the [migration guide]
  on the differences in treatment of variables and losses between TF1 and TF2.
  Before:
    ```
    x = tf.compat.v1.get_variable(
      "x", shape=(2,), partitioner=tf.compat.v1.fixed_size_partitioner(2)
    )
    ```
  After:
    ```
    partitioner = (
        tf.distribute.experimental.partitioners.FixedShardsPartitioner(
            num_shards=2)
    )
    strategy = tf.distribute.experimental.ParameterServerStrategy(
                   cluster_resolver=cluster_resolver,
                   variable_partitioner=partitioner)
    with strategy.scope():
      x = tf.Variable([1.0, 2.0])
    ```
  @end_compatibility
  Args:
    num_shards: `int`, number of shards to partition variable.
    axis: `int`, axis to partition on.
  Returns:
    A partition function usable as the `partitioner` argument to
    `variable_scope` and `get_variable`.
  """
  def _partitioner(shape, **unused_args):
    partitions_list = [1] * len(shape)
    partitions_list[axis] = min(num_shards, shape.dims[axis].value)
    return partitions_list
  return _partitioner
@tf_export(v1=["create_partitioned_variables"])
@deprecation.deprecated(
    date=None,
    instructions="Use `tf.get_variable` with a partitioner set.")
def create_partitioned_variables(
    shape, slicing, initializer, dtype=dtypes.float32,
    trainable=True, collections=None, name=None, reuse=None):
  if len(shape) != len(slicing):
    raise ValueError(
        "The 'shape' and 'slicing' of a partitioned Variable "
        f"must have the length: shape: {shape}, slicing: {slicing}")
  if len(shape) < 1:
    raise ValueError("A partitioned Variable must have rank at least 1: "
                     f"shape: {shape}")
  partitioner = lambda **unused_kwargs: slicing
  with variable_scope.variable_scope(
      name, "PartitionedVariable", reuse=reuse):
    partitioned_var = variable_scope._get_partitioned_variable(
        name=None,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        trainable=trainable,
        partitioner=partitioner,
        collections=collections)
    return list(partitioned_var)
