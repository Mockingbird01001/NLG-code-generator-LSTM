
from tensorflow.python.framework import tensor_shape
_DEFAULT_NUMBER_OF_SHARDS = 1
_DEFAULT_SHARD_DIMENSION = 0
class ShardingPolicy(object):
  def __init__(self):
    self._number_of_shards = None
    self._number_of_partitions = 1
    self._shard_dimension = None
    self._frozen = False
  def __str__(self):
    if self.number_of_shards is None or self.shard_dimension is None:
      return "ShardingPolicy(unset)"
    else:
      return ("ShardingPolicy(%d shards dimension %d)" %
              (self.number_of_shards, self.shard_dimension))
  def _fill_default_values(self):
    if self._number_of_shards is None:
      self._number_of_shards = _DEFAULT_NUMBER_OF_SHARDS
    if self._shard_dimension is None:
      self._shard_dimension = tensor_shape.as_dimension(
          _DEFAULT_SHARD_DIMENSION)
  def freeze(self):
    if not self._frozen:
      self._fill_default_values()
      self._frozen = True
  @property
  def number_of_shards(self):
    return self._number_of_shards
  def set_number_of_shards(self, number_of_shards):
    if self._frozen:
      if self._number_of_shards != number_of_shards:
        raise ValueError(
            f"Can't set sharding policy to use {number_of_shards} shards since "
            f"it has been frozen to use {self._number_of_shards}")
    else:
      if number_of_shards > 0:
        self._number_of_shards = number_of_shards
      else:
        raise ValueError(
            "Can't set sharding policy to use {number_of_shards} shards; "
            "value must be > 0")
  @property
  def number_of_partitions(self):
    return self._number_of_partitions
  def set_number_of_partitions(self, number_of_partitions):
    if self._frozen:
      if self._number_of_partitions != number_of_partitions:
        raise ValueError(
            f"Can't set number_of_partitions to {number_of_partitions} since "
            f"it has been frozen to use {self._number_of_partitions}.")
    else:
      self._number_of_partitions = number_of_partitions
  @property
  def shard_dimension(self):
    return self._shard_dimension
  def set_shard_dimension(self, shard_dimension):
    if self._frozen:
      if self._shard_dimension != shard_dimension:
        raise ValueError(
            "Can't set shard dimension to %d since it has been frozen to "
            "use %d." % (shard_dimension, self._shard_dimension))
    else:
      self._shard_dimension = tensor_shape.as_dimension(shard_dimension)
  def merge(self, other):
    if other.number_of_shards is not None:
      self.set_number_of_shards(other.number_of_shards)
    if other.shard_dimension is not None:
      self.set_shard_dimension(other.shard_dimension)
  def get_unpartitioned_shape(self, shape):
    shape = tensor_shape.as_shape(shape)
    dims = shape.as_list()
    if (self._shard_dimension is None or self._number_of_partitions is None or
        not dims):
      return None
    if dims[self._shard_dimension] is None:
      raise ValueError(f"Shape {shape.as_list()} must have a fixed size for "
                       f"dimension {self._shard_dimension} that is known. ")
    if self._number_of_partitions > 1:
      dims[self._shard_dimension] *= self._number_of_partitions
    return tensor_shape.as_shape(dims)
  def get_sharded_shape(self, shape, shard_index=None):
    """Returns the shape of a shard of a full Tensor.
    When given the shape of a 'full-size' Tensor, returns the shape of
    the sub-Tensor after it has been sharded. Freezes the policy if it
    has not yet been frozen.
    Args:
      shape: The shape of the full-size Tensor to be sharded.
      shard_index: The index of the shard whose shape should be returned.
        shard_index can be None for sharding policies that use the same shape
        for every shard.
    Returns:
      The shape of the sharded version of the Tensor.
    Raises:
      ValueError: If shard_index is None when shards are of different
        shapes; or shard_index is not None and
        !(0<=shard_index<number_of_shards); or shape does not have at
        least self.shard_dimension+1 dimensions; or the value of
        shape's shard dimension is not a multiple of
        self.number_of_shards
    """
    if self._shard_dimension is None or self._number_of_shards is None:
      return None
    if shard_index is not None:
      if shard_index < 0 or shard_index >= self.number_of_shards:
        raise ValueError(
            f"Requested shard_index {shard_index}, but shard_index must be in "
            f"[0,{self._number_of_shards}).")
    shape = tensor_shape.as_shape(shape)
    if self._number_of_shards == 1:
      return shape
    ndims = shape.ndims
    if ndims is None:
      raise ValueError(f"Shape {shape} must be a known shape.")
    if ndims <= self._shard_dimension:
      raise ValueError(
          f"Shape {shape.as_list()} does not contain shard_dimension "
          f"{self._shard_dimension}")
    dims = shape.as_list()
    if dims[self._shard_dimension] is None:
      raise ValueError(
          f"Shape {shape.as_list()} must have a fixed size for dimension "
          f"{self._shard_dimension} that is known at construction time.")
    if (dims[self._shard_dimension] % self._number_of_shards) != 0:
      raise ValueError(
          f"Shape {shape.as_list()} cannot be sharded {self._number_of_shards} "
          f"ways along dimension {self._shard_dimension}")
    dims[self._shard_dimension] //= self._number_of_shards
    return tensor_shape.TensorShape(dims)
  def _unshard_shape(self, shape):
    shape = tensor_shape.as_shape(shape)
    if self._number_of_shards == 1:
      return shape
    ndims = shape.ndims
    if ndims is None:
      raise ValueError(f"Shape {shape} must be statically known.")
    if ndims <= self._shard_dimension:
      raise ValueError(f"Shape {shape.as_list()} does not contain "
                       f"shard_dimension {self._shard_dimension}. "
                       f"Rank is too small.")
    dims = shape.as_list()
    dims[self._shard_dimension] *= self._number_of_shards
    return tensor_shape.TensorShape(dims)
  def get_unsharded_shape(self, shapes):
    self._fill_default_values()
    if len(shapes) != self.number_of_shards:
      raise ValueError(
          f"Shapes {shapes} is length {len(shapes)} but must be a list of "
          f"length number_of_shards={self.number_of_shards}")
    unsharded_shapes = [self._unshard_shape(s) for s in shapes]
    for i in range(self.number_of_shards - 1):
      if not unsharded_shapes[i].is_compatible_with(
          unsharded_shapes[self.number_of_shards - 1]):
        raise ValueError(
            f"Sharded shapes {shapes} are not consistent shards of a full shape "
            f"sharded {self.number_of_shards} ways along "
            f"dimension {self.shard_dimension}.")
    return unsharded_shapes[0]
