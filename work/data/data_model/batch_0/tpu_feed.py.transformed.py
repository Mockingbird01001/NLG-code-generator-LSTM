
import itertools
import numpy as np
from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
def partition_or_replicate_on_host(tensor, dims):
  if dims is None:
    return itertools.repeat(tensor)
  dims = np.array(dims)
  output = [tensor]
  shape_list = np.array(tensor.shape.as_list())
  quotients, remainders = np.divmod(shape_list, dims)
  for axis, (quotient, remainder, dim, original_size) in enumerate(
      zip(quotients, remainders, dims, shape_list)):
    if dim <= 1:
      continue
    if remainder > 0:
      ceil_ratio = quotient + 1
      num_full_slots, left_over = np.divmod(original_size, ceil_ratio)
      num_or_size_splits = [ceil_ratio] * num_full_slots + [left_over]
      if len(num_or_size_splits) < dim:
        num_or_size_splits += [0] * (dim - len(num_or_size_splits))
      new_output = []
      for x in output:
        new_output.append(
            array_ops.split(
                x, num_or_size_splits=num_or_size_splits, axis=axis))
      output = new_output
    else:
      output = [array_ops.split(x, int(dim), axis=axis) for x in output]
    output = nest.flatten(output)
  return output
def _tag_sharding_attribute_for_dequeued_tensor(tensor, dims):
  if dims is None:
    return xla_sharding.replicate(tensor, assign_tuple_sharding=True)
  elif np.prod(dims) == 1:
    return xla_sharding.assign_device(tensor, 0, assign_tuple_sharding=True)
  else:
    tile_assignment = np.arange(np.prod(dims)).reshape(dims)
    return xla_sharding.tile(
        tensor=tensor,
        tile_assignment=tile_assignment,
        assign_tuple_sharding=True)
def tag_sharding_attribute_for_dequeued_tensors(dequeues, dims):
  nest.assert_shallow_structure(dequeues, dims)
  return nest.map_structure_up_to(
      dequeues, _tag_sharding_attribute_for_dequeued_tensor, dequeues, dims)
class InfeedQueue(object):
  def __init__(self,
               number_of_tuple_elements=None,
               tuple_types=None,
               tuple_shapes=None,
               shard_dimensions=None,
               number_of_partitions=None,
               name=None):
    self._frozen = False
    self._generated_enqueue_ops = False
    self._generated_dequeue_op = False
    self._name = "InfeedQueue" if name is None else name
    if number_of_partitions is None:
      self._number_of_partitions = 1
    else:
      self._number_of_partitions = number_of_partitions
    if number_of_tuple_elements is None:
      if tuple_types is not None:
        number_of_tuple_elements = len(tuple_types)
      elif tuple_shapes is not None:
        number_of_tuple_elements = len(tuple_shapes)
      elif shard_dimensions is not None:
        number_of_tuple_elements = len(shard_dimensions)
      else:
        raise ValueError(
            "number of tuple elements cannot be inferred from InfeedQueue "
            "constructor")
    if number_of_tuple_elements <= 0:
      raise ValueError(f"number_of_tuple_elements {number_of_tuple_elements} "
                       "must be > 0")
    self._sharding_policies = [
        tpu_sharding.ShardingPolicy() for _ in range(number_of_tuple_elements)
    ]
    if tuple_types is not None:
      self.set_tuple_types(tuple_types)
    else:
      self._tuple_types = None
    if tuple_shapes is not None:
      self.set_tuple_shapes(tuple_shapes)
    else:
      self._tuple_shapes = None
    if shard_dimensions is not None:
      self.set_shard_dimensions(shard_dimensions)
    self._validate()
  def _validate(self):
    if self.tuple_shapes is not None:
      for (policy, shape) in zip(self._sharding_policies, self._tuple_shapes):
        _ = policy.get_sharded_shape(shape)
  @property
  def number_of_tuple_elements(self):
    return len(self._sharding_policies)
  @property
  def tuple_types(self):
    return self._tuple_types
  def set_tuple_types(self, tuple_types):
    if len(tuple_types) != self.number_of_tuple_elements:
      raise ValueError(
          f"tuple_types is {str(tuple_types)}, but must be a list of "
          f"length {self.number_of_tuple_elements}"
      )
    if self._frozen:
      for (frozen, updated) in zip(self._tuple_types, tuple_types):
        if frozen != updated:
          raise ValueError(
              "Trying to update InfeedQueue with frozen configuration with an "
              f"incompatible type. Frozen types are {str(self._tuple_types)}, "
              f"updated types are {str(tuple_types)}")
    else:
      try:
        self._tuple_types = [dtypes.as_dtype(t) for t in tuple_types]
      except (TypeError) as e:
        raise TypeError(
            f"tuple_types is {str(tuple_types)}, but must be a list of "
            f"elements each convertible to dtype: got error {str(e)}") from e
  @property
  def tuple_shapes(self):
    return self._tuple_shapes
  def set_tuple_shapes(self, tuple_shapes):
    if len(tuple_shapes) != self.number_of_tuple_elements:
      raise ValueError(
          f"tuple_shapes is {str(tuple_shapes)}, but must be a list of "
          f"length {self.number_of_tuple_elements}"
      )
    try:
      tuple_shapes = [tensor_shape.as_shape(shape) for shape in tuple_shapes]
    except (ValueError, TypeError) as e:
      raise TypeError(
          f"tuple_shapes is {str(tuple_shapes)}, but must be a list of "
          "elements each convertible to TensorShape: got error "
          f"{str(e)}") from e
    if self._frozen:
      for (frozen, updated) in zip(self._tuple_shapes, tuple_shapes):
        if frozen != updated:
          raise ValueError(
              "Trying to update InfeedQueue with frozen configuration with an "
              "incompatible shape. Frozen shapes are "
              f"{str(self._tuple_shapes)}, updated shapes are "
              f"{str(tuple_shapes)}")
    else:
      self._tuple_shapes = tuple_shapes
    self._validate()
  @property
  def sharding_policies(self):
    return self._sharding_policies
  @property
  def shard_dimensions(self):
    return [policy.shard_dimension for policy in self._sharding_policies]
  def set_shard_dimensions(self, shard_dimensions):
    if len(shard_dimensions) != self.number_of_tuple_elements:
      raise ValueError(f"shard_dimensions is {str(shard_dimensions)}, but must "
                       f"be a list of length {self.number_of_tuple_elements}")
    for (policy, dimension) in zip(self._sharding_policies, shard_dimensions):
      policy.set_shard_dimension(dimension)
    self._validate()
  @property
  def number_of_shards(self):
    return self._sharding_policies[0].number_of_shards
  def set_number_of_shards(self, number_of_shards):
    for policy in self._sharding_policies:
      policy.set_number_of_shards(number_of_shards)
      policy.set_number_of_partitions(self._number_of_partitions)
    self._validate()
  def set_configuration_from_input_tensors(self, input_tensors):
    if len(input_tensors) != self.number_of_tuple_elements:
      raise ValueError(f"input_tensors is {str(input_tensors)}, but should be "
                       f"a list of {self.number_of_tuple_elements} Tensors")
    self.set_tuple_shapes([t.shape for t in input_tensors])
    self.set_tuple_types([t.dtype for t in input_tensors])
  def set_configuration_from_sharded_input_tensors(self, input_tensors):
    if not self._frozen:
      self._tuple_shapes = None
    number_of_shards = len(input_tensors)
    self.set_number_of_shards(number_of_shards)
    for t in input_tensors:
      if len(t) != self.number_of_tuple_elements:
        raise ValueError(
            f"input_tensors is {str(input_tensors)} but must be a list of "
            "lists, where each inner list has length "
            f"number_of_tuple_elements={self.number_of_tuple_elements}")
    sharded_shapes = [[t[i].shape
                       for t in input_tensors]
                      for i in range(self.number_of_tuple_elements)]
    unsharded_shapes = [
        policy.get_unsharded_shape(s)
        for (policy, s) in zip(self._sharding_policies, sharded_shapes)
    ]
    self.set_tuple_shapes(unsharded_shapes)
    for i in range(1, self.number_of_shards):
      for (t1, t2) in zip(input_tensors[0], input_tensors[i]):
        if t1.dtype != t2.dtype:
          raise TypeError(
              "types of the tuple elements of input_tensors "
              f"{str(input_tensors)} are not consistent")
    self.set_tuple_types([t.dtype for t in input_tensors[0]])
  def freeze(self):
    self._frozen = True
    if self._tuple_types is None:
      raise ValueError(
          "Can't freeze an InfeedQueue without setting all tuple types.")
    if self._tuple_shapes is None:
      raise ValueError(
          "Can't freeze an InfeedQueue without setting all tuple shapes.")
    for shape in self._tuple_shapes:
      if shape.dims is None:
        raise ValueError(
            "Can't freeze an InfeedQueue without setting all tuple shapes.")
    for policy in self._sharding_policies:
      policy.freeze()
    self._validate()
  def generate_dequeue_op(self, tpu_device=0):
    self.freeze()
    if self._generated_dequeue_op and not ops.inside_function():
      raise ValueError("Can't generate two dequeue Ops from the same queue")
    self._generated_dequeue_op = True
    full_name = "%s/dequeue" % self._name
    sharded_shapes = [
        policy.get_unpartitioned_shape(policy.get_sharded_shape(shape))
        for (shape, policy) in zip(self._tuple_shapes, self._sharding_policies)
    ]
    if tpu_device is not None:
      with ops.device(tpu_name_util.core(tpu_device)):
        dequeue_op = tpu_ops.infeed_dequeue_tuple(
            dtypes=self._tuple_types, shapes=sharded_shapes, name=full_name)
    else:
      dequeue_op = tpu_ops.infeed_dequeue_tuple(
          dtypes=self._tuple_types, shapes=sharded_shapes, name=full_name)
    if self._number_of_partitions <= 1:
      return dequeue_op
    partitions = [
        policy.get_unpartitioned_shape([1] * shape.ndims).as_list()
        for (shape, policy) in zip(self._tuple_shapes, self._sharding_policies)
    ]
    return tag_sharding_attribute_for_dequeued_tensors(dequeue_op, partitions)
  def _generate_enqueue_op(self,
                           inputs,
                           name_prefix,
                           index,
                           device=None,
                           tpu_ordinal=-1):
    full_name = "%s/%d" % (name_prefix, index)
    shapes = [t.shape for t in inputs]
    if device is None:
      devices = [t.device for t in inputs]
      for i in range(1, self.number_of_tuple_elements):
        if devices[0] != devices[i]:
          raise ValueError(
              f"input devices for shard {index} are {str(devices)}, but should "
              "all be the same")
      with ops.colocate_with(inputs[0]):
        return tpu_ops.infeed_enqueue_tuple(
            inputs=inputs,
            shapes=shapes,
            name=full_name,
            device_ordinal=tpu_ordinal)
    else:
      with ops.device(device):
        return tpu_ops.infeed_enqueue_tuple(
            inputs=inputs,
            shapes=shapes,
            name=full_name,
            device_ordinal=tpu_ordinal)
  def generate_enqueue_ops(self,
                           sharded_inputs,
                           tpu_ordinal_function=None,
                           placement_function=None):
    self.set_configuration_from_sharded_input_tensors(sharded_inputs)
    self.freeze()
    if self._generated_enqueue_ops and not ops.inside_function():
      raise ValueError("Can't generate two enqueue Ops from the same queue")
    self._generated_enqueue_ops = True
    if tpu_ordinal_function is None:
      tpu_ordinal_function = lambda index: -1
    name_prefix = "%s/enqueue" % self._name
    return [
        self._generate_enqueue_op(
            shard,
            name_prefix,
            index,
            tpu_ordinal=tpu_ordinal_function(index),
            device=placement_function(index) if placement_function else None)
        for (shard, index) in zip(sharded_inputs, range(self.number_of_shards))
    ]
  def _default_placement_function(self, index):
    return "/task:%d/device:CPU:0" % (index / 8)
  def _default_ordinal_function(self, index):
    return index % 8
  def split_inputs_and_generate_enqueue_ops(self,
                                            inputs,
                                            device_assignment=None,
                                            placement_function=None,
                                            tpu_ordinal_function=None):
    """POORLY-PERFORMING ON MULTI-HOST SYSTEMS.
    Generates the host-side Ops to enqueue a tuple.
    This method performs poorly because it takes an entire input on a single
    host, splits it, and distributes it to all of the cores. It is present only
    to simplify tutorial examples.
    inputs is a list of Tensors to use to feed the queue. Each input is split
    into self.number_of_shards shards. Returns an Op for each shard to enqueue
    the shard. The Op for shard i is placed on device placement_function(i).
    Implicitly freezes the queue configuration if it is not already
    frozen. If the configuration has already been frozen, and is not
    compatible with the types and shapes of inputs, an error
    will be raised.
    Args:
      inputs: a list of Tensors which indicates the types and shapes of the
        queue tuple.
     device_assignment: if not `None`, a TPU `DeviceAssignment`. If
        device_assignment is not `None`, but `placement_function` and
        `ordinal_function` are None, then `device_assignment` will be used to
        place infeeds on the first k TPU shards, where k is the number of shards
        in the queue. If all three are `None`, then default placement and
        ordinal functions are used.
      placement_function: if not None, a function that takes the shard
        index as input and returns a device string indicating which
        device the shard's infeed should be placed on. If placement_function
        and tpu_ordinal_function are None, inputs are sharded round-robin
        across the devices in the system.
      tpu_ordinal_function: if not None, a function that takes the
        shard index as input and returns the ordinal of the TPU device
        the shard's infeed should be placed on. If placement_function
        and tpu_ordinal_function are None, inputs are sharded round-robin
        across the devices in the system.
    Returns:
      A list of host-side Ops, one for each shard, that when executed together
      will enqueue a full-size element of infeed.
    Raises:
      ValueError: if the queue configuration has previously been frozen and the
        shapes of the elements of inputs are not compatible with the frozen
        configuration.
      TypeError: if the queue configuration has previously been frozen and the
        types of the elements of inputs are not compatible with the frozen
        configuration.
    """
    if device_assignment is None:
      if placement_function is None:
        placement_function = self._default_placement_function
      if tpu_ordinal_function is None:
        tpu_ordinal_function = self._default_ordinal_function
    else:
      def _placement_function_from_map(index):
        return device_assignment.host_device(replica=index)
      def _ordinal_function_from_map(index):
        return device_assignment.tpu_ordinal(replica=index)
      if placement_function is None:
        placement_function = _placement_function_from_map
      if tpu_ordinal_function is None:
        tpu_ordinal_function = _ordinal_function_from_map
    self.set_configuration_from_input_tensors(inputs)
    self.freeze()
    if self._generated_enqueue_ops and not ops.inside_function():
      raise ValueError("Can't generate two enqueue Ops from the same queue")
    self._generated_enqueue_ops = True
    split_name_prefix = "%s/split" % self._name
    if self.number_of_shards == 1:
      transposed_sharded_inputs = [[inp] for inp in inputs]
    else:
      def split_fn(inp, num_shards, axis, name):
        with ops.colocate_with(inp):
          return array_ops.split(inp, num_shards, axis=axis, name=name)
      transposed_sharded_inputs = [
          split_fn(
              inp,
              self.number_of_shards,
              axis=policy.shard_dimension,
              name="%s/%d" % (split_name_prefix, index))
          for (inp, policy, index) in zip(inputs, self._sharding_policies,
                                          range(self.number_of_tuple_elements))
      ]
    sharded_inputs = [[shard[i]
                       for shard in transposed_sharded_inputs]
                      for i in range(self.number_of_shards)]
    name_prefix = "%s/enqueue" % self._name
    return [
        self._generate_enqueue_op(
            shard,
            name_prefix,
            index,
            device=placement_function(index),
            tpu_ordinal=tpu_ordinal_function(index))
        for (shard, index) in zip(sharded_inputs, range(self.number_of_shards))
    ]
class _PartitionedInfeedQueue(InfeedQueue):
  def __init__(self,
               number_of_tuple_elements,
               device_assignment,
               host_id,
               input_partition_dims=None,
               tuple_types=None,
               tuple_shapes=None,
               name=None):
    super(_PartitionedInfeedQueue, self).__init__(
        number_of_tuple_elements=number_of_tuple_elements,
        tuple_types=tuple_types,
        tuple_shapes=None,
        shard_dimensions=None,
        name="PartitionedInfeedQueue" if name is None else name)
    self._input_partition_dims = input_partition_dims
    self._host_id = host_id
    self._device_assignment = device_assignment
  def generate_dequeue_op(self, tpu_device=0):
    self.freeze()
    if self._generated_dequeue_op and not ops.inside_function():
      raise ValueError("Can't generate two dequeue Ops from the same queue")
    self._generated_dequeue_op = True
    full_name = "%s/dequeue" % self._name
    sharded_shapes = [
        policy.get_sharded_shape(shape)
        for (shape, policy) in zip(self._tuple_shapes, self._sharding_policies)
    ]
    with ops.device(tpu_name_util.core(tpu_device)):
      values = tpu_ops.infeed_dequeue_tuple(
          dtypes=self._tuple_types, shapes=sharded_shapes, name=full_name)
    return tag_sharding_attribute_for_dequeued_tensors(
        values, self._input_partition_dims)
  def generate_enqueue_ops(self, sharded_inputs):
    self.set_configuration_from_sharded_input_tensors(sharded_inputs)
    number_of_replicas = len(sharded_inputs)
    number_of_tuple_elements = len(sharded_inputs[0])
    assert len(self._input_partition_dims) == number_of_tuple_elements
    enqueue_ops = []
    for replica_index in range(number_of_replicas):
      flattened_inputs = sharded_inputs[replica_index]
      inputs_part_dims_flat = nest.flatten_up_to(flattened_inputs,
                                                 self._input_partition_dims)
      inputs_parted_iters = [
          iter(self._check_dims_and_partition_or_replicate_on_host(x, dims))
          for x, dims in zip(sharded_inputs[replica_index],
                             inputs_part_dims_flat)
      ]
      replica_id = self._device_assignment.lookup_replicas(
          task_id=self._host_id, logical_core=0)[replica_index]
      for logical_core in range(self._device_assignment.num_cores_per_replica):
        device = self._device_assignment.host_device(
            replica=replica_id, logical_core=logical_core)
        with ops.device(device):
          ordinal = self._device_assignment.tpu_ordinal(
              replica=replica_id, logical_core=logical_core)
          infeed_inputs = []
          for it in inputs_parted_iters:
            input_for_device = next(it, None)
            if input_for_device is not None:
              infeed_inputs.append(input_for_device)
          if infeed_inputs:
            enqueue_ops.append(
                tpu_ops.infeed_enqueue_tuple(
                    inputs=infeed_inputs,
                    shapes=[x.shape for x in infeed_inputs],
                    name="enqueue/replica_{0}/input_{1}".format(
                        replica_index, logical_core),
                    device_ordinal=ordinal))
    return enqueue_ops
  def _check_input_partition_dims(self, tensor, dims):
    """Checks that input partition dims are valid for the `Tensor`.
    Args:
      tensor: Input tensor for partitioning.
      dims: A list of integer describes how to partition the input tensor.
    Raises:
      ValueError: If the tensor can't be partitioned by dims or the
        num_cores_per_replica doesn't match the number of
        partitions(dims.prod()).
    """
    if dims is None:
      return
    dims = np.array(dims)
    if (dims < 1).any():
      raise ValueError("All input partition dims must be >= 1.")
    if dims.prod() == 1:
      return
    if dims.prod() != self._device_assignment.num_cores_per_replica:
      raise ValueError(
          "The product of each input partition dim should equal to "
          "num_cores_per_replica. (dim = {}, num_cores_per_replica "
          "= {})".format(dims, self._device_assignment.num_cores_per_replica))
    if dims.shape[0] != tensor.shape.ndims:
      raise ValueError(
          "Input partition dims must have the same number of dimensions "
          "as the `Tensor` to be partitioned. (tensor shape = {}, input "
          "partition dims = {}).".format(tensor.shape.as_list(), dims))
    tensor.shape.assert_is_fully_defined()
  def _check_dims_and_partition_or_replicate_on_host(self, tensor, dims):
    self._check_input_partition_dims(tensor, dims)
    return partition_or_replicate_on_host(tensor, dims)
