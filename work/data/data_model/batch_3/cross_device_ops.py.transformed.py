
import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def check_destinations(destinations):
  if isinstance(destinations,
                (resource_variable_ops.BaseResourceVariable, ops.Tensor)):
    return bool(destinations.device)
  return bool(destinations)
def validate_destinations(destinations):
  if not isinstance(
      destinations,
      (value_lib.DistributedValues, ops.Tensor, indexed_slices.IndexedSlices,
       ps_values.AggregatingVariable, six.string_types,
       tpu_values.TPUMirroredVariable
      )) and not resource_variable_ops.is_resource_variable(destinations):
    raise ValueError("destinations must be one of a `DistributedValues` object,"
                     " a tf.Variable object, or a device string.")
  if not check_destinations(destinations):
    raise ValueError("destinations can not be empty")
def reduce_non_distributed_value(reduce_op,
                                 value,
                                 destinations,
                                 num_replicas_in_graph,
                                 canonicalize_devices=True):
  if isinstance(value, value_lib.DistributedValues):
    raise ValueError("You are passing a `DistributedValues` to "
                     "`reduce_non_distributed_value`, which is not allowed.")
  if not tensor_util.is_tf_type(value) and value == 0:
    return 0
  if reduce_op == reduce_util.ReduceOp.MEAN:
    return value
  elif num_replicas_in_graph != 1:
    raise ValueError("A non-DistributedValues value %s cannot be reduced with "
                     "the given reduce op %s." % (value, reduce_op))
  else:
    validate_destinations(destinations)
    return simple_broadcast(
        value, destinations, canonicalize_devices=canonicalize_devices)
def _make_tensor_into_per_replica(input_tensor):
  if isinstance(input_tensor, (tuple, list)):
    raise ValueError("Cannot convert `input_tensor` to a `PerReplica` object, "
                     "got %r but expected a object that is not a tuple or list."
                     % (input_tensor,))
  if isinstance(input_tensor, value_lib.PerReplica):
    return input_tensor
  elif hasattr(input_tensor, "device"):
    return value_lib.PerReplica((input_tensor,))
  else:
    raise ValueError("Cannot convert `input_tensor` to a `PerReplica` object "
                     "because it doesn't have device set.")
def _normalize_value_destination_pairs(value_destination_pairs):
  result = []
  value_destination_pairs = list(value_destination_pairs)
  if not isinstance(value_destination_pairs, (list, tuple)):
    raise ValueError("`value_destination_pairs` should be a list or tuple")
  for pair in value_destination_pairs:
    if not isinstance(pair, tuple):
      raise ValueError(
          "Each element of `value_destination_pairs` should be a tuple.")
    if len(pair) != 2:
      raise ValueError("Each element of `value_destination_pairs` should be a "
                       "tuple of size 2.")
    per_replica = _make_tensor_into_per_replica(pair[0])
    result.append((per_replica, pair[1]))
  return result
def _validate_value_destination_pairs(value_destination_pairs):
  if not value_destination_pairs: return False
  if not isinstance(value_destination_pairs, (list, tuple)): return False
  if not all(isinstance(pair, tuple) for pair in value_destination_pairs):
    return False
  if not all(isinstance(v[0], value_lib.PerReplica)
             for v in value_destination_pairs):
    return False
  return True
def get_devices_from(destinations, canonicalize_devices=True):
  if isinstance(destinations, value_lib.DistributedValues):
  if canonicalize_devices:
    if isinstance(destinations, six.string_types):
      return (device_util.resolve(destinations),)
    return (device_util.resolve(destinations.device),)
  if isinstance(destinations, six.string_types):
    return (device_util.canonicalize_without_job_and_task(destinations),)
  return (device_util.canonicalize_without_job_and_task(destinations.device),)
def _devices_match(left, right, canonicalize_devices=True):
  return left is right or set(get_devices_from(
      left, canonicalize_devices)) == set(
          get_devices_from(right, canonicalize_devices))
def _all_devices_match(value_destination_pairs, canonicalize_devices=True):
  if not all(
      _devices_match(v, d, canonicalize_devices)
      for v, d in value_destination_pairs):
    return False
  if not all(
      _devices_match(v, value_destination_pairs[0][0], canonicalize_devices)
      for v, _ in value_destination_pairs[1:]):
    return False
  return True
def simple_broadcast(value,
                     destinations,
                     always_mirrored=False,
                     canonicalize_devices=True):
  devices = get_devices_from(destinations, canonicalize_devices)
  if len(devices) == 1 and not always_mirrored:
    return cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        value, devices[0])
  else:
    value_updates = []
    for d in devices:
      value_updates.append(
          cross_device_utils.copy_tensor_or_indexed_slices_to_device(value, d))
    return distribute_utils.regroup(value_updates,
                                    wrap_class=value_lib.Mirrored)
def _simple_reduce(per_replica_value, reduce_to_device, accumulation_fn,
                   reduce_op):
  all_values = per_replica_value.values
  if not all_values:
    raise ValueError("`per_replica_value` must be non-empty")
  count = len(all_values)
  with ops.device(reduce_to_device):
    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
      reduced = cross_device_utils.aggregate_tensors_or_indexed_slices(
          all_values, accumulation_fn)
      if reduce_op == reduce_util.ReduceOp.MEAN:
        reduced = cross_device_utils.divide_by_n_tensors_or_indexed_slices(
            reduced, count)
      elif reduce_op != reduce_util.ReduceOp.SUM:
        raise ValueError("`reduce_op` must be Reduce.SUM or Reduce.MEAN.")
  return reduced
def _simple_gather(per_replica_value, reduce_to_device, axis):
  all_values = per_replica_value.values
  if not all_values:
    raise ValueError("`per_replica_value` must be non-empty")
  with ops.device(reduce_to_device):
    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
      gathered = array_ops.concat(all_values, axis)
  return gathered
@tf_export("distribute.CrossDeviceOps")
class CrossDeviceOps(object):
  def __init__(self):
    self._canonicalize_devices = True
    pass
  @property
  def _num_between_graph_workers(self):
    return 1
  def reduce(self, reduce_op, per_replica_value, destinations, options=None):
    if options is None:
      options = collective_util.Options()
    if not isinstance(per_replica_value, value_lib.DistributedValues):
      per_replica_value = _make_tensor_into_per_replica(per_replica_value)
    validate_destinations(destinations)
    if self._num_between_graph_workers == 1 and len(
        per_replica_value.values) == 1 and _devices_match(
            per_replica_value, destinations, self._canonicalize_devices):
      with ops.device(per_replica_value.values[0].device):
        v = array_ops.identity(per_replica_value.values[0])
      return distribute_utils.regroup((v,), wrap_class=value_lib.Mirrored)
    if options is None:
      options = collective_util.Options()
    return self.reduce_implementation(reduce_op, per_replica_value,
                                      destinations, options)
  def _gather(self, per_replica_value, destinations, axis, options=None):
    if isinstance(per_replica_value, indexed_slices.IndexedSlices):
      raise NotImplementedError("gather/all_gather does not support "
                                "IndexedSlices")
    if options is None:
      options = collective_util.Options()
    if not isinstance(per_replica_value, value_lib.DistributedValues):
      per_replica_value = _make_tensor_into_per_replica(per_replica_value)
    validate_destinations(destinations)
    if self._num_between_graph_workers == 1 and len(
        per_replica_value.values) == 1 and _devices_match(
            per_replica_value, destinations, self._canonicalize_devices):
      with ops.device(per_replica_value.values[0].device):
        v = array_ops.identity(per_replica_value.values[0])
      return distribute_utils.regroup((v,), wrap_class=value_lib.Mirrored)
    return self._gather_implementation(per_replica_value, destinations, axis,
                                       options)
  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    raise NotImplementedError(
        "_gather method must be implemented in descendants.")
  def batch_reduce(self, reduce_op, value_destination_pairs, options=None):
    """Reduce values to destinations in batches.
    See `tf.distribute.StrategyExtended.batch_reduce_to`. This can only be
    called in the cross-replica context.
    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `tf.distribute.CrossDeviceOps.reduce` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    Returns:
      A list of `tf.Tensor` or `tf.distribute.DistributedValues`, one per pair
      in `value_destination_pairs`.
    Raises:
      ValueError: if `value_destination_pairs` is not an iterable of
        tuples of `tf.distribute.DistributedValues` and destinations.
    """
    if options is None:
      options = collective_util.Options()
    if not _validate_value_destination_pairs(value_destination_pairs):
      value_destination_pairs = _normalize_value_destination_pairs(
          value_destination_pairs)
    for _, d in value_destination_pairs:
      validate_destinations(d)
    if self._num_between_graph_workers == 1 and _all_devices_match(
        value_destination_pairs, self._canonicalize_devices) and len(
            value_destination_pairs[0][0].values) == 1:
      return [
          distribute_utils.regroup(v.values, wrap_class=value_lib.Mirrored)
          for v, _ in value_destination_pairs
      ]
    if options is None:
      options = collective_util.Options()
    return self.batch_reduce_implementation(reduce_op, value_destination_pairs,
                                            options)
  def broadcast(self, tensor, destinations):
    validate_destinations(destinations)
    return self.broadcast_implementation(tensor, destinations)
  @doc_controls.for_subclass_implementers
  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    raise NotImplementedError(
        "_reduce method must be implemented in descendants.")
  @doc_controls.for_subclass_implementers
  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    """Implementation of `batch_reduce`.
    Overriding this method is useful for subclass implementers.
    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `reduce` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    Returns:
      A list of `tf.Tensor` or `tf.distribute.DistributedValues`, one per pair
      in `value_destination_pairs`.
    Raises:
      ValueError: if `value_destination_pairs` is not an iterable of
        tuples of `tf.distribute.DistributedValues` and destinations.
    """
    raise NotImplementedError(
        "batch_reduce_implementation method must be implemented in descendants."
    )
  @doc_controls.for_subclass_implementers
  def broadcast_implementation(self, tensor, destinations):
    return simple_broadcast(
        tensor,
        destinations,
        always_mirrored=True,
        canonicalize_devices=self._canonicalize_devices)
  def _all_reduce(self, reduce_op, value, replica_id, options):
    """All-reduce the `value` across all replicas so that all get the result.
    `value` can be a nested structure of tensors or `IndexedSlices`. The
    implementation should generally batch the all-reduces when possible.
    `options` can be set to hint the batching behavior.
    This API must be called in a replica context.
    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: Value to be reduced. A tensor or a nested structure of tensors or
        `IndexedSlices`.
      replica_id: An interger indicating the id of the replica where this
        all_reduce is called under. This is the local replica id that ranges
        from 0 to len(local_devices) - 1.
      options: A `tf.distribute.experimental.CommunicationOptions`.
    Returns:
      A tensor/IndexedSlices or a nested strucutre of tensors/IndexedSlices with
      the reduced values. The structure is the same as `value`.
    """
    raise NotImplementedError("_all_reduce must be implemented in descendants.")
@tf_export("distribute.ReductionToOneDevice")
class ReductionToOneDevice(CrossDeviceOps):
  """A CrossDeviceOps implementation that copies values to one device to reduce.
  This implementation always copies values to one device to reduce them, then
  broadcast reduced values to the destinations. It doesn't support efficient
  batching.
  Here is how you can use `ReductionToOneDevice` in
  `tf.distribute.MirroredStrategy`:
  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.ReductionToOneDevice())
  ```
  """
  def __init__(self, reduce_to_device=None, accumulation_fn=None):
    self.reduce_to_device = reduce_to_device
    self.accumulation_fn = accumulation_fn or math_ops.add_n
    super(ReductionToOneDevice, self).__init__()
  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    if check_destinations(destinations):
      devices = get_devices_from(destinations, self._canonicalize_devices)
    else:
      devices = get_devices_from(per_replica_value, self._canonicalize_devices)
    reduce_to_device = self.reduce_to_device or devices[0]
    logging.log_first_n(
        logging.INFO,
        "Reduce to %s then broadcast to %r." % (reduce_to_device, devices), 10)
    reduced = _simple_reduce(per_replica_value, reduce_to_device,
                             self.accumulation_fn, reduce_op)
    return self.broadcast(reduced, destinations)
  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    if check_destinations(destinations):
      devices = get_devices_from(destinations, self._canonicalize_devices)
    else:
      devices = get_devices_from(per_replica_value, self._canonicalize_devices)
    reduce_to_device = self.reduce_to_device or devices[0]
    logging.log_first_n(
        logging.INFO,
        "Gather to %s then broadcast to %r." % (reduce_to_device, devices), 10)
    gathered = _simple_gather(per_replica_value, reduce_to_device, axis)
    return self.broadcast(gathered, destinations)
  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    return [
        self.reduce_implementation(
            reduce_op, t, destinations=v, options=options)
        for t, v in value_destination_pairs
    ]
def _group_value_by_device(per_replica_values):
  """Group values into sublists by their devices.
  This grouping is needed to call the all-reduce library because it expects a
  list of the following form:
    [[(grad0_gpu0, v0_gpu0), (grad1_gpu0, v1_gpu0), (grad2_gpu0, v2_gpu0) ...],
     [(grad0_gpu1, v0_gpu1), (grad1_gpu1, v1_gpu1), (grad2_gpu1, v2_gpu1) ...],
     [(grad0_gpu2, v0_gpu2), (grad1_gpu0, v1_gpu2), (grad2_gpu0, v2_gpu2) ...],
     ...
    ]
  Args:
    per_replica_values: a list of PerReplica objects.
  Returns:
    a list of lists, each sublist has components for its corresponding device of
      PerReplica objects, paired with a None.
  """
  grouped = [[] for _ in range(len(destinations))]
  for per_replica_value in per_replica_values:
    for i, v in enumerate(per_replica_value.values):
      assert per_replica_value._devices == destinations
      grouped[i].append((v, None))
  return grouped
def _ungroup_and_make_mirrored(grouped_reduced,
                               destinations,
                               reduce_op,
                               num_between_graph_workers=1):
  num_replicas = len(get_devices_from(destinations)) * num_between_graph_workers
  index = [[] for _ in range(len(grouped_reduced[0]))]
  for per_replica_reduced in grouped_reduced:
    for i, (v, _) in enumerate(per_replica_reduced):
      if reduce_op == reduce_util.ReduceOp.MEAN:
        with ops.device(v.device):
          index[i].append(v / num_replicas)
      else:
        index[i].append(v)
  return [distribute_utils.regroup(
      v, wrap_class=value_lib.Mirrored) for v in index]
class _ConcatAndSplitPacker(object):
  def __init__(self, num_packs=1):
    if num_packs <= 0:
      raise ValueError("num_packs must be greater than zero.")
    self.num_packs = num_packs
  def pack(self, grouped_grads_and_vars):
    self.grouped_grads_and_vars = grouped_grads_and_vars
    self.all_device_shapes = []
    self.all_device_sizes = []
    device_grad_packs = []
    for device_grads_and_vars in grouped_grads_and_vars:
      with ops.colocate_with(device_grads_and_vars[0][0]):
        flat_grads = [
            array_ops.reshape(g, [-1]) for g, _ in device_grads_and_vars
        ]
        device_shapes = [array_ops.shape(g) for g, _ in device_grads_and_vars]
        device_sizes = [array_ops.size(g) for g, _ in device_grads_and_vars]
        concat_grads = array_ops.concat(flat_grads, 0)
        num_splits = self.num_packs
        if all(g.shape.is_fully_defined() for g, _ in device_grads_and_vars):
          total_grad_size = sum(
              [g.shape.num_elements() for g, _ in device_grads_and_vars])
        else:
          total_grad_size = array_ops.size(concat_grads)
        split_size = total_grad_size // num_splits
        split_size_last = total_grad_size - split_size * (num_splits - 1)
        split_sizes = [split_size] * (num_splits - 1) + [split_size_last]
        grad_packs = array_ops.split(concat_grads, split_sizes)
        device_grad_packs.append(zip(grad_packs, [None] * num_splits))
        self.all_device_shapes.append(device_shapes)
        self.all_device_sizes.append(device_sizes)
    return device_grad_packs
  def unpack(self, summed_device_grad_packs):
    aggregated_device_grads = []
    for (summed_device_grad_packs,
         device_grads_and_vars, device_shapes, device_sizes) in zip(
             summed_device_grad_packs, self.grouped_grads_and_vars,
             self.all_device_shapes, self.all_device_sizes):
      with ops.colocate_with(summed_device_grad_packs[0][0]):
        device_grad_packs = [g for g, _ in summed_device_grad_packs]
        device_grads_concat = array_ops.concat(device_grad_packs, 0)
        grads_with_sizes = array_ops.split(device_grads_concat, device_sizes)
        grads_with_shapes = [
            array_ops.reshape(grad, shape)
            for shape, grad in zip(device_shapes, grads_with_sizes)
        ]
        summed_device_grads = [
            (g, v) for g, (_, v) in zip(grads_with_shapes,
                                        device_grads_and_vars)
        ]
        aggregated_device_grads.append(summed_device_grads)
    return aggregated_device_grads
def _pack_tensors(device_grads, num_packs=0):
  if num_packs > 0:
    tensor_packer = _ConcatAndSplitPacker(num_packs)
    device_grad_packs = tensor_packer.pack(device_grads)
  else:
    tensor_packer = None
    device_grad_packs = device_grads
  return device_grad_packs, tensor_packer
def _unpack_tensors(reduced, tensor_packer=None):
  if tensor_packer:
    return tensor_packer.unpack(reduced)
  return reduced
class AllReduceCrossDeviceOps(CrossDeviceOps):
  def __init__(self, all_reduce_alg="nccl", num_packs=1):
    self._all_reduce_alg = all_reduce_alg
    self._num_packs = num_packs
    self._simple_cross_replica_ops = ReductionToOneDevice()
    super(AllReduceCrossDeviceOps, self).__init__()
  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    if (_devices_match(per_replica_value, destinations) and
        not any("cpu" in d.lower() for d in get_devices_from(destinations))):
      return self._batch_all_reduce(reduce_op, [per_replica_value])[0]
    else:
      return self._simple_cross_replica_ops.reduce(reduce_op, per_replica_value,
                                                   destinations)
  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    if _all_devices_match(value_destination_pairs):
      return self._batch_all_reduce(reduce_op,
                                    [v[0] for v in value_destination_pairs])
    else:
      return [
          self.reduce_implementation(reduce_op, value, dest, options)
          for value, dest in value_destination_pairs
      ]
  def _batch_all_reduce(self, reduce_op, per_replica_values):
    dense_values, dense_indices, sparse_values, sparse_indices = (
        cross_device_utils.split_by_sparsity(per_replica_values))
    if dense_values:
      dense_results = self._do_batch_all_reduce(reduce_op, dense_values)
    else:
      dense_results = []
    if sparse_values:
      sparse_results = self._do_batch_all_reduce_sparse(reduce_op,
                                                        sparse_values)
    else:
      sparse_results = []
    return cross_device_utils.stitch_values(((dense_results, dense_indices),
                                             (sparse_results, sparse_indices)))
  def _do_batch_all_reduce(self, reduce_op, dense_values):
    logging.log_first_n(
        logging.INFO,
        "batch_all_reduce: %d all-reduces with algorithm = %s, num_packs = %d" %
        (len(dense_values), self._all_reduce_alg, self._num_packs), 10)
    grouped = _group_value_by_device(dense_values)
    device_grad_packs, tensor_packer = _pack_tensors(grouped, self._num_packs)
    if self._all_reduce_alg == "nccl":
      reduced = cross_device_utils.aggregate_gradients_using_nccl(
          device_grad_packs)
    else:
      reduced = (
          cross_device_utils.aggregate_gradients_using_hierarchical_copy(
              destinations, device_grad_packs))
    reduced = _unpack_tensors(reduced, tensor_packer)
    return _ungroup_and_make_mirrored(reduced, dense_values[0], reduce_op)
  def _do_batch_all_reduce_sparse(self, reduce_op, sparse_values):
    logging.log_first_n(
        logging.WARN,
        "Efficient allreduce is not supported for %d IndexedSlices" %
        len(sparse_values), 10)
    return self._simple_cross_replica_ops.batch_reduce(
        reduce_op, zip(sparse_values, sparse_values))
  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    logging.warning("gather/all_gather with NCCL or HierarchicalCopy is not "
                    "supported. Falling back to gather on one device and "
                    "then broadcast. We're working on a more efficient "
                    "implementation.")
                                          options)
AllReduceCrossTowerOps = AllReduceCrossDeviceOps
AllReduceSpecTuple = collections.namedtuple("AllReduceSpecTuple",
                                            "alg shards limit")
@tf_export("distribute.NcclAllReduce")
class NcclAllReduce(AllReduceCrossDeviceOps):
  """NCCL all-reduce implementation of CrossDeviceOps.
  It uses Nvidia NCCL for all-reduce. For the batch API, tensors will be
  repacked or aggregated for more efficient cross-device transportation.
  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.
  Here is how you can use `NcclAllReduce` in `tf.distribute.MirroredStrategy`:
  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.NcclAllReduce())
  ```
  """
  def __init__(self, num_packs=1):
    if num_packs < 0:
      raise ValueError(
          "NCCL all-reduce requires num_packs >= 0, but {} is specified".format(
              num_packs))
    super(NcclAllReduce, self).__init__(
        all_reduce_alg="nccl", num_packs=num_packs)
@tf_export("distribute.HierarchicalCopyAllReduce")
class HierarchicalCopyAllReduce(AllReduceCrossDeviceOps):
  """Hierarchical copy all-reduce implementation of CrossDeviceOps.
  It reduces to one GPU along edges in some hierarchy and broadcasts back to
  each GPU along the same path. For the batch API, tensors will be repacked or
  aggregated for more efficient cross-device transportation.
  This is a reduction created for Nvidia DGX-1 which assumes GPUs connects like
  that on DGX-1 machine. If you have different GPU inter-connections, it is
  likely that it would be slower than `tf.distribute.ReductionToOneDevice`.
  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.
  Here is how you can use `HierarchicalCopyAllReduce` in
  `tf.distribute.MirroredStrategy`:
  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
  ```
  """
  def __init__(self, num_packs=1):
    if num_packs < 0:
      raise ValueError(
          "HierarchicalCopy requires num_packs >= 0, but {} is specified"
          .format(num_packs))
    super(HierarchicalCopyAllReduce, self).__init__(
        all_reduce_alg="hierarchical_copy",
        num_packs=num_packs)
CollectiveCommunication = collective_util.CommunicationImplementation
CommunicationImplementation = collective_util.CommunicationImplementation
class CollectiveAllReduce(CrossDeviceOps):
  def __init__(self,
               devices,
               group_size,
               options,
               collective_keys=None,
               canonicalize_devices=True):
    if group_size % len(devices) > 0:
      raise ValueError("group_size must be divisible by the number of devices.")
    self._group_size = group_size
    self._options = options
    self._collective_keys = (collective_keys or
                             cross_device_utils.CollectiveKeys())
    self._lock = threading.Lock()
    if canonicalize_devices:
      self._devices = tuple(device_util.canonicalize(d) for d in devices)
    else:
      self._devices = tuple(
          device_util.canonicalize_without_job_and_task(d) for d in devices)
    group_key = self._collective_keys.get_group_key(self._devices)
    self._launchers = []
    self._limited_nccl = False
    for device in self._devices:
      launcher = cross_device_utils.CollectiveReplicaLauncher(
          group_key, group_size, self._collective_keys, device, options)
      self._launchers.append(launcher)
      if not launcher.can_order_nccl():
        self._limited_nccl = True
    self._pool = multiprocessing.pool.ThreadPool(len(self._devices))
    super(CollectiveAllReduce, self).__init__()
    self._canonicalize_devices = canonicalize_devices
  @property
  def _num_between_graph_workers(self):
    return self._group_size / len(self._devices)
  def _all_reduce(self, reduce_op, value, replica_id, options):
    flat_values = nest.flatten(value)
    if (self._limited_nccl and options.implementation
        == collective_util.CommunicationImplementation.NCCL and
        len(flat_values) == 1):
      options = options.merge(
          collective_util.Options(
              implementation=collective_util.CommunicationImplementation.RING))
    launcher = self._launchers[replica_id]
    dense_values, dense_indices, sparse_values, sparse_indices = (
        cross_device_utils.split_by_sparsity(flat_values))
    dense_results = []
    sparse_results = []
    if dense_values:
      dense_values.reverse()
      packs = cross_device_utils.group_by_size(dense_values,
                                               options.bytes_per_pack)
      if not context.executing_eagerly() and replica_id == 0:
        logging.info(
            "Collective all_reduce tensors: %d all_reduces, num_devices = %d, "
            "group_size = %d, implementation = %s, num_packs = %d",
            len(dense_values), len(self._launchers), self._group_size,
            options.implementation, len(packs))
      dense_results = launcher.batch_all_reduce(packs, options)
      if reduce_op == reduce_util.ReduceOp.MEAN:
        for i, v in enumerate(dense_results):
          with ops.device(self._devices[replica_id]):
            dense_results[i] = v / self._group_size
      dense_results.reverse()
    if sparse_values:
      if not context.executing_eagerly() and replica_id == 0:
        logging.info(
            "Collective all_reduce IndexedSlices: %d all_reduces, num_devices ="
            "%d, group_size = %d, implementation = %s", len(sparse_values),
            len(self._launchers), self._group_size, options.implementation)
      for indexed_slice in sparse_values:
        sparse_results.append(
            launcher.all_reduce_indexed_slices(indexed_slice, options))
      if reduce_op == reduce_util.ReduceOp.MEAN:
        for i, v in enumerate(sparse_results):
          with ops.device(self._devices[replica_id]):
            sparse_results[i] = indexed_slices.IndexedSlices(
                values=sparse_results[i].values / self._group_size,
                indices=sparse_results[i].indices,
                dense_shape=sparse_results[i].dense_shape)
    flat_results = cross_device_utils.stitch_values(
        ((dense_results, dense_indices), (sparse_results, sparse_indices)))
    return nest.pack_sequence_as(value, flat_results)
  def _all_reduce_per_replica_values(self, reduce_op, per_replica_values,
                                     options):
    values_by_device = [[] for _ in self._devices]
    num_devices = len(self._devices)
    for per_replica in per_replica_values:
      for i in range(num_devices):
        values_by_device[i].append(per_replica.values[i])
    if context.executing_eagerly():
      def thread_fn(device_id):
        with context.eager_mode():
          return self._all_reduce(reduce_op, values_by_device[device_id],
                                  device_id, options)
      with self._lock:
        outputs_by_device = self._pool.map(thread_fn, list(range(num_devices)))
    else:
      outputs_by_device = []
      with self._lock:
        for i in range(num_devices):
          outputs_by_device.append(
              self._all_reduce(reduce_op, values_by_device[i], i, options))
    result = []
    for values in zip(*outputs_by_device):
      result.append(
          distribute_utils.regroup(values, wrap_class=value_lib.Mirrored))
    return result
  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    values_util.mark_as_unsaveable()
    all_reduced = self._all_reduce_per_replica_values(reduce_op,
                                                      [per_replica_value],
                                                      options)[0]
    devices = get_devices_from(destinations, self._canonicalize_devices)
    if _devices_match(per_replica_value, destinations,
                      self._canonicalize_devices):
      return all_reduced
    if not isinstance(all_reduced, value_lib.Mirrored):
      all_reduced = value_lib.Mirrored([all_reduced])
    index = []
    with ops.control_dependencies(all_reduced.values):
      for d in devices:
        with ops.device(d):
          for v in all_reduced.values:
            if v.device == d:
              index.append(array_ops.identity(v))
              break
          else:
    return distribute_utils.regroup(index, wrap_class=value_lib.Mirrored)
  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    values_util.mark_as_unsaveable()
    all_devices_match = _all_devices_match(value_destination_pairs,
                                           self._canonicalize_devices)
    if all_devices_match:
      return self._all_reduce_per_replica_values(
          reduce_op, [v[0] for v in value_destination_pairs], options)
    else:
      if not all_devices_match:
        logging.log_first_n(
            logging.WARN, "Efficient batch_reduce is not supported if "
            "destinations are different.", 10)
      return [
          self.reduce_implementation(reduce_op, value, dest, options)
          for value, dest in value_destination_pairs
      ]
  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    all_gathered = self._batch_all_gather([per_replica_value], axis, options)[0]
    values_util.mark_as_unsaveable()
    devices = get_devices_from(destinations, self._canonicalize_devices)
    if _devices_match(per_replica_value, destinations,
                      self._canonicalize_devices):
      return all_gathered
    if not isinstance(all_gathered, value_lib.Mirrored):
      all_gathered = value_lib.Mirrored([all_gathered])
    index = []
    with ops.control_dependencies(all_gathered.values):
      for d in devices:
        with ops.device(d):
          for v in all_gathered.values:
            if v.device == d:
              index.append(array_ops.identity(v))
              break
            else:
    return distribute_utils.regroup(index, wrap_class=value_lib.Mirrored)
  def _batch_all_gather(self, per_replica_values, axis, options):
    batch_size = len(per_replica_values)
    if (self._limited_nccl and options.implementation
        == collective_util.CommunicationImplementation.NCCL and
        batch_size == 1):
      options = options.merge(
          collective_util.Options(
              implementation=collective_util.CommunicationImplementation.RING))
    logging.log_first_n(
        logging.INFO, "Collective batch_all_gather: %d all-gathers, "
        "num_devices = %d, group_size = %d, implementation = %s, " %
        (batch_size, len(
            self._devices), self._group_size, options.implementation), 10)
    def compute_gathered_values():
      gathered_values = []
      with self._lock, ops.name_scope("allgather"):
        for per_replica in per_replica_values:
          outputs = []
          for i in range(len(self._devices)):
            outputs.append(self._launchers[i].all_gather(
                per_replica.values[i], axis, options))
          gathered_values.append(outputs)
      return gathered_values
    if context.executing_eagerly():
      gathered_values = def_function.function(compute_gathered_values)()
    else:
      gathered_values = compute_gathered_values()
    mirrored = []
    for value in gathered_values:
      mirrored.append(
          distribute_utils.regroup(value, wrap_class=value_lib.Mirrored))
    return mirrored
  def __deepcopy__(self, memo):
    collective_keys = copy.deepcopy(self._collective_keys, memo)
    return CollectiveAllReduce(self._devices, self._group_size, self._options,
                               collective_keys, self._canonicalize_devices)
def select_cross_device_ops(devices, session_config=None):
  requested_devices = set(device_util.canonicalize(d) for d in devices)
  if ops.executing_eagerly_outside_functions():
    logical_gpus = context.context().list_logical_devices(device_type="GPU")
    physical_gpus = context.context().list_physical_devices(device_type="GPU")
    if len(logical_gpus) != len(physical_gpus):
      logging.warning("NCCL is not supported when using virtual GPUs, falling"
                      "back to reduction to one device")
      return ReductionToOneDevice()
    machine_devices = context.context().list_logical_devices()
  else:
    machine_devices = device_lib.list_local_devices(
        session_config=session_config)
  using_devices = set()
  for d in machine_devices:
    if device_util.canonicalize(d.name) in requested_devices:
      using_devices.add(d.name)
  if len(using_devices) != len(requested_devices):
    logging.warning(
        "Some requested devices in `tf.distribute.Strategy` are not visible "
        "to TensorFlow: %s", ",".join(list(requested_devices - using_devices)))
  if any("gpu" not in d.lower() for d in requested_devices):
    logging.warning("There are non-GPU devices in `tf.distribute.Strategy`, "
                    "not using nccl allreduce.")
    return ReductionToOneDevice()
  if kernels.get_registered_kernels_for_op("NcclAllReduce"):
    return NcclAllReduce(num_packs=1)
  else:
    logging.warning("Nccl kernel is not found, not using nccl allreduce.")
    return ReductionToOneDevice()
