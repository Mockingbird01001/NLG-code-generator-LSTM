
import enum
import threading
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec as type_spec_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class RemoteValueStatus(enum.Enum):
  NOT_READY = "NOT_READY"
  ABORTED = "ABORTED"
  READY = "READY"
@tf_export("distribute.experimental.coordinator.RemoteValue",
           "distribute.coordinator.RemoteValue", v1=[])
class RemoteValue(object):
  """An asynchronously available value of a scheduled function.
  This class is used as the return value of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` where
  the underlying value becomes available at a later time once the function has
  been executed.
  Using `tf.distribute.experimental.coordinator.RemoteValue` as an input to
  a subsequent function scheduled with
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` is
  currently not supported.
  Example:
  ```python
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))
  with strategy.scope():
    v1 = tf.Variable(initial_value=0.0)
    v2 = tf.Variable(initial_value=1.0)
  @tf.function
  def worker_fn():
    v1.assign_add(0.1)
    v2.assign_sub(0.2)
    return v1.read_value() / v2.read_value()
  result = coordinator.schedule(worker_fn)
  assert result.fetch() == 0.125
  for _ in range(10):
    result = coordinator.schedule(worker_fn)
  ```
  """
  def fetch(self):
    raise NotImplementedError("Must be implemented in subclasses.")
  def get(self):
    """Wait for the result of `RemoteValue` and return the tensor result.
    This makes the value concrete by copying the remote tensor to local.
    Returns:
      The actual output (in the form of `tf.Tensor`s) of the `tf.function`
      associated with this `RemoteValue`, previously returned by a
      `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` call.
      This can be a single Tensor, or a structure of Tensors, depending on the
      output of the `tf.function`.
    Raises:
      tf.errors.CancelledError: If the function that produces this `RemoteValue`
        is aborted or cancelled due to failure.
    """
    raise NotImplementedError("Must be implemented in subclasses.")
class RemoteValueImpl(RemoteValue):
    self._closure = closure
    self._type_spec = type_spec
    self._values = None
    self._has_fetched_to_local = False
    self._has_fetched_to_local_lock = threading.Lock()
    self._fetched_tensors = None
    self._error = None
    self._status_available_event = threading.Event()
    self._status = RemoteValueStatus.NOT_READY
  def _set_aborted(self):
    self._status = RemoteValueStatus.ABORTED
    self._values = None
    self._error = None
    self._status_available_event.set()
  def _rebuild_on(self, worker):
    self._status_available_event.clear()
    self._closure.execute_on(worker)
  def _set_values(self, tensors):
    self._status = RemoteValueStatus.READY
    self._values = tensors
    self._error = None
    self._status_available_event.set()
  def _set_error(self, exception):
    self._status = RemoteValueStatus.READY
    self._values = None
    self._error = exception
    self._status_available_event.set()
  def _get_values(self):
    self._status_available_event.wait()
    return self._values
  def _get_error(self):
    self._status_available_event.wait()
    return self._error
  def _wait_and_maybe_error(self):
    self._status_available_event.wait()
    if self._status is RemoteValueStatus.ABORTED:
      raise errors.CancelledError(
          None, None,
          "The corresponding function is aborted. Please reschedule the "
          "function.")
    if self._error is not None:
      raise self._error
  def fetch(self):
    return nest.map_structure(
        lambda x: x.numpy() if hasattr(x, "numpy") else x, self.get())
  def get(self):
    self._wait_and_maybe_error()
    with self._has_fetched_to_local_lock:
      if not self._has_fetched_to_local:
        def copy_tensor(composite_tensor_obj):
          if isinstance(composite_tensor_obj, input_lib.DistributedIterator):
            return composite_tensor_obj
          with ops.device("/job:%s" % context.get_server_def().job_name):
            return array_ops.identity(composite_tensor_obj)
        if self._values is not None:
          self._fetched_tensors = nest.map_structure(copy_tensor, self._values)
        self._has_fetched_to_local = True
    return self._fetched_tensors
@tf_export("distribute.experimental.coordinator.PerWorkerValues",
           "distribute.coordinator.PerWorkerValue", v1=[])
class PerWorkerValues(composite_tensor.CompositeTensor):
  """A container that holds a list of values, one value per worker.
  `tf.distribute.experimental.coordinator.PerWorkerValues` contains a collection
  of values, where each of the values is located on its corresponding worker,
  and upon being used as one of the `args` or `kwargs` of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule()`, the
  value specific to a worker will be passed into the function being executed at
  that corresponding worker.
  Currently, the only supported path to create an object of
  `tf.distribute.experimental.coordinator.PerWorkerValues` is through calling
  `iter` on a `ClusterCoordinator.create_per_worker_dataset`-returned
  distributed dataset instance. The mechanism to create a custom
  `tf.distribute.experimental.coordinator.PerWorkerValues` is not yet supported.
  """
  def __init__(self, values):
    for v in values:
      if not isinstance(v, RemoteValue):
        raise AssertionError(
            "`PerWorkerValues` should only take `RemoteValue`s.")
    self._values = tuple(values)
  @property
  def _type_spec(self):
    return PerWorkerValuesTypeSpec(
        type(self))
class PerWorkerValuesTypeSpec(type_spec_lib.TypeSpec):
  def __init__(self, value_spec, descendant_type):
    assert value_spec
    self._value_spec = value_spec
    self._descendant_type = descendant_type
  def _serialize(self):
    return (self._value_spec,)
  @property
  def value_type(self):
    return self._descendant_type
  def most_specific_common_supertype(self, others):
    raise NotImplementedError(
        "most_specific_common_supertype is not implemented")
  @property
  def _component_specs(self):
    return self._value_spec
  def _to_components(self, value):
    return self._value_spec
  def _from_components(self, value):
    return value
class PerWorkerDatasetFromDatasetFunction(object):
  def __init__(self, dataset_fn, coordinator):
    def disallow_variable_creation(next_creator, **kwargs):
      raise ValueError("Creating variables in `dataset_fn` is not allowed.")
    if isinstance(dataset_fn, def_function.Function):
      with variable_scope.variable_creator_scope(disallow_variable_creation):
        dataset_fn = dataset_fn.get_concrete_function()
    elif not isinstance(dataset_fn, tf_function.ConcreteFunction):
      with variable_scope.variable_creator_scope(disallow_variable_creation):
        dataset_fn = def_function.function(dataset_fn).get_concrete_function()
    self._dataset_fn = dataset_fn
    self._coordinator = coordinator
    self._element_spec = None
  def __iter__(self):
    if (not context.executing_eagerly() or
        ops.get_default_graph().building_function):
      raise RuntimeError(
          "__iter__() is not supported inside of tf.function or in graph mode.")
    def _create_per_worker_iterator():
      dataset = self._dataset_fn()
      return iter(dataset)
    per_worker_iterator = self._coordinator._create_per_worker_resources(
        _create_per_worker_iterator)
    for iterator_remote_value in per_worker_iterator._values:
      iterator_remote_value._type_spec = (
          input_lib.get_iterator_spec_from_dataset(
              self._coordinator.strategy, self._dataset_fn.structured_outputs))
    return PerWorkerDistributedIterator(per_worker_iterator._values)
  @property
  def element_spec(self):
    if not isinstance(self._dataset_fn, tf_function.ConcreteFunction):
      raise NotImplementedError(
          "`element_spec` is not supported when the `dataset_fn` is not "
          "a `ConcreteFunction`.")
    return self._dataset_fn.structured_outputs.element_spec
def serialize_dataset_to_graph(dataset):
  graph_def = gen_dataset_ops.dataset_to_graph_v2(
      external_state_policy=ExternalStatePolicy.WARN.value,
      strip_device_assignment=True)
  return graph_def
class _RemoteDataset(dataset_ops.DatasetSource):
  def __init__(self, graph_def, element_spec):
    self._elem_spec = element_spec
    variant_tensor = ged_ops.dataset_from_graph(graph_def)
    super(_RemoteDataset, self).__init__(variant_tensor)
  @property
  def element_spec(self):
    return self._elem_spec
def deserialize_dataset_from_graph(graph_def, element_spec):
  return _RemoteDataset(graph_def, element_spec)
class PerWorkerDatasetFromDataset(PerWorkerDatasetFromDatasetFunction):
  def __init__(self, dataset, coordinator):
    if isinstance(dataset, input_lib.DistributedDataset):
      original_dataset = dataset._original_dataset
      serialized = serialize_dataset_to_graph(original_dataset)
      def dataset_fn():
        deserialized = deserialize_dataset_from_graph(
            serialized, original_dataset.element_spec)
        dataset.build(dataset_to_replace=deserialized)
        return dataset
    elif isinstance(dataset, input_lib.DistributedDatasetsFromFunction):
      def dataset_fn():
        dataset.build()
        return dataset
    elif isinstance(dataset, dataset_ops.Dataset):
      serialized = serialize_dataset_to_graph(dataset)
      def dataset_fn():
        return deserialize_dataset_from_graph(serialized, dataset.element_spec)
    else:
      raise ValueError("Unexpected dataset type!")
    super(PerWorkerDatasetFromDataset, self).__init__(dataset_fn, coordinator)
def get_per_worker_dataset(dataset_or_dataset_fn, coordinator):
  if callable(dataset_or_dataset_fn):
    return PerWorkerDatasetFromDatasetFunction(dataset_or_dataset_fn,
                                               coordinator)
  else:
    return PerWorkerDatasetFromDataset(dataset_or_dataset_fn, coordinator)
class PerWorkerDistributedIterator(PerWorkerValues):
  def __next__(self):
    return self.get_next()
  def get_next(self, name=None):
    raise NotImplementedError("Iterating over an `AsyncDistributedIterator` "
                              "is not supported right now.")
