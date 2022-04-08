
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
_WORKER_MAXIMUM_RECOVERY_SEC = 3600
_CLOSURE_QUEUE_MAX_SIZE = 256 * 1024
_RPC_ERROR_FROM_PS = "GRPC error information from remote target /job:ps"
_JOB_WORKER_STRING_IDENTIFIER = "/job:worker"
RemoteValueStatus = values_lib.RemoteValueStatus
RemoteValue = values_lib.RemoteValue
RemoteValueImpl = values_lib.RemoteValueImpl
PerWorkerValues = values_lib.PerWorkerValues
class InputError(Exception):
  def __init__(self, original_exception):
    self.original_exception = original_exception
    message = ("Input has an error, the original exception is %r, "
               "error message is %s." %
               (original_exception, str(original_exception)))
    super().__init__(message)
    self.with_traceback(original_exception.__traceback__)
def _maybe_rebuild_remote_values(worker, structure):
  errors_in_structure = []
  def _get_error(val):
    if isinstance(val, RemoteValue):
        try:
      if error:
        errors_in_structure.append(error)
  nest.map_structure(_get_error, structure)
  if errors_in_structure:
    return errors_in_structure[0]
  else:
    return None
def _maybe_get_remote_value(val):
  if isinstance(val, RemoteValue):
    if error:
      raise AssertionError(
          "RemoteValue doesn't have a value because it has errors.")
    else:
  else:
    return val
def _maybe_as_type_spec(val):
  if isinstance(val, (RemoteValue, PerWorkerValues)):
      raise ValueError("Output of a scheduled function that is not "
                       "tf.function cannot be the input of another function.")
  else:
    return val
def _select_worker_slice(worker_id, structured):
  def _get(x):
  return nest.map_structure(_get, structured)
def _disallow_remote_value_as_input(structured):
  def _raise_if_remote_value(x):
    if isinstance(x, RemoteValue):
      raise ValueError(
          "`tf.distribute.experimental.coordinator.RemoteValue` used "
          "as an input to scheduled function is not yet "
          "supported.")
  nest.map_structure(_raise_if_remote_value, structured)
class Closure(object):
  def __init__(self, function, cancellation_mgr, args=None, kwargs=None):
    if not callable(function):
      raise ValueError("Function passed to `ClusterCoordinator.schedule` must "
                       "be a callable object.")
    self._args = args or ()
    self._kwargs = kwargs or {}
    _disallow_remote_value_as_input(self._args)
    _disallow_remote_value_as_input(self._kwargs)
    if isinstance(function, def_function.Function):
      replica_args = _select_worker_slice(0, self._args)
      replica_kwargs = _select_worker_slice(0, self._kwargs)
      with metric_utils.monitored_timer(
        self._concrete_function = function.get_concrete_function(
            *nest.map_structure(_maybe_as_type_spec, replica_args),
            **nest.map_structure(_maybe_as_type_spec, replica_kwargs))
    elif isinstance(function, tf_function.ConcreteFunction):
      self._concrete_function = function
    if hasattr(self, "_concrete_function"):
      self._output_type_spec = func_graph.convert_structure_to_signature(
          self._concrete_function.structured_outputs)
      self._function = cancellation_mgr.get_cancelable_function(
          self._concrete_function)
    else:
      self._output_type_spec = None
      self._function = function
    self._output_remote_value_ref = None
  def build_output_remote_value(self):
    if self._output_remote_value_ref is None:
      ret = RemoteValueImpl(None, self._output_type_spec)
      self._output_remote_value_ref = weakref.ref(ret)
      return ret
    else:
      raise ValueError(
          "The output of the Closure cannot be built more than once.")
  def maybe_call_with_output_remote_value(self, method):
    if self._output_remote_value_ref is None:
      return None
    output_remote_value = self._output_remote_value_ref()
    if output_remote_value is not None:
      return method(output_remote_value)
    return None
  def mark_cancelled(self):
    e = errors.CancelledError(
        None, None, "The corresponding function is "
        "cancelled. Please reschedule the function.")
  def execute_on(self, worker):
    replica_args = _select_worker_slice(worker.worker_index, self._args)
    replica_kwargs = _select_worker_slice(worker.worker_index, self._kwargs)
    e = (
        _maybe_rebuild_remote_values(worker, replica_args) or
        _maybe_rebuild_remote_values(worker, replica_kwargs))
    if e:
      if not isinstance(e, InputError):
        e = InputError(e)
      raise e
    with ops.device(worker.device_name):
      with context.executor_scope(worker.executor):
        with coordinator_context.with_dispatch_context(worker):
          with metric_utils.monitored_timer("closure_execution"):
            output_values = self._function(
                *nest.map_structure(_maybe_get_remote_value, replica_args),
                **nest.map_structure(_maybe_get_remote_value, replica_kwargs))
    self.maybe_call_with_output_remote_value(
class ResourceClosure(Closure):
  def build_output_remote_value(self):
    if self._output_remote_value_ref is None:
      ret = RemoteValueImpl(self, self._output_type_spec)
      self._output_remote_value_ref = weakref.ref(ret)
      return ret
    else:
      return self._output_remote_value_ref()
class _CoordinatedClosureQueue(object):
  def __init__(self):
    self._inflight_closure_count = 0
    self._queue_lock = threading.Lock()
    self._stop_waiting_condition = threading.Condition(self._queue_lock)
    self._closures_queued_condition = threading.Condition(self._queue_lock)
    self._should_process_closures = True
    self._queue_free_slot_condition = threading.Condition(self._queue_lock)
    self._no_inflight_closure_condition = threading.Condition(self._queue_lock)
    self._cancellation_mgr = cancellation.CancellationManager()
    if _CLOSURE_QUEUE_MAX_SIZE <= 0:
      logging.warning(
          "In a `ClusterCoordinator`, creating an infinite closure queue can "
          "consume a significant amount of memory and even lead to OOM.")
    self._queue = queue.Queue(maxsize=_CLOSURE_QUEUE_MAX_SIZE)
    self._error = None
    self._put_wait_lock = threading.Lock()
    self._watchdog = watchdog.WatchDog(on_triggered=self._on_watchdog_timeout)
  def _on_watchdog_timeout(self):
    logging.info("inflight_closure_count is %d", self._inflight_closure_count)
    logging.info("current error is %s:%r", self._error, self._error)
  def stop(self):
    with self._queue_lock:
      self._should_process_closures = False
      self._closures_queued_condition.notify_all()
    self._watchdog.stop()
  def _cancel_all_closures(self):
    self._cancellation_mgr.start_cancel()
    while self._inflight_closure_count > 0:
      self._no_inflight_closure_condition.wait()
    while True:
      try:
        closure = self._queue.get(block=False)
        self._queue_free_slot_condition.notify()
        closure.mark_cancelled()
      except queue.Empty:
        break
    self._cancellation_mgr = cancellation.CancellationManager()
  def _raise_if_error(self):
    if self._error:
      logging.error("Start cancelling closures due to error %r: %s",
                    self._error, self._error)
      self._cancel_all_closures()
      try:
      finally:
        self._error = None
  def put(self, closure):
    with self._put_wait_lock, self._queue_lock:
      self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
      self._queue.put(closure, block=False)
      self._raise_if_error()
      self._closures_queued_condition.notify()
  def get(self, timeout=None):
    with self._queue_lock:
      while self._queue.empty() and self._should_process_closures:
        if not self._closures_queued_condition.wait(timeout=timeout):
          return None
      if not self._should_process_closures:
        return None
      closure = self._queue.get(block=False)
      self._queue_free_slot_condition.notify()
      self._inflight_closure_count += 1
      return closure
  def mark_finished(self):
    with self._queue_lock:
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to mark_finished.")
      self._inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notify_all()
      if self._queue.empty() and self._inflight_closure_count == 0:
        self._stop_waiting_condition.notify_all()
      self._watchdog.report_closure_done()
  def put_back(self, closure):
    with self._queue_lock:
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to put_back.")
      if self._error:
        closure.mark_cancelled()
      else:
        self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
        self._queue.put(closure, block=False)
        self._closures_queued_condition.notify()
      self._inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notify_all()
  def wait(self, timeout=None):
    with self._put_wait_lock, self._queue_lock:
      while (not self._error and
             (not self._queue.empty() or self._inflight_closure_count > 0)):
        if not self._stop_waiting_condition.wait(timeout=timeout):
          return False
      self._raise_if_error()
      return True
  def mark_failed(self, e):
    with self._queue_lock:
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to mark_failed.")
      if self._error is None:
        self._error = e
      self._inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notify_all()
      self._stop_waiting_condition.notify_all()
  def done(self):
    with self._queue_lock:
      self._raise_if_error()
      return self._queue.empty() and self._inflight_closure_count == 0
class WorkerPreemptionHandler(object):
  def __init__(self, server_def, cluster):
    self._server_def = server_def
    self._cluster = cluster
    self._cluster_update_lock = threading.Lock()
    self._cluster_due_for_update_or_finish = threading.Event()
    self._worker_up_cond = threading.Condition(self._cluster_update_lock)
    self._error_from_recovery = None
    self._should_preemption_thread_run = True
    self._preemption_handler_thread = threading.Thread(
        target=self._preemption_handler,
        name="WorkerPreemptionHandler",
        daemon=True)
    self._preemption_handler_thread.start()
  def stop(self):
    self._should_preemption_thread_run = False
    with self._cluster_update_lock:
      self._cluster_due_for_update_or_finish.set()
  def _validate_preemption_failure(self, e):
    if _is_worker_failure(e) and (
      return
    raise e
  @contextlib.contextmanager
  def wait_on_failure(self,
                      on_failure_fn=None,
                      on_transient_failure_fn=None,
                      on_recovery_fn=None,
                      worker_device_name="(unknown)"):
    assert self._should_preemption_thread_run
    try:
      yield
    except (errors.OpError, InputError) as e:
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "It is treated as a transient connectivity failure for now.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "This derived error is ignored and not reported to users.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return
      if isinstance(e, errors.CancelledError) and "/job:" in str(e):
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "This derived error is ignored and not reported to users.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return
      self._validate_preemption_failure(e)
      logging.error("Worker %s failed with %r:%s", worker_device_name, e, e)
      if on_failure_fn:
        on_failure_fn()
      with self._cluster_update_lock:
        self._cluster_due_for_update_or_finish.set()
        self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
        if self._error_from_recovery:
          try:
            raise self._error_from_recovery
          finally:
            self._error_from_recovery = None
        logging.info("Worker %s has been recovered.", worker_device_name)
      if on_recovery_fn:
        with self.wait_on_failure(
            on_recovery_fn=on_recovery_fn,
            on_transient_failure_fn=on_transient_failure_fn,
            worker_device_name=worker_device_name):
          on_recovery_fn()
  def _preemption_handler(self):
    assert self._should_preemption_thread_run
    while True:
      self._cluster_due_for_update_or_finish.wait()
      if not self._should_preemption_thread_run:
        logging.info("Stopping the failure handing thread.")
        break
      with self._cluster_update_lock:
        try:
          logging.info("Cluster now being recovered.")
          context.context().update_server_def(self._server_def)
          logging.info("Cluster successfully recovered.")
          self._worker_up_cond.notify_all()
          if self._should_preemption_thread_run:
            self._cluster_due_for_update_or_finish.clear()
          try:
            self._validate_preemption_failure(e)
            self._error_from_recovery = ps_e
            self._worker_up_cond.notify_all()
            if self._should_preemption_thread_run:
              self._cluster_due_for_update_or_finish.clear()
          logging.error("Cluster update failed with error: %s. Retrying...", e)
class Worker(object):
  def __init__(self, worker_index, device_name, cluster):
    self.worker_index = worker_index
    self.device_name = device_name
    self.executor = executor.new_executor(enable_async=False)
    self.failure_handler = cluster.failure_handler
    self._cluster = cluster
    self._resource_remote_value_refs = []
    self._should_worker_thread_run = True
    threading.Thread(target=self._process_queue,
                     name="WorkerClosureProcessingLoop-%d" % self.worker_index,
                     daemon=True).start()
  def stop(self):
    self._should_worker_thread_run = False
  def _set_resources_aborted(self):
    for weakref_resource in self._resource_remote_value_refs:
      resource = weakref_resource()
      if resource:
  def _set_dead(self):
    raise NotImplementedError("_set_dead is not implemented.")
  def _process_closure(self, closure):
    assert closure is not None
    try:
      with self.failure_handler.wait_on_failure(
          on_failure_fn=lambda: self._cluster.closure_queue.put_back(closure),
          on_transient_failure_fn=lambda: self._cluster.closure_queue.put_back(
              closure),
          on_recovery_fn=self._set_resources_aborted,
          worker_device_name=self.device_name):
        closure.execute_on(self)
        with metric_utils.monitored_timer("remote_value_fetch"):
          closure.maybe_call_with_output_remote_value(lambda r: r.get())
        self._cluster.closure_queue.mark_finished()
      if not isinstance(e, errors.CancelledError):
        logging.error(
            "/job:worker/task:%d encountered the following error when "
            "processing closure: %r:%s", self.worker_index, e, e)
      self._cluster.closure_queue.mark_failed(e)
  def _maybe_delay(self):
    delay_secs = int(os.environ.get("TF_COORDINATOR_SCHEDULE_START_DELAY", "0"))
    delay_cap = int(
        os.environ.get("TF_COORDINATOR_SCHEDULE_START_DELAY_MAX", "0"))
    if delay_cap:
      delay_secs = min(delay_secs * self.worker_index, delay_cap)
    if delay_secs > 0:
      logging.info("Worker %d sleeping for %d seconds before running function",
                   self.worker_index, delay_secs)
    time.sleep(delay_secs)
  def _process_queue(self):
    self._maybe_delay()
    while self._should_worker_thread_run:
      closure = self._cluster.closure_queue.get()
      if not self._should_worker_thread_run or closure is None:
        return
      self._process_closure(closure)
      del closure
  def create_resource(self, function, args=None, kwargs=None):
    closure = ResourceClosure(
        function,
        args=args,
        kwargs=kwargs)
    resource_remote_value = closure.build_output_remote_value()
    self._register_resource(resource_remote_value)
    return resource_remote_value
  def _register_resource(self, resource_remote_value):
    if not isinstance(resource_remote_value, RemoteValue):
      raise ValueError("Resource being registered is not of type "
                       "`tf.distribute.experimental.coordinator.RemoteValue`.")
    self._resource_remote_value_refs.append(weakref.ref(resource_remote_value))
class Cluster(object):
  def __init__(self, strategy):
    self._num_workers = strategy._num_workers
    self._num_ps = strategy._num_ps
    self._transient_ps_failures_threshold = int(
        os.environ.get("TF_COORDINATOR_IGNORE_TRANSIENT_PS_FAILURES", 3))
    self._potential_ps_failures_lock = threading.Lock()
    self._potential_ps_failures_count = [0] * self._num_ps
    self._transient_timeouts_threshold = int(
        os.environ.get("TF_COORDINATOR_IGNORE_TRANSIENT_TIMEOUTS",
                       self._num_workers // 10))
    self._transient_timeouts_lock = threading.Lock()
    self._transient_timeouts_count = 0
    self.closure_queue = _CoordinatedClosureQueue()
    self.failure_handler = WorkerPreemptionHandler(context.get_server_def(),
                                                   self)
    worker_device_strings = [
        "/job:worker/replica:0/task:%d" % i for i in range(self._num_workers)
    ]
    self.workers = [
        Worker(i, w, self) for i, w in enumerate(worker_device_strings)
    ]
  def stop(self):
    self.failure_handler.stop()
    for worker in self.workers:
      worker.stop()
    self.closure_queue.stop()
  def _record_and_ignore_transient_ps_failure(self, e):
    if self._transient_ps_failures_threshold <= 0 or not _is_ps_failure(e):
      return False
    ps_tasks = _extract_failed_ps_instances(str(e))
    with self._potential_ps_failures_lock:
      for t in ps_tasks:
        self._potential_ps_failures_count[t] += 1
        if (self._potential_ps_failures_count[t] >=
            self._transient_ps_failures_threshold):
          return False
    return True
  def _record_and_ignore_transient_timeouts(self, e):
    if self._transient_timeouts_threshold <= 0:
      return False
    if not isinstance(e, errors.DeadlineExceededError):
      return False
    with self._transient_timeouts_lock:
      self._transient_timeouts_count += 1
      if self._transient_timeouts_count >= self._transient_timeouts_threshold:
        return False
    return True
  def schedule(self, function, args, kwargs):
    closure = Closure(
        function,
        args=args,
        kwargs=kwargs)
    ret = closure.build_output_remote_value()
    self.closure_queue.put(closure)
    return ret
  def join(self):
    self.closure_queue.wait()
  def done(self):
    return self.closure_queue.done()
@tf_export("distribute.experimental.coordinator.ClusterCoordinator",
           "distribute.coordinator.ClusterCoordinator", v1=[])
class ClusterCoordinator(object):
  def __new__(cls, strategy):
    if strategy._cluster_coordinator is None:
      strategy._cluster_coordinator = super(
          ClusterCoordinator, cls).__new__(cls)
    return strategy._cluster_coordinator
  def __init__(self, strategy):
    if not getattr(self, "_has_initialized", False):
      if not isinstance(strategy,
                        parameter_server_strategy_v2.ParameterServerStrategyV2):
        raise ValueError(
            "Only `tf.distribute.experimental.ParameterServerStrategy` "
            "is supported to work with "
            "`tf.distribute.experimental.coordinator.ClusterCoordinator` "
            "currently.")
      self._strategy = strategy
      self.strategy.extended._used_with_coordinator = True
      self._cluster = Cluster(strategy)
      self._has_initialized = True
  def __del__(self):
    self._cluster.stop()
  @property
  def strategy(self):
    return self._strategy
  def schedule(self, fn, args=None, kwargs=None):
    if not isinstance(fn,
                      (def_function.Function, tf_function.ConcreteFunction)):
      raise TypeError(
          "`tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`"
          " only accepts a `tf.function` or a concrete function.")
    with self.strategy.scope():
      remote_value = self._cluster.schedule(fn, args=args, kwargs=kwargs)
      return remote_value
  def join(self):
    self._cluster.join()
  def done(self):
    return self._cluster.done()
  def create_per_worker_dataset(self, dataset_fn):
    """Create dataset on workers by calling `dataset_fn` on worker devices.
    This creates the given dataset generated by dataset_fn on workers
    and returns an object that represents the collection of those individual
    datasets. Calling `iter` on such collection of datasets returns a
    `tf.distribute.experimental.coordinator.PerWorkerValues`, which is a
    collection of iterators, where the iterators have been placed on respective
    workers.
    Calling `next` on a `PerWorkerValues` of iterator is unsupported. The
    iterator is meant to be passed as an argument into
    `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`. When
    the scheduled function is about to be executed by a worker, the
    function will receive the individual iterator that corresponds to the
    worker. The `next` method can be called on an iterator inside a
    scheduled function when the iterator is an input of the function.
    Currently the `schedule` method assumes workers are all the same and thus
    assumes the datasets on different workers are the same, except they may be
    shuffled differently if they contain a `dataset.shuffle` operation and a
    random seed is not set. Because of this, we also recommend the datasets to
    be repeated indefinitely and schedule a finite number of steps instead of
    relying on the `OutOfRangeError` from a dataset.
    Example:
    ```python
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver=...)
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy=strategy)
    @tf.function
    def worker_fn(iterator):
      return next(iterator)
    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(
          lambda x: tf.data.Dataset.from_tensor_slices([3] * 3))
    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iter = iter(per_worker_dataset)
    remote_value = coordinator.schedule(worker_fn, args=(per_worker_iter,))
    assert remote_value.fetch() == 3
    ```
    Args:
      dataset_fn: The dataset function that returns a dataset. This is to be
        executed on the workers.
    Returns:
      An object that represents the collection of those individual
      datasets. `iter` is expected to be called on this object that returns
      a `tf.distribute.experimental.coordinator.PerWorkerValues` of the
      iterators (that are on the workers).
    """
    return values_lib.get_per_worker_dataset(dataset_fn, self)
  def _create_per_worker_resources(self, fn, args=None, kwargs=None):
    results = []
    for w in self._cluster.workers:
    return PerWorkerValues(tuple(results))
  def fetch(self, val):
    """Blocking call to fetch results from the remote values.
    This is a wrapper around
    `tf.distribute.experimental.coordinator.RemoteValue.fetch` for a
    `RemoteValue` structure; it returns the execution results of
    `RemoteValue`s. If not ready, wait for them while blocking the caller.
    Example:
    ```python
    strategy = ...
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy)
    def dataset_fn():
      return tf.data.Dataset.from_tensor_slices([1, 1, 1])
    with strategy.scope():
      v = tf.Variable(initial_value=0)
    @tf.function
    def worker_fn(iterator):
      def replica_fn(x):
        v.assign_add(x)
        return v.read_value()
      return strategy.run(replica_fn, args=(next(iterator),))
    distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
    distributed_iterator = iter(distributed_dataset)
    result = coordinator.schedule(worker_fn, args=(distributed_iterator,))
    assert coordinator.fetch(result) == 1
    ```
    Args:
      val: The value to fetch the results from. If this is structure of
        `tf.distribute.experimental.coordinator.RemoteValue`, `fetch()` will be
        called on the individual
        `tf.distribute.experimental.coordinator.RemoteValue` to get the result.
    Returns:
      If `val` is a `tf.distribute.experimental.coordinator.RemoteValue` or a
      structure of `tf.distribute.experimental.coordinator.RemoteValue`s,
      return the fetched `tf.distribute.experimental.coordinator.RemoteValue`
      values immediately if they are available, or block the call until they are
      available, and return the fetched
      `tf.distribute.experimental.coordinator.RemoteValue` values with the same
      structure. If `val` is other types, return it as-is.
    """
    def _maybe_fetch(val):
      if isinstance(val, RemoteValue):
        return val.fetch()
      else:
        return val
    return nest.map_structure(_maybe_fetch, val)
def _extract_failed_ps_instances(err_msg):
  tasks = re.findall("/job:ps/replica:0/task:[0-9]+", err_msg)
  return set(int(t.split(":")[-1]) for t in tasks)
def _is_ps_failure(error):
  if isinstance(error, InputError):
    error = error.original_exception
  return (isinstance(error, (errors.UnavailableError, errors.AbortedError)) and
          _RPC_ERROR_FROM_PS in str(error))
def _handle_graph_execution_error_as_worker_failure():
  return int(os.environ.get("TF_PS_HANDLE_UNKNOWN_ERROR", "0")) > 0
def _is_worker_failure(error):
  if (_handle_graph_execution_error_as_worker_failure() and
      isinstance(error, errors.UnknownError) and
      "Graph execution error" in str(error)):
    logging.info(f"Handling {type(error)}: {str(error)} as worker failure.")
    return True
  if isinstance(error, InputError):
    error = error.original_exception
  if _JOB_WORKER_STRING_IDENTIFIER not in str(error):
    return False
  if _RPC_ERROR_FROM_PS in str(error):
    return False
  if isinstance(error, (errors.UnavailableError, errors.AbortedError)):
    return True
  if isinstance(error, errors.InvalidArgumentError):
    if ("unknown device" in str(error).lower() or
        "Unable to find the relevant tensor remote_handle" in str(error)):
      return True
  if isinstance(error, errors.NotFoundError):
    if ("is neither a type of a primitive operation nor a name of a function "
        "registered" in str(error)):
      return True
  if isinstance(error, errors.CancelledError):
    return True
  return False
