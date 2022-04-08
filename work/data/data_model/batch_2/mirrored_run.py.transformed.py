
import contextlib
import functools
import threading
import weakref
from tensorflow.python import pywrap_tfe
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.util import traceback_utils
def _is_gpu_device(device):
  return tf_device.DeviceSpec.from_string(device).device_type == "GPU"
def call_for_each_replica(strategy, fn, args=None, kwargs=None):
  """Call `fn` on each worker devices(replica).
  It's highly recommended to wrap the call to this function inside a
  `tf.function`, otherwise the performance is poor.
  Args:
    strategy: `tf.distribute.Strategy`.
    fn: function to call on each worker devices.
    args: positional arguments to `fn`.
    kwargs: keyword arguments to `fn`.
  Returns:
    Wrapped returned value of `fn` from all replicas.
  """
  if args is None:
    args = ()
  if kwargs is None:
    kwargs = {}
  if isinstance(fn, def_function.Function):
        [_is_gpu_device(d) for d in strategy.extended.worker_devices]):
      return _call_for_each_replica(strategy, fn, args, kwargs)
    if strategy not in _cfer_fn_cache:
      _cfer_fn_cache[strategy] = weakref.WeakKeyDictionary()
    wrapped = _cfer_fn_cache[strategy].get(fn)
    if wrapped is None:
          python_function=functools.partial(call_for_each_replica, strategy,
                                            fn.python_function))
      _cfer_fn_cache[strategy][fn] = wrapped
    return wrapped(args, kwargs)
  if context.executing_eagerly():
    logging.log_first_n(
        logging.WARN, "Using %s eagerly has significant "
        "overhead currently. We will be working on improving "
        "this in the future, but for now please wrap "
        "`call_for_each_replica` or `experimental_run` or "
        "`run` inside a tf.function to get "
        "the best performance." % strategy.__class__.__name__, 5)
  else:
    fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
  return _call_for_each_replica(strategy, fn, args, kwargs)
_cfer_fn_cache = weakref.WeakKeyDictionary()
@contextlib.contextmanager
def _enter_graph(g, eager, creator_stack=None):
  if eager:
    with g.as_default(), context.eager_mode():
      if creator_stack is not None:
      yield
  else:
    with g.as_default():
      if creator_stack is not None:
      yield
@contextlib.contextmanager
def _maybe_enter_eager_mode(eager):
  if eager:
    with context.eager_mode():
      yield
  else:
    yield
def _cpu_device(device):
  cpu_device = tf_device.DeviceSpec.from_string(device)
  cpu_device = cpu_device.replace(device_type="CPU", device_index=0)
  return cpu_device.to_string()
  pass
def _get_thread_local_configuration_callable():
  if traceback_utils.is_traceback_filtering_enabled():
    thread_local_callables = {traceback_utils.enable_traceback_filtering}
  else:
    thread_local_callables = {traceback_utils.disable_traceback_filtering}
  return thread_local_callables
def _call_for_each_replica(distribution, fn, args, kwargs):
  """Run `fn` in separate threads, once per replica/worker device.
  Args:
    distribution: the DistributionStrategy object.
    fn: function to run (will be run once per replica, each in its own thread).
    args: positional arguments for `fn`
    kwargs: keyword arguments for `fn`.
  Returns:
    Merged return value of `fn` across all replicas.
  Raises:
    RuntimeError: If fn() calls get_replica_context().merge_call() a different
        number of times from the available devices.
  """
  run_concurrently = False
  if not context.executing_eagerly():
    ops.get_default_graph().switch_to_thread_local()
  coord = coordinator.Coordinator(clean_stop_exception_types=(_RequestedStop,))
  shared_variable_store = {}
  devices = distribution.extended.worker_devices
  thread_local_callables = _get_thread_local_configuration_callable()
  threads = []
  for index in range(len(devices)):
    variable_creator_fn = shared_variable_creator.make_fn(
        shared_variable_store, index)
    t = _MirroredReplicaThread(distribution, coord, index, devices,
                               variable_creator_fn, fn,
                               distribute_utils.caching_scope_local,
                               distribute_utils.select_replica(index, args),
                               distribute_utils.select_replica(index, kwargs),
                               thread_local_callables)
    threads.append(t)
  for t in threads:
    t.start()
  try:
    with coord.stop_on_exception():
      all_done = False
      while not all_done and not coord.should_stop():
        done = []
        if run_concurrently:
          for t in threads:
            t.should_run.set()
          for t in threads:
            t.has_paused.wait()
            t.has_paused.clear()
            if coord.should_stop():
              return None
            done.append(t.done)
        else:
          for t in threads:
            t.should_run.set()
            t.has_paused.wait()
            t.has_paused.clear()
            if coord.should_stop():
              return None
            done.append(t.done)
        if coord.should_stop():
          return None
        all_done = all(done)
        if not all_done:
          if any(done):
            raise RuntimeError("Some replicas made a different number of "
                               "replica_context().merge_call() calls.")
          merge_args = distribute_utils.regroup(
              tuple(t.merge_args for t in threads))
          merge_kwargs = distribute_utils.regroup(
              tuple(t.merge_kwargs for t in threads))
          mtt_captured_name_scope = threads[0].captured_name_scope
          mtt_captured_var_scope = threads[0].captured_var_scope
          mtt_captured_control_deps = set()
          for t in threads:
            mtt_captured_control_deps.update(t.captured_control_deps)
          with ops.name_scope(
              mtt_captured_name_scope), ops.control_dependencies(
                  mtt_captured_control_deps), variable_scope.variable_scope(
                      mtt_captured_var_scope), _maybe_enter_eager_mode(
                          threads[0].merge_call_entered_in_eager):
            merge_result = threads[0].merge_fn(distribution, *merge_args,
                                               **merge_kwargs)
          for r, t in enumerate(threads):
            t.merge_result = distribute_utils.select_replica(r, merge_result)
  finally:
    for t in threads:
      t.should_run.set()
    coord.join(threads)
  return distribute_utils.regroup(tuple(t.main_result for t in threads))
class _MirroredReplicaThread(threading.Thread):
  def __init__(self, dist, coord, replica_id, devices, variable_creator_fn, fn,
               caching_scope, args, kwargs, thread_local_callables=None):
    super(_MirroredReplicaThread, self).__init__()
    self.coord = coord
    self.distribution = dist
    self.devices = devices
    self.replica_id = replica_id
    self.replica_id_in_sync_group = (
    self.variable_creator_fn = variable_creator_fn
    self.main_fn = fn
    self.main_args = args
    self.main_kwargs = kwargs
    self.main_result = None
    self.done = False
    self.merge_fn = None
    self.merge_args = None
    self.merge_kwargs = None
    self.merge_result = None
    self.captured_name_scope = None
    self.captured_var_scope = None
    try:
      self.caching_scope_entered = caching_scope.new_cache_scope_count
      self.caching_scope_exited = caching_scope.cache_scope_exited_count
    except AttributeError:
      self.caching_scope_entered = None
      self.caching_scope_exited = None
    self.should_run = threading.Event()
    self.has_paused = threading.Event()
    context.ensure_initialized()
    ctx = context.context()
    self.in_eager = ctx.executing_eagerly()
    self.record_thread_local_summary_state()
    self.record_thread_local_eager_context_state()
    self.context_device_policy = (
        pywrap_tfe.TFE_ContextGetDevicePlacementPolicy(
    self.graph = ops.get_default_graph()
    with ops.init_scope():
      self._init_in_eager = context.executing_eagerly()
      self._init_graph = ops.get_default_graph()
    self._var_scope = variable_scope.get_variable_scope()
    self._name_scope = self.graph.get_name_scope()
    if self._name_scope:
      self._name_scope += "/"
    if self.replica_id > 0:
      if not self._name_scope:
        self._name_scope = ""
      self._name_scope += "replica_%d/" % self.replica_id
    self._thread_local_callables = thread_local_callables
  def run(self):
    self.should_run.wait()
    self.should_run.clear()
    try:
      if self.coord.should_stop():
        return
      self.restore_thread_local_summary_state()
      self.restore_thread_local_callable()
      self.restore_thread_local_eager_context_state()
      if (self.caching_scope_entered is not None and
          self.caching_scope_exited is not None):
        distribute_utils.caching_scope_local.new_cache_scope_count = self.caching_scope_entered
        distribute_utils.caching_scope_local.cache_scope_exited_count = self.caching_scope_exited
      with self.coord.stop_on_exception(), \
          _enter_graph(self._init_graph, self._init_in_eager), \
          _enter_graph(self.graph, self.in_eager,
                       self._variable_creator_stack), \
          context.device_policy(self.context_device_policy), \
          _MirroredReplicaContext(self.distribution,
                                  self.replica_id_in_sync_group), \
          ops.device(self.devices[self.replica_id]), \
          ops.name_scope(self._name_scope), \
          variable_scope.variable_scope(
              self._var_scope, reuse=self.replica_id > 0), \
          variable_scope.variable_creator_scope(self.variable_creator_fn):
        self.main_result = self.main_fn(*self.main_args, **self.main_kwargs)
        self.done = True
    finally:
      self.has_paused.set()
  def record_thread_local_summary_state(self):
    self._summary_step = summary_state.step
    self._summary_writer = summary_state.writer
    self._summary_recording = summary_state.is_recording
    self._summary_recording_distribution_strategy = (
        summary_state.is_recording_distribution_strategy)
  def restore_thread_local_summary_state(self):
    summary_state.step = self._summary_step
    summary_state.writer = self._summary_writer
    summary_state.is_recording = self._summary_recording
    summary_state.is_recording_distribution_strategy = (
        self._summary_recording_distribution_strategy)
  def record_thread_local_eager_context_state(self):
    ctx = context.context()
    self._eager_context_op_callbacks = eager_context_state.op_callbacks
  def restore_thread_local_eager_context_state(self):
    ctx = context.context()
    eager_context_state.op_callbacks = self._eager_context_op_callbacks
  def restore_thread_local_callable(self):
    if self._thread_local_callables:
      for fn in self._thread_local_callables:
        fn()
class _MirroredReplicaContext(distribute_lib.ReplicaContext):
  def _merge_call(self, fn, args, kwargs):
    """`merge_call()` implementation for synchronized replica.
    This pauses the current replica thread and passes `fn` and its arguments to
    the main thread. The main thread will wait until all replicas pause, then
    invoke `fn` with grouped arguments. The current replica thread will continue
    after `fn` completes.
    See `_call_for_each_replica` for the logic in the main thread.
    Args:
      fn: a function that is called in cross replica context with grouped
        arguments from each replica. `fn` should returns grouped values.
      args: positional arguments to `fn`.
      kwargs: keyward arguments to `fn`.
    Returns:
      Return value of `fn` for the current replica.
    Raises:
      RuntimeError: when merge_call happens in a different graph, e.g. in a
        different tf.function, which is not supported now.
      _RequestedStop: when stop is requested.
    """
    t = threading.current_thread()
    assert isinstance(t, _MirroredReplicaThread)
    t.merge_fn = fn
    t.merge_args = args
    t.merge_kwargs = kwargs
    t.captured_name_scope = t.graph.get_name_scope()
    if t.captured_name_scope:
      t.captured_name_scope += "/"
    t.captured_var_scope = variable_scope.get_variable_scope()
    t.merge_call_entered_in_eager = context.context().executing_eagerly()
    if ops.get_default_graph() != t.graph:
      raise RuntimeError(
          "`merge_call` called while defining a new graph or a tf.function."
          " This can often happen if the function `fn` passed to"
          " `strategy.run()` contains a nested `@tf.function`, and the nested "
          "`@tf.function` contains a synchronization point, such as aggregating"
          " gradients (e.g, optimizer.apply_gradients), or if the function `fn`"
          " uses a control flow statement which contains a synchronization"
          " point in the body. Such behaviors are not yet supported. Instead,"
          " please avoid nested `tf.function`s or control flow statements that"
          " may potentially cross a synchronization boundary, for example,"
          " wrap the `fn` passed to `strategy.run` or the entire `strategy.run`"
          " inside a `tf.function` or move the control flow out of `fn`. If"
          " you are subclassing a `tf.keras.Model`, please avoid decorating"
          " overridden methods `test_step` and `train_step` in `tf.function`.")
    t.has_paused.set()
    t.should_run.wait()
    t.should_run.clear()
    if t.coord.should_stop():
      raise _RequestedStop()
    t.merge_call_entered_in_eager = None
    return t.merge_result
  @property
  def devices(self):
    distribute_lib.require_replica_context(self)
    return [
        self._strategy.extended.worker_devices_by_replica[
            self._replica_id_in_sync_group]
    ]
