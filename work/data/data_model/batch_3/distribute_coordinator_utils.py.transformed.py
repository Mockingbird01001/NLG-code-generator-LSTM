
"""Utilities related to distribute coordinator.
The module is used only for utils to support legacy TF1 code path involving
distribute coordinator, and is not expected to change in any way. This is
subject to cleanup once TF1 is no longer supported.
TODO(rchao): Remove this module once TF1 is not supported.
"""
import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
_worker_context = threading.local()
_thread_local = threading.local()
def get_current_worker_context():
  try:
    return _worker_context.current
  except AttributeError:
    return None
class _TaskType(object):
  PS = "ps"
  WORKER = "worker"
  CHIEF = "chief"
  EVALUATOR = "evaluator"
  CLIENT = "client"
def _get_num_workers(cluster_spec):
  if not cluster_spec:
    return 0
  return len(cluster_spec.as_dict().get(_TaskType.WORKER, [])) + len(
      cluster_spec.as_dict().get(_TaskType.CHIEF, []))
class _WorkerContext(object):
  def __init__(self,
               strategy,
               cluster_spec,
               task_type,
               task_id,
               session_config=None,
               rpc_layer="grpc",
               worker_barrier=None):
    self._strategy = strategy
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._session_config = session_config
    self._worker_barrier = worker_barrier
    self._rpc_layer = rpc_layer
    self._master_target = self._get_master_target()
    self._num_workers = _get_num_workers(cluster_spec)
    self._is_chief_node = self._is_chief()
  def _debug_message(self):
    if self._cluster_spec:
      return "[cluster_spec: %r, task_type: %r, task_id: %r]" % (
          self._cluster_spec, self.task_type, self.task_id)
    else:
      return "[local]"
  def __enter__(self):
    old_context = get_current_worker_context()
    if old_context:
      raise ValueError(
          "You cannot run distribute coordinator in a `worker_fn`.\t" +
          self._debug_message())
    _worker_context.current = self
  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    _worker_context.current = None
  def _get_master_target(self):
    if not self._cluster_spec or self._task_type == _TaskType.EVALUATOR:
      return ""
    if not self._task_type:
      if _TaskType.CHIEF in self._cluster_spec.jobs:
        task_type = _TaskType.CHIEF
        task_id = 0
      else:
        assert _TaskType.WORKER in self._cluster_spec.jobs
        task_type = _TaskType.WORKER
        task_id = 0
    else:
      task_type = self._task_type
      task_id = self._task_id
    prefix = ""
    if self._rpc_layer:
      prefix = self._rpc_layer + "://"
    return prefix + self._cluster_spec.job_tasks(task_type)[task_id or 0]
  def _is_chief(self):
    if (not self._cluster_spec or
        self._task_type in [_TaskType.CHIEF, _TaskType.EVALUATOR, None]):
      return True
    if (_TaskType.CHIEF not in self._cluster_spec.jobs and
        self._task_type == _TaskType.WORKER and self._task_id == 0):
      return True
    return False
  def wait_for_other_workers(self):
    if not self._worker_barrier:
      return
    self._worker_barrier.wait()
  def session_creator(self,
                      scaffold=None,
                      config=None,
                      checkpoint_dir=None,
                      checkpoint_filename_with_path=None,
                      max_wait_secs=7200):
    if config:
      session_config = copy.deepcopy(config)
      session_config.MergeFrom(self._session_config)
    else:
      session_config = self._session_config
    if not self._strategy or self._strategy.extended.experimental_should_init:
      logging.info("Creating chief session creator with config: %r", config)
      return monitored_session.ChiefSessionCreator(
          scaffold,
          master=self.master_target,
          config=session_config,
          checkpoint_dir=checkpoint_dir,
          checkpoint_filename_with_path=checkpoint_filename_with_path)
    else:
      logging.info("Creating worker session creator with config: %r", config)
      return monitored_session.WorkerSessionCreator(
          scaffold,
          master=self.master_target,
          config=session_config,
          max_wait_secs=max_wait_secs)
  @property
  def session_config(self):
    return copy.deepcopy(self._session_config)
  @property
  def has_barrier(self):
    return self._worker_barrier is not None
  @property
  def distributed_mode(self):
    return bool(self._cluster_spec) and self._task_type != _TaskType.EVALUATOR
  @property
  def cluster_spec(self):
    return copy.deepcopy(self._cluster_spec)
  @property
  def task_type(self):
    return self._task_type
  @property
  def task_id(self):
    return self._task_id
  @property
  def master_target(self):
    return self._master_target
  @property
  def is_chief(self):
    return self._is_chief_node
  @property
  def num_workers(self):
    return self._num_workers
  @property
  def experimental_should_init(self):
    return self._strategy.extended.experimental_should_init
  @property
  def should_checkpoint(self):
    return self._strategy.extended.should_checkpoint
  @property
  def should_save_summary(self):
    return self._strategy.extended.should_save_summary
def _run_single_worker(worker_fn,
                       strategy,
                       cluster_spec,
                       task_type,
                       task_id,
                       session_config,
                       rpc_layer="",
                       worker_barrier=None,
                       coord=None):
  session_config = copy.deepcopy(session_config)
  strategy = copy.deepcopy(strategy)
  if task_type == _TaskType.EVALUATOR:
    if strategy:
      strategy.configure(session_config)
  else:
    assert strategy
    strategy.configure(session_config, cluster_spec, task_type, task_id)
  context = _WorkerContext(
      strategy,
      cluster_spec,
      task_type,
      task_id,
      session_config=session_config,
      rpc_layer=rpc_layer,
      worker_barrier=worker_barrier)
  with context:
    if coord:
      with coord.stop_on_exception():
        return worker_fn(strategy)
    else:
      return worker_fn(strategy)
def _split_cluster_for_evaluator(cluster_spec, task_type):
  new_cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
  if task_type == _TaskType.EVALUATOR:
    assert _TaskType.EVALUATOR in new_cluster_spec
    new_cluster_spec = {
        _TaskType.EVALUATOR: new_cluster_spec[_TaskType.EVALUATOR]
    }
  else:
    new_cluster_spec.pop(_TaskType.EVALUATOR, None)
  return normalize_cluster_spec(new_cluster_spec)
def _run_std_server(cluster_spec=None,
                    task_type=None,
                    task_id=None,
                    session_config=None,
                    rpc_layer=None,
                    environment=None):
  if getattr(_thread_local, "server", None) is not None:
    assert _thread_local.cluster_spec == cluster_spec
    assert _thread_local.task_type == task_type
    assert _thread_local.task_id == task_id
    assert _thread_local.session_config_str == repr(session_config)
    assert _thread_local.rpc_layer == rpc_layer
    assert _thread_local.environment == environment
    return _thread_local.server
  else:
    _thread_local.server_started = True
    _thread_local.cluster_spec = cluster_spec
    _thread_local.task_type = task_type
    _thread_local.task_id = task_id
    _thread_local.session_config_str = repr(session_config)
    _thread_local.rpc_layer = rpc_layer
    _thread_local.environment = environment
  assert cluster_spec
  target = cluster_spec.task_address(task_type, task_id)
  if rpc_layer:
    target = rpc_layer + "://" + target
  class _FakeServer(object):
    def start(self):
      logging.info(
          "Creating a remote session to start a TensorFlow server, "
          "target = %r, session_config=%r", target, session_config)
      session.Session(target=target, config=session_config)
    def join(self):
      while True:
        time.sleep(5)
  if environment == "google":
    server = _FakeServer()
  else:
    if session_config:
      logging.info(
          "Starting standard TensorFlow server, target = %r, session_config= "
          "%r", target, session_config)
    else:
      logging.info("Starting standard TensorFlow server, target = %r", target)
    cluster_spec = _split_cluster_for_evaluator(cluster_spec, task_type)
    server = server_lib.Server(
        cluster_spec,
        job_name=task_type,
        task_index=task_id,
        config=session_config,
        protocol=rpc_layer)
  server.start()
  _thread_local.server = server
  return server
def _configure_session_config_for_std_servers(strategy, eval_strategy,
                                              session_config, cluster_spec,
                                              task_type, task_id):
  if task_type == _TaskType.EVALUATOR:
    if eval_strategy:
      eval_strategy.configure(session_config=session_config)
  else:
    strategy = copy.deepcopy(strategy)
    strategy.configure(
        session_config=session_config,
        cluster_spec=cluster_spec,
        task_type=task_type,
        task_id=task_id)
  del session_config.device_filters[:]
def run_distribute_coordinator(worker_fn,
                               strategy,
                               eval_fn=None,
                               eval_strategy=None,
                               cluster_spec=None,
                               task_type=None,
                               task_id=None,
                               session_config=None,
                               rpc_layer="grpc"):
  """Runs the coordinator for distributed TensorFlow.
  This function runs a split coordinator for distributed TensorFlow in its
  default mode, i.e the STANDALONE_CLIENT mode. Given a `cluster_spec`
  specifying server addresses and their roles in a cluster, this coordinator
  will figure out how to set them up, give the underlying function the right
  targets for master sessions via a scope object and coordinate their training.
  The cluster consisting of standard servers needs to be brought up either with
  the standard server binary or with a binary running distribute coordinator
  with `task_type` set to non-client type which will then turn into standard
  servers.
  In addition to be the distribute coordinator, this is also the source of
  configurations for each job in the distributed training. As there are multiple
  ways to configure a distributed TensorFlow cluster, its context object
  provides these configurations so that users or higher-level APIs don't have to
  figure out the configuration for each job by themselves.
  In the between-graph replicated training, this coordinator will create
  multiple threads and each calls the `worker_fn` which is supposed to create
  its own graph and connect to one worker master given by its context object. In
  the in-graph replicated training, it has only one thread calling this
  `worker_fn`.
  Another mode is the INDEPENDENT_WORKER mode where each server runs a
  distribute coordinator which will start a standard server and optionally runs
  `worker_fn` depending whether it is between-graph training or in-graph
  replicated training.
  The `strategy` object is expected to be a DistributionStrategy object which
  has implemented methods needed by distributed coordinator such as
  `configure(session_config, cluster_spec, task_type, task_id)` which configures
  the strategy object for a specific task and `experimental_should_init`
  property which instructs the distribute coordinator whether to run init ops
  for a task. The distribute coordinator will make a copy of the `strategy`
  object, call its `configure` method and pass it to `worker_fn` as an argument.
  The `worker_fn` defines the training logic and is called under its own
  worker context which can be accessed to via `get_current_worker_context`. A
  worker context provides access to configurations for each task, e.g. the
  task_type, task_id, master target and so on. Since `worker_fn` will be called
  in a thread and possibly multiple times, caller should be careful when it
  accesses global data. For example, it is unsafe to define flags in a
  `worker_fn` or to define different environment variables for different
  `worker_fn`s.
  The `worker_fn` for the between-graph replication is defined as if there is
  only one worker corresponding to the `worker_fn` and possibly ps jobs. For
  example, when training with parameter servers, it assigns variables to
  parameter servers and all other operations to that worker. In the in-graph
  replication case, the `worker_fn` has to define operations for all worker
  jobs. Using a distribution strategy can simplify the `worker_fn` by not having
  to worry about the replication and device assignment of variables and
  operations.
  This method is intended to be invoked by high-level APIs so that users don't
  have to explicitly call it to run this coordinator. For those who don't use
  high-level APIs, to change a program to use this coordinator, wrap everything
  in a the program after global data definitions such as commandline flag
  definition into the `worker_fn` and get task-specific configurations from
  the worker context.
  The `cluster_spec` can be either passed by the argument or parsed from the
  "TF_CONFIG" environment variable. Example of a TF_CONFIG:
  ```
    cluster = {'chief': ['host0:2222'],
               'ps': ['host1:2222', 'host2:2222'],
               'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
    os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster})
  ```
  If `cluster_spec` is not given in any format, it becomes local training and
  this coordinator will connect to a local session.
  For evaluation, if "evaluator" exists in the cluster_spec, a separate thread
  will be created to call `eval_fn` with its `task_type` set to "evaluator". If
  `eval_fn` is not defined, fall back to `worker_fn`. This implies that
  evaluation will be done on a single machine if there is an "evaluator" task.
  If "evaluator" doesn't exist in the cluster_spec, it entirely depends on the
  `worker_fn` for how to do evaluation.
  Args:
    worker_fn: the function to be called. The function should accept a
      `strategy` object and will be given access to a context object via a
      context manager scope.
    strategy: a DistributionStrategy object specifying whether it should run
      between-graph replicated training or not, whether to run init ops, etc.
      This object will also be configured given `session_config`,
      `cluster_spec`, `task_type` and `task_id`.
    eval_fn: optional function for "evaluator" task. If `eval_fn` is not passed
      in but a "evaluator" task is found in the `cluster_spec`, the `worker_fn`
      will be used for this task.
    eval_strategy: optional DistributionStrategy object for "evaluator" task.
    cluster_spec: a dict, ClusterDef or ClusterSpec specifying servers and roles
      in a cluster. If not set or empty, fall back to local training.
    task_type: the current task type, optional if this is a client.
    task_id: the current task id, optional if this is a client.
    session_config: an optional `tf.compat.v1.ConfigProto` object which will be
      passed to `strategy`'s `configure` method and used to create a session.
    rpc_layer: optional string, the protocol for RPC, e.g. "grpc".
  Raises:
    ValueError: if `cluster_spec` is supplied but not a dict or a ClusterDef or
      a ClusterSpec.
  Returns:
    In the client job, return the value returned by `worker_fn` if
    it is in-graph replication or INDEPENDENT_WORKER mode; return None
    otherwise.
  """
  tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
  rpc_layer = tf_config.get("rpc_layer", rpc_layer)
  environment = tf_config.get("environment", None)
  if not cluster_spec:
    cluster_spec = tf_config.get("cluster", {})
    task_env = tf_config.get("task", {})
    if task_env:
      task_type = task_env.get("type", task_type)
      task_id = int(task_env.get("index", task_id))
  if cluster_spec:
    cluster_spec = normalize_cluster_spec(cluster_spec)
  elif hasattr(strategy.extended, "_cluster_resolver"):
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    rpc_layer = cluster_resolver.rpc_layer or rpc_layer
    environment = cluster_resolver.environment
    cluster_spec = cluster_resolver.cluster_spec()
  session_config = session_config or config_pb2.ConfigProto(
      allow_soft_placement=True)
  if cluster_spec:
    logging.info(
        "Running Distribute Coordinator with cluster_spec = %r, "
        "task_type = %r, task_id = %r, environment = %r, rpc_layer = %r",
        cluster_spec.as_dict(), task_type, task_id, environment, rpc_layer)
  if not cluster_spec:
    logging.info("Running local Distribute Coordinator.")
    _run_single_worker(worker_fn, strategy, None, None, None, session_config,
                       rpc_layer)
    if eval_fn:
      _run_single_worker(eval_fn, eval_strategy, None, None, None,
                         session_config, rpc_layer)
    else:
      logging.warning("Skipped evaluation since `eval_fn` is not passed in.")
  else:
    if not eval_fn:
      logging.warning("`eval_fn` is not passed in. The `worker_fn` will be "
                      "used if an \"evaluator\" task exists in the cluster.")
    eval_fn = eval_fn or worker_fn
    if not eval_strategy:
      logging.warning("`eval_strategy` is not passed in. No distribution "
                      "strategy will be used for evaluation.")
    _configure_session_config_for_std_servers(strategy, eval_strategy,
                                              session_config, cluster_spec,
                                              task_type, task_id)
    if (task_type != _TaskType.EVALUATOR and
        not getattr(strategy.extended, "_std_server_started", False)):
      server = _run_std_server(
          cluster_spec=cluster_spec,
          task_type=task_type,
          task_id=task_id,
          session_config=session_config,
          rpc_layer=rpc_layer,
          environment=environment)
    if task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
      if strategy.extended.experimental_between_graph:
        return _run_single_worker(worker_fn, strategy, cluster_spec, task_type,
                                  task_id, session_config, rpc_layer)
      else:
        context = _WorkerContext(strategy, cluster_spec, task_type, task_id)
        if context.is_chief:
          return _run_single_worker(worker_fn, strategy, cluster_spec, None,
                                    None, session_config, rpc_layer)
        else:
          server.join()
    elif task_type == _TaskType.EVALUATOR:
      return _run_single_worker(eval_fn, eval_strategy, cluster_spec, task_type,
                                task_id, session_config, rpc_layer)
    else:
      if task_type != _TaskType.PS:
        raise ValueError("Unexpected task_type: %r" % task_type)
      server.join()
def normalize_cluster_spec(cluster_spec):
  if isinstance(cluster_spec, (dict, cluster_pb2.ClusterDef)):
    return server_lib.ClusterSpec(cluster_spec)
  elif not isinstance(cluster_spec, server_lib.ClusterSpec):
    raise ValueError(
        "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
        "`tf.train.ClusterDef` object")
  return cluster_spec
