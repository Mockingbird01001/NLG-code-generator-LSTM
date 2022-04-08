
import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import unittest
import six
_portpicker_import_error = None
try:
  _portpicker_import_error = _error
  portpicker = None
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
ASSIGNED_PORTS = set()
lock = threading.Lock()
def pick_unused_port():
  if _portpicker_import_error:
  global ASSIGNED_PORTS
  with lock:
    while True:
      try:
        port = portpicker.pick_unused_port()
      except portpicker.NoFreePortFoundError:
        raise unittest.SkipTest('Flakes in portpicker library do not represent '
                                'TensorFlow errors.')
      if port > 10000 and port not in ASSIGNED_PORTS:
        ASSIGNED_PORTS.add(port)
        logging.info('Using local port %r', port)
        return port
def _create_cluster(num_workers,
                    num_ps,
                    has_chief=False,
                    has_eval=False,
                    protocol='grpc',
                    worker_config=None,
                    ps_config=None,
                    eval_config=None,
                    worker_name='worker',
                    ps_name='ps',
                    chief_name='chief'):
  if _portpicker_import_error:
  worker_ports = [pick_unused_port() for _ in range(num_workers)]
  ps_ports = [pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {}
  if num_workers > 0:
    cluster_dict[worker_name] = ['localhost:%s' % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict[ps_name] = ['localhost:%s' % port for port in ps_ports]
  if has_eval:
    cluster_dict['evaluator'] = ['localhost:%s' % pick_unused_port()]
  if has_chief:
    cluster_dict[chief_name] = ['localhost:%s' % pick_unused_port()]
  cs = server_lib.ClusterSpec(cluster_dict)
  for i in range(num_workers):
    server_lib.Server(
        cs,
        job_name=worker_name,
        protocol=protocol,
        task_index=i,
        config=worker_config,
        start=True)
  for i in range(num_ps):
    server_lib.Server(
        cs,
        job_name=ps_name,
        protocol=protocol,
        task_index=i,
        config=ps_config,
        start=True)
  if has_chief:
    server_lib.Server(
        cs,
        job_name=chief_name,
        protocol=protocol,
        task_index=0,
        config=worker_config,
        start=True)
  if has_eval:
    server_lib.Server(
        cs,
        job_name='evaluator',
        protocol=protocol,
        task_index=0,
        config=eval_config,
        start=True)
  return cluster_dict
def create_in_process_cluster(num_workers,
                              num_ps,
                              has_chief=False,
                              has_eval=False,
                              rpc_layer='grpc'):
  gpu_mem_frac = 0.7 / (num_workers + int(has_chief) + int(has_eval))
  worker_config = config_pb2.ConfigProto()
  worker_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac
  if worker_config.inter_op_parallelism_threads < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1
  if has_chief:
    worker_config.experimental.collective_group_leader = (
        '/job:chief/replica:0/task:0')
  else:
    worker_config.experimental.collective_group_leader = (
        '/job:worker/replica:0/task:0')
  ps_config = config_pb2.ConfigProto()
  ps_config.device_count['GPU'] = 0
  eval_config = config_pb2.ConfigProto()
  eval_config.experimental.collective_group_leader = ''
  cluster = None
  try:
    cluster = _create_cluster(
        num_workers,
        num_ps=num_ps,
        has_chief=has_chief,
        has_eval=has_eval,
        worker_config=worker_config,
        ps_config=ps_config,
        eval_config=eval_config,
        protocol=rpc_layer)
  except errors.UnknownError as e:
    if 'Could not start gRPC server' in e.message:
      raise unittest.SkipTest('Cannot start std servers.')
    else:
      raise
  return cluster
class MultiProcessCluster(object):
  def __init__(self,
               cluster_resolver,
               stream_output=False,
               collective_leader=None):
    self._cluster_resolver = cluster_resolver
    self._cluster_spec = cluster_resolver.cluster_spec().as_dict()
    self._rpc_layer = cluster_resolver.rpc_layer
    self._stream_output = stream_output
    self._start_events = {}
    self._finish_events = {}
    self._mpr_manager = multi_process_runner.manager()
    def task_function(start_events, finish_events):
      cluster_resolver = TFConfigClusterResolver()
      cluster_spec = cluster_resolver.cluster_spec()
      task_type = cluster_resolver.task_type
      task_id = cluster_resolver.task_id
      rpc_layer = cluster_resolver.rpc_layer
      server_config = config_pb2.ConfigProto()
      server_config.device_count['GPU'] = 0
      if collective_leader:
        server_config.experimental.collective_group_leader = collective_leader
        server_config.experimental.collective_nccl = False
        logging.info(
            'Enabling collective ops with cluster_spec = %r, task_type = %r, '
            'task_id = %r, rpc_layer = %r, collective_leader = %s',
            cluster_spec, task_type, task_id, rpc_layer, collective_leader)
      else:
        logging.info(
            'Starting server with cluster_spec = %r, task_type = %r, '
            'task_id = %r, rpc_layer = %r', cluster_spec, task_type, task_id,
            rpc_layer)
      server_lib.Server(
          cluster_spec,
          job_name=task_type,
          protocol=rpc_layer,
          task_index=task_id,
          config=server_config,
          start=True)
      start_event = start_events[task_type][task_id]
      start_event.set()
      finish_event = finish_events[task_type][task_id]
      finish_event.wait()
    self._task_function = task_function
    self._mpr = None
  def start(self):
    if self._mpr:
      raise ValueError('The cluster has already been started.')
    for task_type, task_addresses in self._cluster_spec.items():
      self._start_events[task_type] = []
      self._finish_events[task_type] = []
      for _ in task_addresses:
        self._start_events[task_type].append(self._mpr_manager.Event())
        self._finish_events[task_type].append(self._mpr_manager.Event())
    self._mpr = multi_process_runner.MultiProcessRunner(
        self._task_function,
        self._cluster_spec,
        args=(self._start_events, self._finish_events),
        rpc_layer=self._rpc_layer,
        stream_output=self._stream_output,
        return_output=False,
        use_dill_for_args=False)
    self._mpr.start()
    for task_type, task_addresses in self._cluster_spec.items():
      for i in range(len(task_addresses)):
        self._start_events[task_type][i].wait()
  def stop(self):
    for task_type, task_addresses in self._cluster_spec.items():
      for i in range(len(task_addresses)):
        self._finish_events[task_type][i].set()
    try:
      self._mpr.join()
    except multi_process_runner.UnexpectedSubprocessExitError:
      pass
    self._mpr = None
    self._start_events = {}
    self._finish_events = {}
  def kill_task(self, task_type, task_id):
    assert self._mpr
    if (not self._start_events[task_type][task_id].is_set() or
        self._finish_events[task_type][task_id].is_set()):
      raise ValueError("The task %s:%d doesn't exist." % (task_type, task_id))
    self._finish_events[task_type][task_id].set()
    self._mpr._processes[(task_type, task_id)].join()
  def start_task(self, task_type, task_id):
    assert self._mpr
    if (not self._start_events[task_type][task_id].is_set() or
        not self._finish_events[task_type][task_id].is_set()):
      raise ValueError(
          'The task %s:%d is still alive. You cannot start another one.' %
          (task_type, task_id))
    self._start_events[task_type][task_id] = self._mpr_manager.Event()
    self._finish_events[task_type][task_id] = self._mpr_manager.Event()
    self._mpr.start_single_process(task_type=task_type, task_id=task_id)
    self._start_events[task_type][task_id].wait()
  @property
  def cluster_resolver(self):
    return copy.deepcopy(self._cluster_resolver)
def create_multi_process_cluster(num_workers,
                                 num_ps,
                                 has_chief=False,
                                 has_eval=False,
                                 rpc_layer='grpc',
                                 stream_output=False,
                                 collective_leader=None):
  logging.info('Now creating a MultiProcessCluster with '
               f'num_workers={num_workers}, num_ps={num_ps}.')
  cluster_spec = create_cluster_spec(
      has_chief=has_chief,
      num_workers=num_workers,
      num_ps=num_ps,
      has_eval=has_eval)
  cluster = MultiProcessCluster(
      SimpleClusterResolver(
          server_lib.ClusterSpec(cluster_spec), rpc_layer=rpc_layer),
      stream_output=stream_output,
      collective_leader=collective_leader)
  cluster.start()
  return cluster
@tf_export(
    '__internal__.distribute.multi_process_runner.create_cluster_spec', v1=[])
def create_cluster_spec(has_chief=False,
                        num_workers=1,
                        num_ps=0,
                        has_eval=False):
  """Create a cluster spec with tasks with unused local ports.
  This utility finds available ports at localhost, and returns a dict that
  represents the cluster spec that utilizes those ports, according to the
  arguments. The dict representing the cluster spec contains task types, and
  their instances' addresses. Note that this is usually only for testing purpose
  using multiple processes in the local machine, and should not be used for real
  multi-worker TensorFlow programs, where the addresses need to point to the
  processes at separate machines.
  This util is useful when creating the `cluster_spec` arg for
  `tf.__internal__.distribute.multi_process_runner.run`.
  Args:
    has_chief: Whether the generated cluster spec should contain "chief" task
      type.
    num_workers: Number of workers to use in the cluster spec.
    num_ps: Number of parameter servers to use in the cluster spec.
    has_eval: Whether this cluster spec has evaluator.
  Returns:
    A dict that represents the cluster spec using localhost ports for the tasks.
  Example:
  ```python
  cluster_spec =
  tf.__internal__.distribute.multi_process_runner.create_cluster_spec(
      has_chief=True, num_workers=2, num_ps=2)
  cluster_spec =
  tf.__internal__.distribute.multi_process_runner.create_cluster_spec(
      has_chief=False, num_workers=0, num_ps=0, has_eval=True)
  ```
  """
  if _portpicker_import_error:
  cluster_spec = {}
  if has_chief:
    cluster_spec['chief'] = ['localhost:%s' % pick_unused_port()]
  if num_workers:
    cluster_spec['worker'] = [
        'localhost:%s' % pick_unused_port() for _ in range(num_workers)
    ]
  if num_ps:
    cluster_spec['ps'] = [
        'localhost:%s' % pick_unused_port() for _ in range(num_ps)
    ]
  if has_eval:
    cluster_spec['evaluator'] = ['localhost:%s' % pick_unused_port()]
  return cluster_spec
@contextlib.contextmanager
def skip_if_grpc_server_cant_be_started(test_obj):
  try:
    yield
  except errors.UnknownError as e:
    if 'Could not start gRPC server' in e.message:
      reason = 'Cannot start std servers.'
      test_obj.test_skipped_reason = reason
      test_obj.skipTest(reason)
    else:
      raise
class MultiWorkerTestBase(test.TestCase):
  @classmethod
    cls._cluster_spec = create_in_process_cluster(num_workers=num_workers,
                                                  num_ps=num_ps)
    cls._default_target = 'grpc://' + cls._cluster_spec['worker'][0]
  def setUp(self):
    self._thread_local = threading.local()
    self._thread_local.cached_session = None
    self._coord = coordinator.Coordinator()
  @contextlib.contextmanager
  def session(self, graph=None, config=None, target=None):
    config = self._create_config(config)
    if target is None:
      target = self._default_target
    with session.Session(graph=graph, config=config, target=target) as sess:
      yield sess
  @contextlib.contextmanager
  def cached_session(self, graph=None, config=None, target=None):
    config = self._create_config(config)
    if target is None:
      target = self._default_target
    if getattr(self._thread_local, 'cached_session', None) is None:
      self._thread_local.cached_session = session.Session(
          graph=None, config=config, target=target)
    sess = self._thread_local.cached_session
    with sess.graph.as_default(), sess.as_default():
      yield sess
  def _create_config(self, config):
    if config is None:
      config = config_pb2.ConfigProto(allow_soft_placement=True)
    else:
      config = copy.deepcopy(config)
    config.graph_options.optimizer_options.opt_level = -1
    config.graph_options.rewrite_options.constant_folding = (
        rewriter_config_pb2.RewriterConfig.OFF)
    return config
  def _run_client(self, client_fn, task_type, task_id, num_gpus, eager_mode,
                  *args, **kwargs):
    def wrapped_client_fn():
      with self._coord.stop_on_exception():
        client_fn(task_type, task_id, num_gpus, *args, **kwargs)
    if eager_mode:
      with context.eager_mode():
        wrapped_client_fn()
    else:
      with context.graph_mode():
        wrapped_client_fn()
  def _run_between_graph_clients(self, client_fn, cluster_spec, num_gpus, *args,
                                 **kwargs):
    threads = []
    for task_type in ['chief', 'worker']:
      for task_id in range(len(cluster_spec.get(task_type, []))):
        t = threading.Thread(
            target=self._run_client,
            args=(client_fn, task_type, task_id, num_gpus,
                  context.executing_eagerly()) + args,
            kwargs=kwargs)
        t.start()
        threads.append(t)
    self._coord.join(threads)
class SingleWorkerTestBaseGraph(MultiWorkerTestBase):
  @classmethod
  def setUpClass(cls):
    super(SingleWorkerTestBaseGraph, cls).setUpClass(num_workers=1)
class SingleWorkerTestBaseEager(test.TestCase):
  def setUp(self):
    super(SingleWorkerTestBaseEager, self).setUp()
    workers, _ = test_util.create_local_cluster(num_workers=1, num_ps=0)
    remote.connect_to_remote_host(workers[0].target)
  def cached_session(self):
    return DummySession()
class DummySession(object):
  def __enter__(self):
    return
  def __exit__(self, exception_type, exception_value, traceback):
    pass
class MockOsEnv(collections_abc.Mapping):
  def __init__(self, *args):
    self._dict = dict()
    self._thread_local = threading.local()
    super(MockOsEnv, self).__init__(*args)
  def get(self, key, default=None):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    if key == 'TF_CONFIG':
      return dict.get(self._thread_local.dict, key, default)
    else:
      return dict.get(self._dict, key, default)
  def __getitem__(self, key):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    if key == 'TF_CONFIG':
      return dict.__getitem__(self._thread_local.dict, key)
    else:
      return dict.__getitem__(self._dict, key)
  def __setitem__(self, key, val):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    if key == 'TF_CONFIG':
      return dict.__setitem__(self._thread_local.dict, key, val)
    else:
      return dict.__setitem__(self._dict, key, val)
  def __iter__(self):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    for x in self._thread_local.dict:
      yield x
    for x in self._dict:
      yield x
  def __len__(self):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    return self._thread_local.dict.__len__() + self._dict.__len__()
class IndependentWorkerTestBase(test.TestCase):
  def _make_mock_run_std_server(self):
    def _mock_run_std_server(*args, **kwargs):
      with skip_if_grpc_server_cant_be_started(self):
        ret = original_run_std_server(*args, **kwargs)
      if not getattr(self._thread_local, 'server_started', False):
        self._barrier.wait()
      self._thread_local.server_started = True
      return ret
    return _mock_run_std_server
  def setUp(self):
    self._mock_os_env = MockOsEnv()
    self._mock_context = test.mock.patch.object(os, 'environ',
                                                self._mock_os_env)
    self._coord = coordinator.Coordinator()
    super(IndependentWorkerTestBase, self).setUp()
    self._mock_context.__enter__()
    self._thread_local = threading.local()
  def tearDown(self):
    self._mock_context.__exit__(None, None, None)
    super(IndependentWorkerTestBase, self).tearDown()
  def _task_thread(self, task_fn, tf_config, executing_eagerly, *args,
                   **kwargs):
    with self._coord.stop_on_exception():
      os.environ['TF_CONFIG'] = json.dumps(tf_config)
      if executing_eagerly:
        with context.eager_mode():
          task_fn(*args, **kwargs)
      else:
        with ops.Graph().as_default(), context.graph_mode():
          task_fn(*args, **kwargs)
  def _run_task_in_thread(self, task_fn, cluster_spec, task_type, task_id,
                          *args, **kwargs):
    tf_config = kwargs.pop('tf_config', None)
    if tf_config is None:
      if task_type:
        tf_config = {
            'cluster': cluster_spec,
            'task': {
                'type': task_type,
                'index': task_id
            }
        }
      else:
        tf_config = {
            'cluster': cluster_spec,
        }
    t = threading.Thread(
        target=self._task_thread,
        args=(task_fn, tf_config, context.executing_eagerly()) + args,
        kwargs=kwargs)
    t.start()
    return t
  def run_multiple_tasks_in_threads(self, task_fn, cluster_spec, *args,
                                    **kwargs):
    threads = {}
    for task_type in cluster_spec.keys():
      threads[task_type] = []
      for task_id in range(len(cluster_spec[task_type])):
        t = self._run_task_in_thread(task_fn, cluster_spec, task_type, task_id,
                                     *args, **kwargs)
        threads[task_type].append(t)
    return threads
  def join_independent_workers(self, worker_threads):
    with skip_if_grpc_server_cant_be_started(self):
      self._coord.join(worker_threads)
class MultiWorkerMultiProcessTest(test.TestCase):
  def _run_task_in_process(self, cmd_args, cluster_spec, task_type, task_id):
    env = os.environ.copy()
    env['TF_CONFIG'] = json.dumps({
        'cluster': cluster_spec,
        'task': {
            'type': task_type,
            'index': task_id
        }
    })
    return subprocess.Popen(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
  @deprecation.deprecated(
      None, '`run_multiple_tasks_in_processes` is deprecated; any new test '
      'requiring multiple processes should use `multi_process_runner` for '
      'better support of log printing, streaming, and more functionality.')
  def run_multiple_tasks_in_processes(self, cmd_args, cluster_spec):
    processes = {}
    for task_type in cluster_spec.keys():
      processes[task_type] = []
      for task_id in range(len(cluster_spec[task_type])):
        p = self._run_task_in_process(cmd_args, cluster_spec, task_type,
                                      task_id)
        processes[task_type].append(p)
    return processes
  @deprecation.deprecated(
      None, '`join_independent_workers` is deprecated; any new test '
      'requiring multiple processes should use `multi_process_runner` for '
      'better support of log printing, streaming, and more functionality.')
  def join_independent_workers(self, worker_processes):
    return_codes = []
    for p in nest.flatten(worker_processes):
      try:
        p.communicate()
      except ValueError:
        pass
      finally:
        return_codes.append(p.returncode)
    for return_code in return_codes:
      self.assertEqual(return_code, 0)
  @deprecation.deprecated(
      None, '`stream_stderr` is deprecated; any new test '
      'requiring multiple processes should use `multi_process_runner` for '
      'better support of log printing, streaming, and more functionality.')
  def stream_stderr(self, processes, print_only_first=False):
    def _stream_stderr_single_process(process, type_string, index,
                                      print_to_stdout):
      while True:
        output = process.stderr.readline()
        if not output and process.poll() is not None:
          break
        if output and print_to_stdout:
          print('{}{} {}'.format(type_string, index, output.strip()))
          sys.stdout.flush()
    stream_threads = []
    for process_type, process_list in six.iteritems(processes):
      for i in range(len(process_list)):
        print_to_stdout = (not print_only_first) or (i == 0)
        thread = threading.Thread(
            target=_stream_stderr_single_process,
            args=(process_list[i], process_type, i, print_to_stdout))
        thread.start()
        stream_threads.append(thread)
    for thread in stream_threads:
      thread.join()
def get_tf_config_task():
  return json.loads(os.environ['TF_CONFIG'])['task']
def get_tf_config_cluster_spec():
  return json.loads(os.environ['TF_CONFIG'])['cluster']
def get_task_type():
  return get_tf_config_task()['type']
def get_task_index():
  return get_tf_config_task()['index']
def is_chief():
  return ('chief' not in get_tf_config_cluster_spec()
          and get_task_type() == 'worker'
          and get_task_index() == 0)
