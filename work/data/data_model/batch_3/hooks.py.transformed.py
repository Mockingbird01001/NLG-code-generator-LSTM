
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook
class LocalCLIDebugHook(session_run_hook.SessionRunHook):
  def __init__(self,
               ui_type="curses",
               dump_root=None,
               thread_name_filter=None,
               config_file_path=None):
    """Create a local debugger command-line interface (CLI) hook.
    Args:
      ui_type: (`str`) requested user-interface type. Currently supported:
        (curses | readline).
      dump_root: (`str`) optional path to the dump root directory. Must be a
        directory that does not exist or an empty directory. If the directory
        does not exist, it will be created by the debugger core during debug
        `run()` calls and removed afterwards.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      config_file_path: Optional override to the default configuration file
        path, which is at `${HOME}/.tfdbg_config`.
    """
    self._ui_type = ui_type
    self._dump_root = dump_root
    self._thread_name_filter = thread_name_filter
    self._session_wrapper = None
    self._pending_tensor_filters = {}
    self._config_file_path = config_file_path
  def add_tensor_filter(self, filter_name, tensor_filter):
    """Add a tensor filter.
    See doc of `LocalCLIDebugWrapperSession.add_tensor_filter()` for details.
    Override default behavior to accommodate the possibility of this method
    being
    called prior to the initialization of the underlying
    `LocalCLIDebugWrapperSession` object.
    Args:
      filter_name: See doc of `LocalCLIDebugWrapperSession.add_tensor_filter()`
        for details.
      tensor_filter: See doc of
        `LocalCLIDebugWrapperSession.add_tensor_filter()` for details.
    """
    if self._session_wrapper:
      self._session_wrapper.add_tensor_filter(filter_name, tensor_filter)
    else:
      self._pending_tensor_filters[filter_name] = tensor_filter
  def begin(self):
    pass
  def before_run(self, run_context):
    if not self._session_wrapper:
      self._session_wrapper = local_cli_wrapper.LocalCLIDebugWrapperSession(
          run_context.session,
          ui_type=self._ui_type,
          dump_root=self._dump_root,
          thread_name_filter=self._thread_name_filter,
          config_file_path=self._config_file_path)
      for filter_name in self._pending_tensor_filters:
        self._session_wrapper.add_tensor_filter(
            filter_name, self._pending_tensor_filters[filter_name])
    self._session_wrapper.increment_run_call_count()
    on_run_start_request = framework.OnRunStartRequest(
        run_context.original_args.fetches, run_context.original_args.feed_dict,
        None, None, self._session_wrapper.run_call_count)
    on_run_start_response = self._session_wrapper.on_run_start(
        on_run_start_request)
    self._performed_action = on_run_start_response.action
    run_args = session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=config_pb2.RunOptions())
    if self._performed_action == framework.OnRunStartAction.DEBUG_RUN:
      self._session_wrapper._decorate_run_options_for_debug(
          run_args.options,
          on_run_start_response.debug_urls,
          debug_ops=on_run_start_response.debug_ops,
          node_name_regex_allowlist=(
              on_run_start_response.node_name_regex_allowlist),
          op_type_regex_allowlist=(
              on_run_start_response.op_type_regex_allowlist),
          tensor_dtype_regex_allowlist=(
              on_run_start_response.tensor_dtype_regex_allowlist),
          tolerate_debug_op_creation_failures=(
              on_run_start_response.tolerate_debug_op_creation_failures))
    elif self._performed_action == framework.OnRunStartAction.PROFILE_RUN:
      self._session_wrapper._decorate_run_options_for_profile(run_args.options)
    return run_args
  def after_run(self, run_context, run_values):
    on_run_end_request = framework.OnRunEndRequest(self._performed_action,
                                                   run_values.run_metadata)
    self._session_wrapper.on_run_end(on_run_end_request)
class DumpingDebugHook(session_run_hook.SessionRunHook):
  def __init__(self,
               session_root,
               watch_fn=None,
               thread_name_filter=None,
               log_usage=True):
    """Create a local debugger command-line interface (CLI) hook.
    Args:
      session_root: See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
      watch_fn: See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      log_usage: (bool) Whether usage is to be logged.
    """
    self._session_root = session_root
    self._watch_fn = watch_fn
    self._thread_name_filter = thread_name_filter
    self._log_usage = log_usage
    self._session_wrapper = None
  def begin(self):
    pass
  def before_run(self, run_context):
    reset_disk_byte_usage = False
    if not self._session_wrapper:
      self._session_wrapper = dumping_wrapper.DumpingDebugWrapperSession(
          run_context.session,
          self._session_root,
          watch_fn=self._watch_fn,
          thread_name_filter=self._thread_name_filter,
          log_usage=self._log_usage)
      reset_disk_byte_usage = True
    self._session_wrapper.increment_run_call_count()
    debug_urls, watch_options = self._session_wrapper._prepare_run_watch_config(
        run_context.original_args.fetches, run_context.original_args.feed_dict)
    run_options = config_pb2.RunOptions()
    debug_utils.watch_graph(
        run_options,
        run_context.session.graph,
        debug_urls=debug_urls,
        debug_ops=watch_options.debug_ops,
        node_name_regex_allowlist=watch_options.node_name_regex_allowlist,
        op_type_regex_allowlist=watch_options.op_type_regex_allowlist,
        tensor_dtype_regex_allowlist=watch_options.tensor_dtype_regex_allowlist,
        tolerate_debug_op_creation_failures=(
            watch_options.tolerate_debug_op_creation_failures),
        reset_disk_byte_usage=reset_disk_byte_usage)
    run_args = session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=run_options)
    return run_args
  def after_run(self, run_context, run_values):
    pass
class GrpcDebugHook(session_run_hook.SessionRunHook):
  def __init__(self,
               grpc_debug_server_addresses,
               watch_fn=None,
               thread_name_filter=None,
               log_usage=True):
    """Constructs a GrpcDebugHook.
    Args:
      grpc_debug_server_addresses: (`list` of `str`) A list of the gRPC debug
        server addresses, in the format of <host:port>, with or without the
        "grpc://" prefix. For example: ["localhost:7000", "192.168.0.2:8000"]
      watch_fn: A function that allows for customizing which ops to watch at
        which specific steps. See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__` for details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      log_usage: (bool) Whether usage is to be logged.
    """
    self._grpc_debug_wrapper_session = None
    self._thread_name_filter = thread_name_filter
    self._grpc_debug_server_addresses = (
        grpc_debug_server_addresses
        if isinstance(grpc_debug_server_addresses, list) else
        [grpc_debug_server_addresses])
    self._watch_fn = watch_fn
    self._log_usage = log_usage
  def before_run(self, run_context):
    if not self._grpc_debug_wrapper_session:
      self._grpc_debug_wrapper_session = grpc_wrapper.GrpcDebugWrapperSession(
          run_context.session,
          self._grpc_debug_server_addresses,
          watch_fn=self._watch_fn,
          thread_name_filter=self._thread_name_filter,
          log_usage=self._log_usage)
    fetches = run_context.original_args.fetches
    feed_dict = run_context.original_args.feed_dict
    watch_options = self._watch_fn(fetches, feed_dict)
    run_options = config_pb2.RunOptions()
    debug_utils.watch_graph(
        run_options,
        run_context.session.graph,
        debug_urls=self._grpc_debug_wrapper_session.prepare_run_debug_urls(
            fetches, feed_dict),
        debug_ops=watch_options.debug_ops,
        node_name_regex_allowlist=watch_options.node_name_regex_allowlist,
        op_type_regex_allowlist=watch_options.op_type_regex_allowlist,
        tensor_dtype_regex_allowlist=watch_options.tensor_dtype_regex_allowlist,
        tolerate_debug_op_creation_failures=(
            watch_options.tolerate_debug_op_creation_failures))
    return session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=run_options)
class TensorBoardDebugHook(GrpcDebugHook):
  def __init__(self,
               grpc_debug_server_addresses,
               thread_name_filter=None,
               send_traceback_and_source_code=True,
               log_usage=True):
    """Constructor of TensorBoardDebugHook.
    Args:
      grpc_debug_server_addresses: gRPC address(es) of debug server(s), as a
        `str` or a `list` of `str`s. E.g., "localhost:2333",
        "grpc://localhost:2333", ["192.168.0.7:2333", "192.168.0.8:2333"].
      thread_name_filter: Optional filter for thread names.
      send_traceback_and_source_code: Whether traceback of graph elements and
        the source code are to be sent to the debug server(s).
      log_usage: Whether the usage of this class is to be logged (if
        applicable).
    """
    def _gated_grpc_watch_fn(fetches, feeds):
      return framework.WatchOptions(
          debug_ops=["DebugIdentity(gated_grpc=true)"])
    super(TensorBoardDebugHook, self).__init__(
        grpc_debug_server_addresses,
        watch_fn=_gated_grpc_watch_fn,
        thread_name_filter=thread_name_filter,
        log_usage=log_usage)
    self._grpc_debug_server_addresses = grpc_debug_server_addresses
    self._send_traceback_and_source_code = send_traceback_and_source_code
    self._sent_graph_version = -1
    grpc_wrapper.register_signal_handler()
  def before_run(self, run_context):
    if self._send_traceback_and_source_code:
      self._sent_graph_version = grpc_wrapper.publish_traceback(
          self._grpc_debug_server_addresses, run_context.session.graph,
          run_context.original_args.feed_dict,
          run_context.original_args.fetches, self._sent_graph_version)
    return super(TensorBoardDebugHook, self).before_run(run_context)
