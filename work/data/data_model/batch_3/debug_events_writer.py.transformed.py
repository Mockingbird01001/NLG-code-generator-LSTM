
import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
DEFAULT_CIRCULAR_BUFFER_SIZE = 1000
class DebugEventsWriter(object):
  def __init__(self,
               dump_root,
               tfdbg_run_id,
               circular_buffer_size=DEFAULT_CIRCULAR_BUFFER_SIZE):
    if not dump_root:
      raise ValueError("Empty or None dump root")
    self._dump_root = dump_root
    self._tfdbg_run_id = tfdbg_run_id
    _pywrap_debug_events_writer.Init(self._dump_root, self._tfdbg_run_id,
                                     circular_buffer_size)
  def WriteSourceFile(self, source_file):
    debug_event = debug_event_pb2.DebugEvent(source_file=source_file)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteSourceFile(self._dump_root, debug_event)
  def WriteStackFrameWithId(self, stack_frame_with_id):
    debug_event = debug_event_pb2.DebugEvent(
        stack_frame_with_id=stack_frame_with_id)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteStackFrameWithId(self._dump_root,
                                                      debug_event)
  def WriteGraphOpCreation(self, graph_op_creation):
    debug_event = debug_event_pb2.DebugEvent(
        graph_op_creation=graph_op_creation)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteGraphOpCreation(self._dump_root,
                                                     debug_event)
  def WriteDebuggedGraph(self, debugged_graph):
    debug_event = debug_event_pb2.DebugEvent(debugged_graph=debugged_graph)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteDebuggedGraph(self._dump_root, debug_event)
  def WriteExecution(self, execution):
    debug_event = debug_event_pb2.DebugEvent(execution=execution)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteExecution(self._dump_root, debug_event)
  def WriteGraphExecutionTrace(self, graph_execution_trace):
    debug_event = debug_event_pb2.DebugEvent(
        graph_execution_trace=graph_execution_trace)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteGraphExecutionTrace(
        self._dump_root, debug_event)
  def RegisterDeviceAndGetId(self, device_name):
    return _pywrap_debug_events_writer.RegisterDeviceAndGetId(
        self._dump_root, device_name)
  def FlushNonExecutionFiles(self):
    _pywrap_debug_events_writer.FlushNonExecutionFiles(self._dump_root)
  def FlushExecutionFiles(self):
    _pywrap_debug_events_writer.FlushExecutionFiles(self._dump_root)
  def Close(self):
    _pywrap_debug_events_writer.Close(self._dump_root)
  @property
  def dump_root(self):
    return self._dump_root
  def _EnsureTimestampAdded(self, debug_event):
    if debug_event.wall_time == 0:
      debug_event.wall_time = time.time()
