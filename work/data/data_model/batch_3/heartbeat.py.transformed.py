
"""A heartbeat service (go/dtensor-heartbeat) periodically pinging all workers.
In normal cases, all workers will exchange the same randomly generated number
until normal program termination. If any worker stops or restarts, other workers
will detect that and crash themselves.
In this module, logging.fatal is used to guarantee a worker crash no matter how
the functions below are called, in a thread or not.
"""
import atexit
import threading
import time
import numpy as np
from tensorflow.dtensor import python as dtensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops.collective_ops import all_reduce
from tensorflow.python.platform import tf_logging as logging
_CONSECUTIVE_FAILURES_LIMIT = 3
_failure_count = 0
_heartbeat_timer = None
def _heartbeat(
    timer: threading.Event,
    token: int,
    num_tasks: int,
    task_id: int,
    device: tf_device.DeviceSpec,
):
  logging.info('Starting a heartbeat thread')
  global _failure_count
  while True:
    if timer.wait(period):
      logging.info('Exiting the heartbeat thread normally')
      return
    signal = np.zeros([num_tasks], dtype=np.int32)
    signal[task_id] = token
    logging.vlog(2, 'Sending heartbeat signal %s', signal)
    try:
      with ops.device(device):
        signal = all_reduce(
            constant_op.constant(signal),
            group_size=num_tasks,
            group_key=0,
            instance_key=0,
            timeout=max(period - 10, 2)).numpy()
      _failure_count += 1
      if _failure_count < _CONSECUTIVE_FAILURES_LIMIT:
        logging.warning('Heartbeat failure %d, %d more until limit: %s',
                        _failure_count,
                        _CONSECUTIVE_FAILURES_LIMIT - _failure_count, e)
      else:
        logging.fatal('Heartbeat failure %d, limit of %d reached: %s',
                      _failure_count, _CONSECUTIVE_FAILURES_LIMIT, e)
    logging.vlog(2, 'Received heartbeat signal %s', signal)
    if not np.all(signal == token):
      logging.fatal('Unexpected heartbeat signal received: %s', signal)
    _failure_count = 0
def start(period: int) -> threading.Event:
  """Starts a persistent thread exchanging heartbeats between workers.
  Args:
    period: Heartbeat interval in seconds. Heartbeat timeout is set to the
      larger of `period` - 10 and 2s.
  Returns:
    A threading.Event object. Users can choose to call its set() method to shut
    down the heartbeat service gracefully. This isn't necessary in most cases,
    because the heartbeat service automatically shuts down at successful program
    exit through atexit handlers. But in situations when atexit handlers are not
    invoked, such as when multiprocessing processes exit in tests, users can
    manually request a shutdown.
  """
  global _heartbeat_timer
  if _heartbeat_timer is not None:
    logging.warning('A heartbeat thread is already running, skipping this one.')
    return _heartbeat_timer
  task_id = dtensor.client_id()
  num_tasks = dtensor.num_clients()
  if task_id == 0:
    signal = np.full([num_tasks], token, dtype=np.int32)
  else:
    signal = np.zeros([num_tasks], dtype=np.int32)
  logging.info('Initial heartbeat signal: %s', signal)
  device = tf_device.DeviceSpec(
      job=dtensor.job_name(),
      replica=0,
      task=task_id,
      device_type='CPU',
      device_index=0)
  with ops.device(device):
    signal = all_reduce(
        constant_op.constant(signal),
        group_size=num_tasks,
        group_key=0,
        instance_key=0,
        timeout=max(period - 10, 2)).numpy()
  logging.info('Merged heartbeat signal %s', signal)
  if task_id == 0:
    if not np.all(signal == token):
      logging.fatal('Merged heartbeat signal has value != %d', token)
  else:
    if len(set(signal)) != 1:
      logging.fatal('Merged heartbeat signal has unequal elements')
    token = signal[0]
  _heartbeat_timer = threading.Event()
  def stop_heartbeat():
    logging.info('Stopping the heartbeat thread')
    _heartbeat_timer.set()
    time.sleep(max(period // 10, 2))
  atexit.register(stop_heartbeat)
  thread = threading.Thread(
      target=_heartbeat,
      args=[period, _heartbeat_timer, token, num_tasks, task_id, device],
      daemon=True)
  thread.start()
  return _heartbeat_timer
