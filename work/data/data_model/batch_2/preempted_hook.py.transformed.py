
import logging as _logging
import os
import threading
import time
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
class CloudTPUPreemptedHook(session_run_hook.SessionRunHook):
  def __init__(self, cluster):
    self._cluster = cluster
  def after_create_session(self, session, coord):
    if tpu_cluster_resolver.is_running_in_gce():
      self._tpu_poller = _TPUPollingThread(self._cluster, session)
      self._tpu_poller.start()
  def end(self, session):
    self._tpu_poller.stop()
class _TPUPollingThread(threading.Thread):
  """A thread that polls the state of a TPU node.
  When the node transitions into a TERMINAL state (PREEMPTED, TERMINATED)
  that's considered as not recoverable by the underlying infrastructure,
  it attempts to close the session, and exits the entire process if the
  session.close() stucks.
  """
  def __init__(self, cluster, session):
    super(_TPUPollingThread, self).__init__()
    self.daemon = True
    self._running = True
    self._session_closed = False
    self._cluster = cluster
    self._session = session
    self._interval = 30
    for name in ['googleapiclient.discovery', 'oauth2client.client']:
      _logging.getLogger(name).setLevel(_logging.WARNING)
  def stop(self):
    self._running = False
    self._session_closed = True
    self.join()
  def run(self):
    if not tpu_cluster_resolver.is_running_in_gce():
      logging.warning(
          'TPUPollingThread is running in a non-GCE environment, exiting...')
      self._running = False
      return
    while self._running:
      if not recoverable:
        logging.warning(
            'TPUPollingThread found TPU %s in state %s',
      time.sleep(self._interval)
