
import faulthandler
import os
import sys
import threading
import time
from absl import logging
class WatchDog(object):
  def __init__(self,
               timeout=os.environ.get(
                   "TF_CLUSTER_COORDINATOR_WATCH_DOG_TIMEOUT", -1),
               traceback_file=sys.stdout,
               on_triggered=None):
    self._timeout = timeout
    self._last_activity_time = time.time()
    self._traceback_file = traceback_file
    self._on_triggered = on_triggered
    self._stopped = False
    if timeout > 0:
      self._watchdog_thread = threading.Thread(
          target=self._watchdog_function, name="WatchDog", daemon=True)
      self._watchdog_thread.start()
  def stop(self):
    self._stopped = True
  def _watchdog_function(self):
    logging.info("Starting watchdog thread with timeout %r", self._timeout)
    while not self._stopped:
      time.sleep(self._timeout / 10.0)
      current_time = time.time()
      if current_time - self._last_activity_time >= self._timeout:
        logging.warning(
            "No activity for ClusterCoordinator for %r seconds. "
            "Dumping stack traces.", self._timeout)
        if self._on_triggered:
          self._on_triggered()
        faulthandler.dump_traceback(file=self._traceback_file)
        self._traceback_file.write("==== End of stack traces ====\n")
        self._last_activity_time = current_time
  def report_closure_done(self):
    if self._timeout > 0:
      self._last_activity_time = time.time()
