
import collections
import os.path
import sys
import threading
import time
import six
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
class EventFileWriter(object):
  def __init__(self, logdir, max_queue=10, flush_secs=120,
               filename_suffix=None):
    self._logdir = str(logdir)
    gfile.MakeDirs(self._logdir)
    self._max_queue = max_queue
    self._flush_secs = flush_secs
    self._flush_complete = threading.Event()
    self._flush_sentinel = object()
    self._close_sentinel = object()
    self._ev_writer = _pywrap_events_writer.EventsWriter(
        compat.as_bytes(os.path.join(self._logdir, "events")))
    if filename_suffix:
      self._ev_writer.InitWithSuffix(compat.as_bytes(filename_suffix))
    self._initialize()
    self._closed = False
  def _initialize(self):
    self._event_queue = CloseableQueue(self._max_queue)
    self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                      self._flush_secs, self._flush_complete,
                                      self._flush_sentinel,
                                      self._close_sentinel)
    self._worker.start()
  def get_logdir(self):
    return self._logdir
  def reopen(self):
    """Reopens the EventFileWriter.
    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.
    Does nothing if the EventFileWriter was not closed.
    """
    if self._closed:
      self._initialize()
      self._closed = False
  def add_event(self, event):
    if not self._closed:
      self._try_put(event)
  def _try_put(self, item):
    try:
      self._event_queue.put(item)
    except QueueClosedError:
      self._internal_close()
      if self._worker.failure_exc_info:
  def flush(self):
    if not self._closed:
      self._flush_complete.clear()
      self._try_put(self._flush_sentinel)
      self._flush_complete.wait()
      if self._worker.failure_exc_info:
        self._internal_close()
  def close(self):
    if not self._closed:
      self.flush()
      self._try_put(self._close_sentinel)
      self._internal_close()
  def _internal_close(self):
    self._closed = True
    self._worker.join()
    self._ev_writer.Close()
class _EventLoggerThread(threading.Thread):
  def __init__(self, queue, ev_writer, flush_secs, flush_complete,
               flush_sentinel, close_sentinel):
    threading.Thread.__init__(self, name="EventLoggerThread")
    self.daemon = True
    self._queue = queue
    self._ev_writer = ev_writer
    self._flush_secs = flush_secs
    self._next_event_flush_time = 0
    self._flush_complete = flush_complete
    self._flush_sentinel = flush_sentinel
    self._close_sentinel = close_sentinel
    self.failure_exc_info = ()
  def run(self):
    try:
      while True:
        event = self._queue.get()
        if event is self._close_sentinel:
          return
        elif event is self._flush_sentinel:
          self._ev_writer.Flush()
          self._flush_complete.set()
        else:
          self._ev_writer.WriteEvent(event)
          now = time.time()
          if now > self._next_event_flush_time:
            self._ev_writer.Flush()
            self._next_event_flush_time = now + self._flush_secs
    except Exception as e:
      logging.error("EventFileWriter writer thread error: %s", e)
      self.failure_exc_info = sys.exc_info()
      raise
    finally:
      self._flush_complete.set()
      self._queue.close()
class CloseableQueue(object):
  def __init__(self, maxsize=0):
    self._maxsize = maxsize
    self._queue = collections.deque()
    self._closed = False
    self._mutex = threading.Lock()
    self._not_empty = threading.Condition(self._mutex)
    self._not_full = threading.Condition(self._mutex)
  def get(self):
    with self._not_empty:
      while not self._queue:
        self._not_empty.wait()
      item = self._queue.popleft()
      self._not_full.notify()
      return item
  def put(self, item):
    """Put an item into the queue.
    If the queue is closed, fails immediately.
    If the queue is full, blocks until space is available or until the queue
    is closed by a call to close(), at which point this call fails.
    Args:
      item: an item to add to the queue
    Raises:
      QueueClosedError: if insertion failed because the queue is closed
    """
    with self._not_full:
      if self._closed:
        raise QueueClosedError()
      if self._maxsize > 0:
        while len(self._queue) == self._maxsize:
          self._not_full.wait()
          if self._closed:
            raise QueueClosedError()
      self._queue.append(item)
      self._not_empty.notify()
  def close(self):
    with self._not_full:
      self._closed = True
      self._not_full.notify_all()
class QueueClosedError(Exception):
