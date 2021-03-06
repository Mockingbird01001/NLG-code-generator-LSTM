
import os
import threading
import time
from typing import Any, List, Optional, Text
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.summary_io import SummaryWriterCache
class AsyncCheckpointSaverHook(basic_session_run_hooks.CheckpointSaverHook):
  def __init__(self,
               checkpoint_dir: Text,
               save_secs: Optional[int] = None,
               save_steps: Optional[int] = None,
               saver: Optional[saver_lib.Saver] = None,
               checkpoint_basename: Text = "model.ckpt",
               scaffold: Optional[monitored_session.Scaffold] = None,
               listeners: Optional[List[
                   basic_session_run_hooks.CheckpointSaverListener]] = None):
    save_path = os.path.join(checkpoint_dir, checkpoint_basename)
    logging.info("Create AsyncCheckpointSaverHook saving to path\n%s",
                 save_path)
    if listeners:
      logging.info(" with %d listener(s).", len(listeners))
    if saver is not None and scaffold is not None:
      raise ValueError("You cannot provide both saver and scaffold.")
    self._saver = saver
    self._save_thread = None
    self._write_graph_thread = None
    self._checkpoint_dir = checkpoint_dir
    self._save_path = save_path
    self._scaffold = scaffold
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)
    self._listeners = listeners or []
    self._steps_per_run = 1
    self._summary_writer = None
    self._global_step_tensor = None
    self._last_checkpoint_step = None
  def _set_steps_per_run(self, steps_per_run):
    self._steps_per_run = steps_per_run
  def begin(self):
    self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use CheckpointSaverHook.")
    for l in self._listeners:
      l.begin()
  def after_create_session(self, session: session_lib.Session, coord: Any):
    global_step = session.run(self._global_step_tensor)
    def _write_graph_fn(self):
      training_util.write_graph(
          ops.get_default_graph().as_graph_def(add_shapes=True),
          self._checkpoint_dir, "graph.pbtxt")
    self._write_graph_thread = threading.Thread(target=_write_graph_fn,
                                                args=[self])
    self._write_graph_thread.start()
    saver_def = self._get_saver().saver_def if self._get_saver() else None
    graph = ops.get_default_graph()
    meta_graph_def = meta_graph.create_meta_graph_def(
        graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
    self._summary_writer.add_graph(graph)
    self._summary_writer.add_meta_graph(meta_graph_def)
    self._save(session, global_step)
    self._timer.update_last_triggered_step(global_step)
    return session_run_hook.SessionRunArgs(self._global_step_tensor)
  def after_run(self, run_context: session_run_hook.SessionRunContext,
                run_values: Any):
    global_step = run_context.session.run(self._global_step_tensor)
    if self._timer.should_trigger_for_step(global_step):
      self._timer.update_last_triggered_step(global_step)
      logging.info("Triggering checkpoint. %s", global_step)
      if self._save(run_context.session, global_step):
        run_context.request_stop()
  def end(self, session: session_lib.Session):
    if self._save_thread:
      logging.info("Waiting for any pending checkpoints to finish.")
      self._save_thread.join()
    if self._write_graph_thread:
      logging.info("Waiting for any pending write_graph to finish.")
      self._write_graph_thread.join()
    last_step = session.run(self._global_step_tensor)
    if self._last_checkpoint_step != last_step:
      self._save(session, last_step, asynchronous=False)
    for l in self._listeners:
      l.end(session, last_step)
  def _save(self, session, step, asynchronous=True):
    def _save_fn():
      logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
      start_time = time.time()
      for l in self._listeners:
        l.before_save(session, step)
      self._get_saver().save(session, self._save_path, global_step=step)
      self._summary_writer.add_session_log(
          event_pb2.SessionLog(
              status=event_pb2.SessionLog.CHECKPOINT,
              checkpoint_path=self._save_path), step)
      for l in self._listeners:
        l.after_save(session, step)
      end_time = time.time()
      logging.info("Checkpoint actual writing time: (%.3f sec)",
                   end_time - start_time)
      logging.info("Checkpoint finished for %d into %s.", step, self._save_path)
    if not asynchronous:
      self._last_checkpoint_step = step
      _save_fn()
      return
    if self._save_thread is not None:
      self._save_thread.join(timeout=0.1)
      if self._save_thread.is_alive():
        logging.info("Saver thread still in progress, skipping checkpoint.")
        return
    self._last_checkpoint_step = step
    self._save_thread = threading.Thread(target=_save_fn)
    self._save_thread.start()
  def _get_saver(self):
    if self._saver is not None:
      return self._saver
    elif self._scaffold is not None:
      return self._scaffold.saver
    collection_key = ops.GraphKeys.SAVERS
    savers = ops.get_collection(collection_key)
    if not savers:
      raise RuntimeError(
          "No items in collection {}. Please add a saver to the collection "
          "or provide a saver or scaffold.".format(collection_key))
    elif len(savers) > 1:
      raise RuntimeError(
          "More than one item in collection {}. "
          "Please indicate which one to use by passing it to the constructor."
          .format(collection_key))
    self._saver = savers[0]
    return savers[0]
