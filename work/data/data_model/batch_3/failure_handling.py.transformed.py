
import os
import signal
import sys
import threading
import time
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import gce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
_RUN_COUNT_KEY = 'RUN_TO_CHECKPOINT'
_ACKNOWLEDGE_KEY = 'RECEIVED_SIGNAL'
_ITERATION_VARIABLE = 'checkpointed_runs'
_RESTARTABLE_EXIT_CODE = 42
_STOP_WATCHING_CLUSTER_VALUE = 'STOP_WATCHER'
def _mwms_write_checkpoint_dir(checkpoint_dir, task_type, task_id,
                               cluster_spec):
  dirpath = os.path.dirname(checkpoint_dir)
  base = os.path.basename(checkpoint_dir)
  if not multi_worker_util.is_chief(
      cluster_spec=cluster_spec, task_type=task_type, task_id=task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    dirpath = os.path.join(dirpath, base_dirpath)
    gfile.MakeDirs(dirpath)
  return os.path.join(dirpath, base)
class TerminationConfig(object):
  """Configurations to customize for a platform other than Google's Borg or GCP.
  A TerminationConfig can be created and passed to the
  `WorkerPreemptionHandler` to provide customization based on the platform.
  It will deliver three pieces of information:
  * How to decide if there is a termination event soon
  The termination notification and how to fetch it varies across platforms. Thus
  we accept a user-defined function, `termination_watcher_function`, and execute
  it repeatedly to check for termination notification.
  `termination_watcher_function` should be a function that returns True if a
  termination notification has been made available and False otherwise. And the
  function should be lightweight and non-blocking so that we can clean up the
  resources properly if no termination signal is ever raised until training
  finishes.
  * How to exit the program
  We are asking for an `restart_code` to execute `sys.exit(restart_code)` after
  saving the checkpoint to exit the training program gracefully. A restart is
  inevitable to reset the program's state. However, you can configure the
  `restart_code` to facilitate the restart and make the training experience
  smooth. How so? Maybe your platform has an agreement to a RESTART_CODE that’s
  recognized as a program auto-restart signal, or you may have a coordinating
  script that starts up the training, in which you can configure the program to
  auto-restart if it ever exits with this RESTART_CODE. In both cases,
  you can pass in this RESTART_CODE and then wouldn’t even notice that the
  training has been interrupted and restarted.
  * How long do we have from receiving a termination event notice till the
  actual termination.
  Some platforms have the gap time as long as, say, one hour. In this case, you
  might want to utilize this time for training as much as possible until you
  have to save a checkpoint and exit. We can utilize this information if you
  pass it through the `time_till_termination` argument.
  *The default behavior*:
  If you are training with Google’s Borg system or GCP, we automatically detect
  the platform and make the right configuration for you. Besides these two
  platforms, the default behavior on an unrecognized platform is:
  * If `termination_event` is `None`, we will treat `signal.SIGTERM` as a
  termination event.
  * If `restart_code` not configured, we exit with an arbitrary choice, 42.
  * If `time_till_termination` is not configured, the default is 0, and we will
  wrap up the current training step, save a checkpoint, and exit the program as
  soon as we receive the termination signal.
  """
  def __init__(self,
               termination_watcher_function=None,
               restart_code=None,
               time_till_termination=None):
    self.termination_watcher_function = termination_watcher_function
    self.restart_code = restart_code
    self.time_till_termination = time_till_termination
class GCPTerminationConfig(TerminationConfig):
      self,
      termination_watcher_function=None,
      restart_code=None,
      time_till_termination=None):
    self.termination_watcher_function = termination_watcher_function or gce_util.termination_watcher_function_gce
    self.restart_code = restart_code or gce_util._RESTARTABLE_EXIT_CODE
    self.time_till_termination = time_till_termination or gce_util.GRACE_PERIOD_GCE
class BorgTerminationConfig(TerminationConfig):
      self,
      termination_watcher_function=None,
      restart_code=None,
      time_till_termination=None):
    self.termination_watcher_function = termination_watcher_function
    self.restart_code = restart_code or 42
    self.time_till_termination = time_till_termination or 0
def _complete_config_for_environement(platform_device, termination_config):
  if platform_device is gce_util.PlatformDevice.GCE_GPU:
    return GCPTerminationConfig(termination_config.termination_watcher_function,
                                termination_config.restart_code,
                                termination_config.time_till_termination)
  else:
    return BorgTerminationConfig(
        termination_config.termination_watcher_function,
        termination_config.restart_code,
        termination_config.time_till_termination)
class WorkerPreemptionHandler(object):
  """Preemption and error handler for synchronous training.
  The API helps coordinate all workers to save a checkpoint upon receiving a
  preemption signal and helps propagate accurate error messages during training.
  When the program recovers from preemption, the checkpoint that is passed to
  initialize a `WorkerPreemptionHandler` object will be loaded
  automatically.
  Right after the initialization, a thread starts to watch out for a termination
  signal for any member in the cluster, but the signal will only be handled
  (which includes aligning the step to save a checkpoint, saving a checkpoint,
  and exiting with a platform recognized restart code) after entering a
  `WorkerPreemptionHandler.run` call.
  Example usage:
  ```python
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  with strategy.scope():
    dataset, model, optimizer = ...
    fh_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    worker_preemption_watcher = tf.distribute.WorkerPreemptionHandler(
        cluster_resolver, fh_checkpoint, checkpoint_directory)
    for epoch in range(worker_preemption_watcher.total_runs //
                       STEPS_PER_EPOCH, num_epochs):
      for step in range(worker_preemption_watcher.total_runs %
                        STEPS_PER_EPOCH, num_steps):
        loss += worker_preemption_watcher.run(distributed_train_step,
                                                   args=(next(dataset),))
  ```
  `WorkerPreemptionHandler` will create a CheckpointManager to manage the
  checkpoint and only one CheckpointManager should be active in a particular
  directory at a time. Thus, if the user would like to save a checkpoint for
  purpose other than fault tolerance, e.g., for evaluation, they should save it
  in a directory different from the one passed to a
  `WorkerPreemptionHandler`.
  This API targets multi-client distributed training, and right now only
  `tf.distribute.MultiWorkerMirroredStrategy` is supported.
  """
  def __init__(self,
               cluster_resolver,
               checkpoint,
               checkpoint_dir,
               termination_config=TerminationConfig()):
    self._cluster_resolver = cluster_resolver
    self._checkpoint = checkpoint
    self._id_in_cluster = str(
        multi_worker_util.id_in_cluster(
            self._cluster_resolver.cluster_spec(),
            self._cluster_resolver.task_type,
            self._cluster_resolver.task_id))
    self._checkpointed_runs = variables.Variable(
        initial_value=constant_op.constant(0, dtype=dtypes.int64),
        trainable=False,
        name=_ITERATION_VARIABLE)
    if not hasattr(self._checkpoint,
                   _ITERATION_VARIABLE):
      setattr(self._checkpoint, _ITERATION_VARIABLE,
              self._checkpointed_runs)
    self._read_checkpoint_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1)
    if multi_worker_util.is_chief(
        cluster_spec=cluster_resolver.cluster_spec(),
        task_type=cluster_resolver.task_type,
        task_id=cluster_resolver.task_id):
      self._write_checkpoint_manager = self._read_checkpoint_manager
    else:
      self._write_checkpoint_manager = checkpoint_management.CheckpointManager(
          checkpoint,
          _mwms_write_checkpoint_dir(checkpoint_dir, cluster_resolver.task_type,
                                     cluster_resolver.task_id,
                                     cluster_resolver.cluster_spec()),
          max_to_keep=1)
    self._read_checkpoint_manager.restore_or_initialize()
    self._run_counter = self._checkpointed_runs.numpy()
    self._received_own_sigterm = threading.Event()
    self._received_sigterm_and_step = threading.Event()
    self._cluster_wise_termination_watcher_thread = threading.Thread(
        target=self._wait_for_signal,
        name='PeerTerminationWatcher-%s' % self._id_in_cluster,
        daemon=True)
    self._cluster_wise_termination_watcher_thread.start()
    logging.info('Start watcher for peer\'s signal.')
    self._poll_termination_signal_thread = None
    self._platform_device = gce_util.detect_platform()
    completed_termination_config = _complete_config_for_environement(
        self._platform_device, termination_config)
    self._termination_watcher_function = completed_termination_config.termination_watcher_function
    self._restart_code = completed_termination_config.restart_code
    self._time_till_termination = completed_termination_config.time_till_termination
    if completed_termination_config.termination_watcher_function:
      self._start_polling_for_termination_signal()
    else:
      self._start_watching_for_signal()
  def _start_watching_for_signal(self):
    signal.signal(signal.SIGTERM, self._sigterm_handler_fn)
  def _start_polling_for_termination_signal(self):
    self._poll_termination_signal_thread_should_stop = threading.Event()
    self._poll_termination_signal_thread = threading.Thread(
        target=self._poll_termination_signal,
        name='WorkerTerminationSignalWatcher-%s' % self._id_in_cluster,
        daemon=True)
    self._poll_termination_signal_thread.start()
    logging.info('Start polling for termination signal.')
  def _poll_termination_signal(self):
    while True:
      if self._poll_termination_signal_thread_should_stop.is_set():
        return
      if self._termination_watcher_function():
        self._signal_receipt_time = time.time()
        break
      time.sleep(1)
    logging.info('Member %s has received termination notice.',
                 self._id_in_cluster)
    self._received_own_sigterm.set()
  def _stop_poll_termination_signal_thread(self):
    if self._poll_termination_signal_thread:
      self._poll_termination_signal_thread_should_stop.set()
      self._poll_termination_signal_thread.join()
      self._poll_termination_signal_thread = None
      logging.info('Shut down watcher for one\'s own termination signal')
  def _stop_cluster_wise_termination_watcher_thread(self):
    if self._cluster_wise_termination_watcher_thread:
      try:
        if self._cluster_wise_termination_watcher_thread.is_alive():
          context.context().set_config_key_value(_RUN_COUNT_KEY,
                                                 _STOP_WATCHING_CLUSTER_VALUE)
        pass
      finally:
        self._cluster_wise_termination_watcher_thread.join()
        self._cluster_wise_termination_watcher_thread = None
        logging.info('Shut down watcher for peer\'s termination signal.')
  def __del__(self):
    self._stop_cluster_wise_termination_watcher_thread()
    self._stop_poll_termination_signal_thread()
  @property
  def total_runs(self):
    return self._run_counter
  def run(self,
          distributed_train_function,
          *args,
          **kwargs):
    """Runs a training function with error and preemption handling.
    This function handles the preemption signal from any peer in the cluster by
    saving the training progress and exiting gracefully. (Specifically, when
    running on Borg, it exits with a special code so that the cluster
    automatically restarts the training after the down worker is back.) It will
    also propagate any program error encountered during execution of
    `distributed_train_function` to all workers so that they can raise the same
    error.
    The `distributed_train_function` argument should be a distributed train
    function (i.e., containing a call to `tf.distribute.Strategy.run`). For
    `tf.distribute.MultiWorkerMirroredStrategy` users, we recommend passing in a
    single-step `distributed_train_function` to
    `WorkerPreemptionHandler.run` so that the checkpoint can be saved in
    time in case a preemption signal or maintenance notice is sent.
    Besides the preemption and error handling part,
    `WorkerPreemptionHandler.run(distributed_train_function, *args,
    **kwargs)` has the same effect and output as
    `distributed_train_function(*args, **kwargs)`. `distributed_train_function`
    can return either some or no result. The following is a shortened example:
    ```python
    @tf.function
    def distributed_train_step(iterator):
      def step_fn(inputs):
        x, y = inputs
        ...
        return loss
      per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
      return strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    for epoch in range(worker_preemption_watcher.total_runs //
                       STEPS_PER_EPOCH, EPOCHS_TO_RUN):
      iterator = iter(multi_worker_dataset)
      total_loss = 0.0
      num_batches = 0
      for step in range(worker_preemption_watcher.total_runs %
                        STEPS_PER_EPOCH, STEPS_PER_EPOCH):
        total_loss += worker_preemption_watcher.run(distributed_train_step)
        num_batches += 1
      train_loss = total_loss / num_batches
      print('Epoch: %d, train_loss: %f.' %(epoch.numpy(), train_loss))
      train_accuracy.reset_states()
    ```
    Args:
      distributed_train_function: A (single-step) distributed training function.
      *args: args for `distributed_train_function`.
      **kwargs: kwargs for `distributed_train_function`.
    Raises:
      Program error encountered by any member in the cluster encounters one
      while executing the `distributed_train_function`, or any error from the
      program error propagation process.
    Returns:
      Result of running the `distributed_train_function`.
    """
    try:
      self._checkpoint_if_preempted()
      result = distributed_train_function(*args, **kwargs)
      self._run_counter += 1
    except errors.OpError as e:
      logging.info('Propagating error to cluster: %r: %s', e, e)
      try:
        context.context().report_error_to_cluster(e.error_code, e.message)
        logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
      raise
    return result
  def _save_checkpoint_and_exit(self):
    logging.info('Starting checkpoint and exit')
    self._checkpointed_runs.assign(self.total_runs)
    start_time = time.monotonic()
    self._write_checkpoint_manager.save()
    if not multi_worker_util.is_chief(
        cluster_spec=self._cluster_resolver.cluster_spec(),
        task_type=self._cluster_resolver.task_type,
        task_id=self._cluster_resolver.task_id):
      gfile.DeleteRecursively(
          os.path.dirname(self._write_checkpoint_manager.directory))
    end_time = time.monotonic()
    logging.info('Checkpoint finished at path %s',
                 self._write_checkpoint_manager.directory)
    logging.info('Checkpoint time: %f', end_time - start_time)
    self._stop_poll_termination_signal_thread()
    self._stop_cluster_wise_termination_watcher_thread()
    sys.exit(self._restart_code)
  def _checkpoint_if_preempted(self):
    """Checkpoint if any worker has received a preemption signal.
    This function handles preemption signal reported by any worker in the
    cluster. The current implementation relies on the fact that all workers in a
    MultiWorkerMirroredStrategy training cluster have a step number difference
    maximum of 1.
    - If the signal comes from the worker itself (i.e., where this failure
    handler sits), the worker will notify all peers to checkpoint after they
    finish CURRENT_STEP+1 steps, where CURRENT_STEP is the step this worker has
    just finished. And the worker will wait for all peers to acknowledge that
    they have received its preemption signal and the final-step number before
    the worker proceeds on training the final step.
    - If the signal comes from another member in the cluster but NO final-step
    info is available, proceed on training, because it will be available after
    finishing the next step.
    - If the signal comes from some other member in the cluster, and final-step
    info is available, if the worker has not finished these steps yet, keep
    training; otherwise, checkpoint and exit with a cluster-recognized restart
    code.
    """
    if self._received_sigterm_and_step.is_set():
      run_count_key = context.context().get_config_key_value(_RUN_COUNT_KEY)
      if run_count_key == str(self._run_counter):
        self._save_checkpoint_and_exit()
    elif self._received_own_sigterm.is_set():
      step_to_save_at = str(self._run_counter + 1)
      try:
        context.context().set_config_key_value(_RUN_COUNT_KEY, step_to_save_at)
        logging.info('Termination caught in main thread on preempted worker')
        logging.info('%s set to %s', _RUN_COUNT_KEY, step_to_save_at)
        n_workers = multi_worker_util.worker_count(
            self._cluster_resolver.cluster_spec(),
            self._cluster_resolver.task_type)
        for i in range(n_workers):
          context.context().get_config_key_value(f'{_ACKNOWLEDGE_KEY}_{i}')
          logging.info('Sigterm acknowledgement from replica %d received', i)
      except errors.AlreadyExistsError:
        logging.info(
            'Member %s has received termination notice. But some other'
            ' worker has received it as well! Leaving'
            ' it to them to decide when to checkpoint. ', self._id_in_cluster)
        return
  def _sigterm_handler_fn(self, signum, frame):
    del signum, frame
    logging.info('Member %s has received termination signal.',
                 self._id_in_cluster)
    self._received_own_sigterm.set()
  def _wait_for_signal(self):
    step_key = context.context().get_config_key_value(_RUN_COUNT_KEY)
    if step_key != _STOP_WATCHING_CLUSTER_VALUE:
      self._received_sigterm_and_step.set()
      ack_key = f'{_ACKNOWLEDGE_KEY}_{self._id_in_cluster}'
      context.context().set_config_key_value(ack_key, '1')
      logging.info(
          'WorkerPreemptionHandler._wait_for_signal: %s set, '
          'preemption awareness acknowledged', ack_key)
