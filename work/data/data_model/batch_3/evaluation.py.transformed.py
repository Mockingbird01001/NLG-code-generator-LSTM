
import math
import time
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
def _get_or_create_eval_step():
  graph = ops.get_default_graph()
  eval_steps = graph.get_collection(ops.GraphKeys.EVAL_STEP)
  if len(eval_steps) == 1:
    return eval_steps[0]
  elif len(eval_steps) > 1:
    raise ValueError('Multiple tensors added to tf.GraphKeys.EVAL_STEP')
  else:
    counter = variable_scope.get_variable(
        'eval_step',
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.EVAL_STEP])
    return counter
def _get_latest_eval_step_value(update_ops):
  if isinstance(update_ops, dict):
    update_ops = list(update_ops.values())
  with ops.control_dependencies(update_ops):
    return array_ops.identity(_get_or_create_eval_step().read_value())
class _MultiStepStopAfterNEvalsHook(session_run_hook.SessionRunHook):
  def __init__(self, num_evals, steps_per_run=1):
    self._num_evals = num_evals
    self._evals_completed = None
    self._steps_per_run_initial_value = steps_per_run
  def _set_evals_completed_tensor(self, updated_eval_step):
    self._evals_completed = updated_eval_step
  def begin(self):
    self._steps_per_run_variable = \
        basic_session_run_hooks.get_or_create_steps_per_run_variable()
  def after_create_session(self, session, coord):
    if self._num_evals is None:
      steps = self._steps_per_run_initial_value
    else:
      steps = min(self._steps_per_run_initial_value, self._num_evals)
    self._steps_per_run_variable.load(steps, session=session)
  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(
        {'evals_completed': self._evals_completed})
  def after_run(self, run_context, run_values):
    evals_completed = run_values.results['evals_completed']
    if self._num_evals is None:
      steps = self._steps_per_run_initial_value
    else:
      steps = min(self._num_evals - evals_completed,
                  self._steps_per_run_initial_value)
    self._steps_per_run_variable.load(steps, session=run_context.session)
    if self._num_evals is None:
      logging.info('Evaluation [%d]', evals_completed)
    else:
      logging.info('Evaluation [%d/%d]', evals_completed, self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()
class _StopAfterNEvalsHook(session_run_hook.SessionRunHook):
  def __init__(self, num_evals, log_progress=True):
    self._num_evals = num_evals
    self._evals_completed = None
    self._log_progress = log_progress
    self._log_frequency = (1 if (num_evals is None or num_evals < 20) else
                           math.floor(num_evals / 10.))
  def _set_evals_completed_tensor(self, updated_eval_step):
    self._evals_completed = updated_eval_step
  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(
        {'evals_completed': self._evals_completed})
  def after_run(self, run_context, run_values):
    evals_completed = run_values.results['evals_completed']
    if self._log_progress:
      if self._num_evals is None:
        logging.info('Evaluation [%d]', evals_completed)
      else:
        if ((evals_completed % self._log_frequency) == 0 or
            (self._num_evals == evals_completed)):
          logging.info('Evaluation [%d/%d]', evals_completed, self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()
def _evaluate_once(checkpoint_path,
                   master='',
                   scaffold=None,
                   eval_ops=None,
                   feed_dict=None,
                   final_ops=None,
                   final_ops_feed_dict=None,
                   hooks=None,
                   config=None):
  eval_step = _get_or_create_eval_step()
  hooks = list(hooks or [])
  if eval_ops is not None:
    if any(isinstance(h, _MultiStepStopAfterNEvalsHook) for h in hooks):
      steps_per_run_variable = \
          basic_session_run_hooks.get_or_create_steps_per_run_variable()
      update_eval_step = state_ops.assign_add(
          eval_step,
          math_ops.cast(steps_per_run_variable, dtype=eval_step.dtype),
          use_locking=True)
    else:
      update_eval_step = state_ops.assign_add(eval_step, 1, use_locking=True)
    if isinstance(eval_ops, dict):
      eval_ops['update_eval_step'] = update_eval_step
    elif isinstance(eval_ops, (tuple, list)):
      eval_ops = list(eval_ops) + [update_eval_step]
    else:
      eval_ops = [eval_ops, update_eval_step]
    eval_step_value = _get_latest_eval_step_value(eval_ops)
    for h in hooks:
      if isinstance(h, (_StopAfterNEvalsHook, _MultiStepStopAfterNEvalsHook)):
  logging.info('Starting evaluation at ' +
               time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()))
  start = time.time()
  session_creator = monitored_session.ChiefSessionCreator(
      scaffold=scaffold,
      checkpoint_filename_with_path=checkpoint_path,
      master=master,
      config=config)
  final_ops_hook = basic_session_run_hooks.FinalOpsHook(final_ops,
                                                        final_ops_feed_dict)
  hooks.append(final_ops_hook)
  with monitored_session.MonitoredSession(
      session_creator=session_creator, hooks=hooks) as session:
    if eval_ops is not None:
      while not session.should_stop():
        session.run(eval_ops, feed_dict)
  logging.info('Inference Time : {:0.5f}s'.format(time.time() - start))
  logging.info('Finished evaluation at ' +
               time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()))
  return final_ops_hook.final_ops_values
