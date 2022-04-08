
import logging as _logging
import os as _os
import sys as _sys
import time as _time
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading
import six
from tensorflow.python.util.tf_export import tf_export
_logger = None
_logger_lock = threading.Lock()
def _get_caller(offset=3):
  f = _sys._getframe(offset)
  our_file = f.f_code.co_filename
  f = f.f_back
  while f:
    code = f.f_code
    if code.co_filename != our_file:
      return code, f
    f = f.f_back
  return None, None
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 8:
    code, frame = _get_caller(4)
    sinfo = None
    if stack_info:
      sinfo = '\n'.join(_traceback.format_stack())
    if code:
      return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
    else:
      return '(unknown file)', 0, '(unknown function)', sinfo
elif _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:
    code, frame = _get_caller(4)
    sinfo = None
    if stack_info:
      sinfo = '\n'.join(_traceback.format_stack())
    if code:
      return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
    else:
      return '(unknown file)', 0, '(unknown function)', sinfo
else:
    code, frame = _get_caller(4)
    if code:
      return (code.co_filename, frame.f_lineno, code.co_name)
    else:
      return '(unknown file)', 0, '(unknown function)'
@tf_export('get_logger')
def get_logger():
  global _logger
  if _logger:
    return _logger
  _logger_lock.acquire()
  try:
    if _logger:
      return _logger
    logger = _logging.getLogger('tensorflow')
    logger.findCaller = _logger_find_caller
    if not _logging.getLogger().handlers:
      _interactive = False
      try:
        if _sys.ps1: _interactive = True
      except AttributeError:
        _interactive = _sys.flags.interactive
      if _interactive:
        logger.setLevel(INFO)
        _logging_target = _sys.stdout
      else:
        _logging_target = _sys.stderr
      _handler = _logging.StreamHandler(_logging_target)
      _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
      logger.addHandler(_handler)
    _logger = logger
    return _logger
  finally:
    _logger_lock.release()
@tf_export(v1=['logging.log'])
def log(level, msg, *args, **kwargs):
  get_logger().log(level, msg, *args, **kwargs)
@tf_export(v1=['logging.debug'])
def debug(msg, *args, **kwargs):
  get_logger().debug(msg, *args, **kwargs)
@tf_export(v1=['logging.error'])
def error(msg, *args, **kwargs):
  get_logger().error(msg, *args, **kwargs)
@tf_export(v1=['logging.fatal'])
def fatal(msg, *args, **kwargs):
  get_logger().fatal(msg, *args, **kwargs)
@tf_export(v1=['logging.info'])
def info(msg, *args, **kwargs):
  get_logger().info(msg, *args, **kwargs)
@tf_export(v1=['logging.warn'])
def warn(msg, *args, **kwargs):
  get_logger().warning(msg, *args, **kwargs)
@tf_export(v1=['logging.warning'])
def warning(msg, *args, **kwargs):
  get_logger().warning(msg, *args, **kwargs)
_level_names = {
    FATAL: 'FATAL',
    ERROR: 'ERROR',
    WARN: 'WARN',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
}
_THREAD_ID_MASK = 2 * _sys.maxsize + 1
_log_counter_per_token = {}
@tf_export(v1=['logging.TaskLevelStatusMessage'])
def TaskLevelStatusMessage(msg):
  error(msg)
@tf_export(v1=['logging.flush'])
def flush():
  raise NotImplementedError()
@tf_export(v1=['logging.vlog'])
def vlog(level, msg, *args, **kwargs):
  get_logger().log(level, msg, *args, **kwargs)
def _GetNextLogCountPerToken(token):
  """Wrapper for _log_counter_per_token.
  Args:
    token: The token for which to look up the count.
  Returns:
    The number of times this function has been called with
    *token* as an argument (starting at 0)
  """
  _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
  return _log_counter_per_token[token]
@tf_export(v1=['logging.log_every_n'])
def log_every_n(level, msg, n, *args):
  """Log 'msg % args' at level 'level' once per 'n' times.
  Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
  Not threadsafe.
  Args:
    level: The level at which to log.
    msg: The message to be logged.
    n: The number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
  """
  count = _GetNextLogCountPerToken(_GetFileAndLine())
  log_if(level, msg, not (count % n), *args)
@tf_export(v1=['logging.log_first_n'])
  count = _GetNextLogCountPerToken(_GetFileAndLine())
  log_if(level, msg, count < n, *args)
@tf_export(v1=['logging.log_if'])
def log_if(level, msg, condition, *args):
  if condition:
    vlog(level, msg, *args)
def _GetFileAndLine():
  code, f = _get_caller()
  if not code:
    return ('<unknown>', 0)
  return (code.co_filename, f.f_lineno)
def google2_log_prefix(level, timestamp=None, file_and_line=None):
  global _level_names
  now = timestamp or _time.time()
  now_tuple = _time.localtime(now)
  now_microsecond = int(1e6 * (now % 1.0))
  (filename, line) = file_and_line or _GetFileAndLine()
  basename = _os.path.basename(filename)
  severity = 'I'
  if level in _level_names:
    severity = _level_names[level][0]
  s = '%c%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] ' % (
      severity,
      now_microsecond,
      _get_thread_id(),
      basename,
      line)
  return s
@tf_export(v1=['logging.get_verbosity'])
def get_verbosity():
  return get_logger().getEffectiveLevel()
@tf_export(v1=['logging.set_verbosity'])
def set_verbosity(v):
  get_logger().setLevel(v)
def _get_thread_id():
  thread_id = six.moves._thread.get_ident()
  return thread_id & _THREAD_ID_MASK
_log_prefix = google2_log_prefix
tf_export(v1=['logging.DEBUG']).export_constant(__name__, 'DEBUG')
tf_export(v1=['logging.ERROR']).export_constant(__name__, 'ERROR')
tf_export(v1=['logging.FATAL']).export_constant(__name__, 'FATAL')
tf_export(v1=['logging.INFO']).export_constant(__name__, 'INFO')
tf_export(v1=['logging.WARN']).export_constant(__name__, 'WARN')
