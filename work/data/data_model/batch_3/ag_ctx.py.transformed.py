
import enum
import inspect
import threading
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.util.tf_export import tf_export
stacks = threading.local()
def _control_ctx():
  if not hasattr(stacks, 'control_status'):
    stacks.control_status = [_default_control_status_ctx()]
  return stacks.control_status
@tf_export('__internal__.autograph.control_status_ctx', v1=[])
def control_status_ctx():
  """Returns the current control context for autograph.
  This method is useful when calling `tf.__internal__.autograph.tf_convert`,
  The context will be used by tf_convert to determine whether it should convert
  the input function. See the sample usage like below:
  ```
  def foo(func):
    return tf.__internal__.autograph.tf_convert(
       input_fn, ctx=tf.__internal__.autograph.control_status_ctx())()
  ```
  Returns:
    The current control context of autograph.
  """
  ret = _control_ctx()[-1]
  return ret
class Status(enum.Enum):
  UNSPECIFIED = 0
  ENABLED = 1
  DISABLED = 2
class ControlStatusCtx(object):
  def __init__(self, status, options=None):
    self.status = status
    self.options = options
  def __enter__(self):
    _control_ctx().append(self)
    return self
  def __repr__(self):
    return '{}[status={}, options={}]'.format(
        self.__class__.__name__, self.status, self.options)
  def __exit__(self, unused_type, unused_value, unused_traceback):
    assert _control_ctx()[-1] is self
    _control_ctx().pop()
class NullCtx(object):
  def __enter__(self):
    pass
  def __exit__(self, unused_type, unused_value, unused_traceback):
    pass
def _default_control_status_ctx():
  return ControlStatusCtx(status=Status.UNSPECIFIED)
INSPECT_SOURCE_SUPPORTED = True
try:
  inspect.getsource(ag_logging.log)
except OSError:
  INSPECT_SOURCE_SUPPORTED = False
  ag_logging.warning(
      'AutoGraph is not available in this environment: functions lack code'
      ' information. This is typical of some environments like the interactive'
      ' Python shell. See'
      ' for more information.')
