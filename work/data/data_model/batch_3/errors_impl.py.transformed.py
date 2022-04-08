
import traceback
import warnings
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import _pywrap_py_exception_registry
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class InaccessibleTensorError(ValueError):
  pass
@tf_export("errors.OperatorNotAllowedInGraphError", v1=[])
class OperatorNotAllowedInGraphError(TypeError):
  pass
@tf_export("errors.OpError", v1=["errors.OpError", "OpError"])
@deprecation.deprecated_endpoints("OpError")
class OpError(Exception):
  def __init__(self, node_def, op, message, error_code, *args):
    super(OpError, self).__init__()
    self._node_def = node_def
    self._op = op
    self._message = message
    self._error_code = error_code
    if args:
      self._experimental_payloads = args[0]
    else:
      self._experimental_payloads = {}
  def __reduce__(self):
    init_argspec = tf_inspect.getargspec(self.__class__.__init__)
    args = tuple(getattr(self, arg) for arg in init_argspec.args[1:])
    return self.__class__, args
  @property
  def message(self):
    return self._message
  @property
  def op(self):
    return self._op
  @property
  def error_code(self):
    return self._error_code
  @property
  def node_def(self):
    return self._node_def
  @property
  def experimental_payloads(self):
    return self._experimental_payloads
  def __str__(self):
    if self._op is not None:
      output = [
          "%s\n\nOriginal stack trace for %r:\n" % (
              self.message,
              self._op.name,
          )
      ]
      curr_traceback_list = traceback.format_list(self._op.traceback)
      output.extend(curr_traceback_list)
      original_op = self._op._original_op
      while original_op is not None:
        output.append(
            "\n...which was originally created as op %r, defined at:\n" %
            (original_op.name,))
        prev_traceback_list = curr_traceback_list
        curr_traceback_list = traceback.format_list(original_op.traceback)
        is_eliding = False
        elide_count = 0
        last_elided_line = None
        for line, line_in_prev in zip(curr_traceback_list, prev_traceback_list):
          if line == line_in_prev:
            if is_eliding:
              elide_count += 1
              last_elided_line = line
            else:
              output.append(line)
              is_eliding = True
              elide_count = 0
          else:
            if is_eliding:
              if elide_count > 0:
                output.extend([
                    "[elided %d identical lines from previous traceback]\n" %
                    (elide_count - 1,), last_elided_line
                ])
              is_eliding = False
            output.extend(line)
        original_op = original_op._original_op
      return "".join(output)
    else:
      return self.message
OK = error_codes_pb2.OK
tf_export("errors.OK").export_constant(__name__, "OK")
CANCELLED = error_codes_pb2.CANCELLED
tf_export("errors.CANCELLED").export_constant(__name__, "CANCELLED")
UNKNOWN = error_codes_pb2.UNKNOWN
tf_export("errors.UNKNOWN").export_constant(__name__, "UNKNOWN")
INVALID_ARGUMENT = error_codes_pb2.INVALID_ARGUMENT
tf_export("errors.INVALID_ARGUMENT").export_constant(__name__,
                                                     "INVALID_ARGUMENT")
DEADLINE_EXCEEDED = error_codes_pb2.DEADLINE_EXCEEDED
tf_export("errors.DEADLINE_EXCEEDED").export_constant(__name__,
                                                      "DEADLINE_EXCEEDED")
NOT_FOUND = error_codes_pb2.NOT_FOUND
tf_export("errors.NOT_FOUND").export_constant(__name__, "NOT_FOUND")
ALREADY_EXISTS = error_codes_pb2.ALREADY_EXISTS
tf_export("errors.ALREADY_EXISTS").export_constant(__name__, "ALREADY_EXISTS")
PERMISSION_DENIED = error_codes_pb2.PERMISSION_DENIED
tf_export("errors.PERMISSION_DENIED").export_constant(__name__,
                                                      "PERMISSION_DENIED")
UNAUTHENTICATED = error_codes_pb2.UNAUTHENTICATED
tf_export("errors.UNAUTHENTICATED").export_constant(__name__, "UNAUTHENTICATED")
RESOURCE_EXHAUSTED = error_codes_pb2.RESOURCE_EXHAUSTED
tf_export("errors.RESOURCE_EXHAUSTED").export_constant(__name__,
                                                       "RESOURCE_EXHAUSTED")
FAILED_PRECONDITION = error_codes_pb2.FAILED_PRECONDITION
tf_export("errors.FAILED_PRECONDITION").export_constant(__name__,
                                                        "FAILED_PRECONDITION")
ABORTED = error_codes_pb2.ABORTED
tf_export("errors.ABORTED").export_constant(__name__, "ABORTED")
OUT_OF_RANGE = error_codes_pb2.OUT_OF_RANGE
tf_export("errors.OUT_OF_RANGE").export_constant(__name__, "OUT_OF_RANGE")
UNIMPLEMENTED = error_codes_pb2.UNIMPLEMENTED
tf_export("errors.UNIMPLEMENTED").export_constant(__name__, "UNIMPLEMENTED")
INTERNAL = error_codes_pb2.INTERNAL
tf_export("errors.INTERNAL").export_constant(__name__, "INTERNAL")
UNAVAILABLE = error_codes_pb2.UNAVAILABLE
tf_export("errors.UNAVAILABLE").export_constant(__name__, "UNAVAILABLE")
DATA_LOSS = error_codes_pb2.DATA_LOSS
tf_export("errors.DATA_LOSS").export_constant(__name__, "DATA_LOSS")
@tf_export("errors.CancelledError")
class CancelledError(OpError):
  """Raised when an operation or step is cancelled.
  For example, a long-running operation (e.g.
  `tf.QueueBase.enqueue` may be
  cancelled by running another operation (e.g.
  `tf.QueueBase.close`,
  or by `tf.Session.close`.
  A step that is running such a long-running operation will fail by raising
  `CancelledError`.
  @@__init__
  """
  def __init__(self, node_def, op, message, *args):
    super(CancelledError, self).__init__(node_def, op, message, CANCELLED,
                                         *args)
@tf_export("errors.UnknownError")
class UnknownError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(UnknownError, self).__init__(node_def, op, message, UNKNOWN, *args)
@tf_export("errors.InvalidArgumentError")
class InvalidArgumentError(OpError):
  """Raised when an operation receives an invalid argument.
  This error is typically raised when an op receives mismatched arguments.
  Example:
  >>> tf.reshape([1, 2, 3], (2,))
  Traceback (most recent call last):
     ...
  InvalidArgumentError: ...
  @@__init__
  """
  def __init__(self, node_def, op, message, *args):
    super(InvalidArgumentError, self).__init__(node_def, op, message,
                                               INVALID_ARGUMENT, *args)
@tf_export("errors.DeadlineExceededError")
class DeadlineExceededError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(DeadlineExceededError, self).__init__(node_def, op, message,
                                                DEADLINE_EXCEEDED, *args)
@tf_export("errors.NotFoundError")
class NotFoundError(OpError):
  """Raised when a requested entity (e.g., a file or directory) was not found.
  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `NotFoundError` if it receives the name of a file that
  does not exist.
  @@__init__
  """
  def __init__(self, node_def, op, message, *args):
    super(NotFoundError, self).__init__(node_def, op, message, NOT_FOUND, *args)
@tf_export("errors.AlreadyExistsError")
class AlreadyExistsError(OpError):
  """Raised when an entity that we attempted to create already exists.
  For example, running an operation that saves a file
  (e.g. `tf.train.Saver.save`)
  could potentially raise this exception if an explicit filename for an
  existing file was passed.
  @@__init__
  """
  def __init__(self, node_def, op, message, *args):
    super(AlreadyExistsError, self).__init__(node_def, op, message,
                                             ALREADY_EXISTS, *args)
@tf_export("errors.PermissionDeniedError")
class PermissionDeniedError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(PermissionDeniedError, self).__init__(node_def, op, message,
                                                PERMISSION_DENIED, *args)
@tf_export("errors.UnauthenticatedError")
class UnauthenticatedError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(UnauthenticatedError, self).__init__(node_def, op, message,
                                               UNAUTHENTICATED, *args)
@tf_export("errors.ResourceExhaustedError")
class ResourceExhaustedError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(ResourceExhaustedError, self).__init__(node_def, op, message,
                                                 RESOURCE_EXHAUSTED, *args)
@tf_export("errors.FailedPreconditionError")
class FailedPreconditionError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(FailedPreconditionError, self).__init__(node_def, op, message,
                                                  FAILED_PRECONDITION, *args)
@tf_export("errors.AbortedError")
class AbortedError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(AbortedError, self).__init__(node_def, op, message, ABORTED, *args)
@tf_export("errors.OutOfRangeError")
class OutOfRangeError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(OutOfRangeError, self).__init__(node_def, op, message, OUT_OF_RANGE,
                                          *args)
@tf_export("errors.UnimplementedError")
class UnimplementedError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(UnimplementedError, self).__init__(node_def, op, message,
                                             UNIMPLEMENTED, *args)
@tf_export("errors.InternalError")
class InternalError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(InternalError, self).__init__(node_def, op, message, INTERNAL, *args)
@tf_export("errors.UnavailableError")
class UnavailableError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(UnavailableError, self).__init__(node_def, op, message, UNAVAILABLE,
                                           *args)
@tf_export("errors.DataLossError")
class DataLossError(OpError):
  def __init__(self, node_def, op, message, *args):
    super(DataLossError, self).__init__(node_def, op, message, DATA_LOSS, *args)
_CODE_TO_EXCEPTION_CLASS = {
    CANCELLED: CancelledError,
    UNKNOWN: UnknownError,
    INVALID_ARGUMENT: InvalidArgumentError,
    DEADLINE_EXCEEDED: DeadlineExceededError,
    NOT_FOUND: NotFoundError,
    ALREADY_EXISTS: AlreadyExistsError,
    PERMISSION_DENIED: PermissionDeniedError,
    UNAUTHENTICATED: UnauthenticatedError,
    RESOURCE_EXHAUSTED: ResourceExhaustedError,
    FAILED_PRECONDITION: FailedPreconditionError,
    ABORTED: AbortedError,
    OUT_OF_RANGE: OutOfRangeError,
    UNIMPLEMENTED: UnimplementedError,
    INTERNAL: InternalError,
    UNAVAILABLE: UnavailableError,
    DATA_LOSS: DataLossError,
}
_pywrap_py_exception_registry.PyExceptionRegistry_Init(_CODE_TO_EXCEPTION_CLASS)
_EXCEPTION_CLASS_TO_CODE = {
    class_: code for code, class_ in _CODE_TO_EXCEPTION_CLASS.items()
}
@tf_export(v1=["errors.exception_type_from_error_code"])
def exception_type_from_error_code(error_code):
  return _CODE_TO_EXCEPTION_CLASS[error_code]
@tf_export(v1=["errors.error_code_from_exception_type"])
def error_code_from_exception_type(cls):
  try:
    return _EXCEPTION_CLASS_TO_CODE[cls]
  except KeyError:
    warnings.warn("Unknown class exception")
    return UnknownError(None, None, "Unknown class exception", None)
def _make_specific_exception(node_def, op, message, error_code):
  try:
    exc_type = exception_type_from_error_code(error_code)
    return exc_type(node_def, op, message)
  except KeyError:
    warnings.warn("Unknown error code: %d" % error_code)
    return UnknownError(node_def, op, message, error_code)
class raise_exception_on_not_ok_status(object):
  def __enter__(self):
    self.status = c_api_util.ScopedTFStatus()
    return self.status.status
  def __exit__(self, type_arg, value_arg, traceback_arg):
    try:
      if c_api.TF_GetCode(self.status.status) != 0:
        raise _make_specific_exception(
            None, None, compat.as_text(c_api.TF_Message(self.status.status)),
            c_api.TF_GetCode(self.status.status))
    finally:
      del self.status
