
import functools
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def _is_bound_method(fn):
  _, fn = tf_decorator.unwrap(fn)
  return tf_inspect.ismethod(fn) and (fn.__self__ is not None)
def _is_callable_object(obj):
  return hasattr(obj, '__call__') and tf_inspect.ismethod(obj.__call__)
def fn_args(fn):
  """Get argument names for function-like object.
  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).
  Returns:
    `tuple` of string argument names.
  Raises:
    ValueError: if partial function has positionally bound arguments
  """
  if isinstance(fn, functools.partial):
    args = fn_args(fn.func)
    args = [a for a in args[len(fn.args):] if a not in (fn.keywords or [])]
  else:
    if _is_callable_object(fn):
      fn = fn.__call__
    args = tf_inspect.getfullargspec(fn).args
    if _is_bound_method(fn) and args:
      args.pop(0)
  return tuple(args)
def has_kwargs(fn):
  """Returns whether the passed callable has **kwargs in its signature.
  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).
  Returns:
    `bool`: if `fn` has **kwargs in its signature.
  Raises:
     `TypeError`: If fn is not a Function, or function-like object.
  """
  if isinstance(fn, functools.partial):
    fn = fn.func
  elif _is_callable_object(fn):
    fn = fn.__call__
  elif not callable(fn):
    raise TypeError(
        'Argument `fn` should be a callable. '
        f'Received: fn={fn} (of type {type(fn)})')
  return tf_inspect.getfullargspec(fn).varkw is not None
def get_func_name(func):
  _, func = tf_decorator.unwrap(func)
  if callable(func):
    if tf_inspect.isfunction(func):
      return func.__name__
    elif tf_inspect.ismethod(func):
      return '%s.%s' % (six.get_method_self(func).__class__.__name__,
                        six.get_method_function(func).__name__)
      return str(type(func))
  else:
    raise ValueError(
        'Argument `func` must be a callable. '
        f'Received func={func} (of type {type(func)})')
def get_func_code(func):
  _, func = tf_decorator.unwrap(func)
  if callable(func):
    if tf_inspect.isfunction(func) or tf_inspect.ismethod(func):
      return six.get_function_code(func)
    try:
      return six.get_function_code(func.__call__)
    except AttributeError:
      return None
  else:
    raise ValueError(
        'Argument `func` must be a callable. '
        f'Received func={func} (of type {type(func)})')
_rewriter_config_optimizer_disabled = None
def get_disabled_rewriter_config():
  global _rewriter_config_optimizer_disabled
  if _rewriter_config_optimizer_disabled is None:
    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.disable_meta_optimizer = True
    _rewriter_config_optimizer_disabled = config.SerializeToString()
  return _rewriter_config_optimizer_disabled
