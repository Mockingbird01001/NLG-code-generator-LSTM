
import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
ArgSpec = _inspect.ArgSpec
if hasattr(_inspect, 'FullArgSpec'):
else:
  FullArgSpec = collections.namedtuple('FullArgSpec', [
      'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
      'annotations'
  ])
def _convert_maybe_argspec_to_fullargspec(argspec):
  if isinstance(argspec, FullArgSpec):
    return argspec
  return FullArgSpec(
      args=argspec.args,
      varargs=argspec.varargs,
      varkw=argspec.keywords,
      defaults=argspec.defaults,
      kwonlyargs=[],
      kwonlydefaults=None,
      annotations={})
if hasattr(_inspect, 'getfullargspec'):
  def _getargspec(target):
    fullargspecs = getfullargspec(target)
    argspecs = ArgSpec(
        args=fullargspecs.args,
        varargs=fullargspecs.varargs,
        keywords=fullargspecs.varkw,
        defaults=fullargspecs.defaults)
    return argspecs
else:
  _getargspec = _inspect.getargspec
  def _getfullargspec(target):
    return _convert_maybe_argspec_to_fullargspec(getargspec(target))
def currentframe():
  return _inspect.stack()[1][0]
def getargspec(obj):
  if isinstance(obj, functools.partial):
    return _get_argspec_for_partial(obj)
  decorators, target = tf_decorator.unwrap(obj)
  spec = next((d.decorator_argspec
               for d in decorators
               if d.decorator_argspec is not None), None)
  if spec:
    return spec
  try:
    return _getargspec(target)
  except TypeError:
    pass
  if isinstance(target, type):
    try:
      return _getargspec(target.__init__)
    except TypeError:
      pass
    try:
      return _getargspec(target.__new__)
    except TypeError:
      pass
  return _getargspec(type(target).__call__)
def _get_argspec_for_partial(obj):
  n_prune_args = len(obj.args)
  partial_keywords = obj.keywords or {}
  args, varargs, keywords, defaults = getargspec(obj.func)
  args = args[n_prune_args:]
  no_default = object()
  all_defaults = [no_default] * len(args)
  if defaults:
    all_defaults[-len(defaults):] = defaults
  for kw, default in partial_keywords.items():
    if kw in args:
      idx = args.index(kw)
      all_defaults[idx] = default
    elif not keywords:
      raise ValueError('Function does not have **kwargs parameter, but '
                       'contains an unknown partial keyword.')
  first_default = next(
      (idx for idx, x in enumerate(all_defaults) if x is not no_default), None)
  if first_default is None:
    return ArgSpec(args, varargs, keywords, None)
  invalid_default_values = [
      args[i] for i, j in enumerate(all_defaults)
      if j is no_default and i > first_default
  ]
  if invalid_default_values:
    raise ValueError('Some arguments %s do not have default value, but they '
                     'are positioned after those with default values. This can '
                     'not be expressed with ArgSpec.' % invalid_default_values)
  return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))
def getfullargspec(obj):
  """TFDecorator-aware replacement for `inspect.getfullargspec`.
  This wrapper emulates `inspect.getfullargspec` in[^)]* Python2.
  Args:
    obj: A callable, possibly decorated.
  Returns:
    The `FullArgSpec` that describes the signature of
    the outermost decorator that changes the callable's signature. If the
    callable is not decorated, `inspect.getfullargspec()` will be called
    directly on the callable.
  """
  decorators, target = tf_decorator.unwrap(obj)
  for d in decorators:
    if d.decorator_argspec is not None:
      return _convert_maybe_argspec_to_fullargspec(d.decorator_argspec)
  return _getfullargspec(target)
def getcallargs(*func_and_positional, **named):
  """TFDecorator-aware replacement for inspect.getcallargs.
  Args:
    *func_and_positional: A callable, possibly decorated, followed by any
      positional arguments that would be passed to `func`.
    **named: The named argument dictionary that would be passed to `func`.
  Returns:
    A dictionary mapping `func`'s named arguments to the values they would
    receive if `func(*positional, **named)` were called.
  `getcallargs` will use the argspec from the outermost decorator that provides
  it. If no attached decorators modify argspec, the final unwrapped target's
  argspec will be used.
  """
  func = func_and_positional[0]
  positional = func_and_positional[1:]
  argspec = getfullargspec(func)
  call_args = named.copy()
  this = getattr(func, 'im_self', None) or getattr(func, '__self__', None)
  if ismethod(func) and this:
    positional = (this,) + positional
  remaining_positionals = [arg for arg in argspec.args if arg not in call_args]
  call_args.update(dict(zip(remaining_positionals, positional)))
  default_count = 0 if not argspec.defaults else len(argspec.defaults)
  if default_count:
    for arg, value in zip(argspec.args[-default_count:], argspec.defaults):
      if arg not in call_args:
        call_args[arg] = value
  if argspec.kwonlydefaults is not None:
    for k, v in argspec.kwonlydefaults.items():
      if k not in call_args:
        call_args[k] = v
  return call_args
def getframeinfo(*args, **kwargs):
  return _inspect.getframeinfo(*args, **kwargs)
  return _inspect.getdoc(object)
  unwrapped_object = tf_decorator.unwrap(object)[1]
  if (hasattr(unwrapped_object, 'f_globals') and
      '__file__' in unwrapped_object.f_globals):
    return unwrapped_object.f_globals['__file__']
  return _inspect.getfile(unwrapped_object)
  return _inspect.getmembers(object, predicate)
  return _inspect.getmodule(object)
def getmro(cls):
  return _inspect.getmro(cls)
  return _inspect.getsource(tf_decorator.unwrap(object)[1])
  return _inspect.getsourcefile(tf_decorator.unwrap(object)[1])
  return _inspect.getsourcelines(tf_decorator.unwrap(object)[1])
  return _inspect.isbuiltin(tf_decorator.unwrap(object)[1])
  return _inspect.isclass(tf_decorator.unwrap(object)[1])
  return _inspect.isfunction(tf_decorator.unwrap(object)[1])
  return _inspect.isframe(tf_decorator.unwrap(object)[1])
  return _inspect.isgenerator(tf_decorator.unwrap(object)[1])
  return _inspect.isgeneratorfunction(tf_decorator.unwrap(object)[1])
  return _inspect.ismethod(tf_decorator.unwrap(object)[1])
  return _inspect.ismodule(tf_decorator.unwrap(object)[1])
  return _inspect.isroutine(tf_decorator.unwrap(object)[1])
def stack(context=1):
  return _inspect.stack(context)[1:]
