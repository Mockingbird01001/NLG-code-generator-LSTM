
import functools
import inspect
import sys
import unittest
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager import function
from tensorflow.python.util import tf_inspect
_ALLOWLIST_CACHE = cache.UnboundInstanceCache()
def _is_of_known_loaded_module(f, module_name):
  mod = sys.modules.get(module_name, None)
  if mod is None:
    return False
  if any(v is not None for v in mod.__dict__.values() if f is v):
    return True
  return False
def _is_known_loaded_type(f, module_name, entity_name):
  if (module_name not in sys.modules or
      not hasattr(sys.modules[module_name], entity_name)):
    return False
  type_entity = getattr(sys.modules[module_name], entity_name)
  if isinstance(f, type_entity):
    return True
  if inspect.ismethod(f):
    if isinstance(f.__func__, type_entity):
      return True
  return False
def is_unsupported(o):
  if (_is_known_loaded_type(o, 'wrapt', 'FunctionWrapper') or
      _is_known_loaded_type(o, 'wrapt', 'BoundFunctionWrapper')):
    logging.warning(
        '{} appears to be decorated by wrapt, which is not yet supported'
        ' by AutoGraph. The function will run as-is.'
        ' You may still apply AutoGraph before the wrapt decorator.'.format(o))
    logging.log(2, 'Permanently allowed: %s: wrapt decorated', o)
    return True
  if _is_known_loaded_type(o, 'functools', '_lru_cache_wrapper'):
    logging.log(2, 'Permanently allowed: %s: lru_cache', o)
    return True
  if inspect_utils.isconstructor(o):
    logging.log(2, 'Permanently allowed: %s: constructor', o)
    return True
  if any(
      _is_of_known_loaded_module(o, m)
      for m in ('collections', 'pdb', 'copy', 'inspect', 're')):
    logging.log(2, 'Permanently allowed: %s: part of builtin module', o)
    return True
  if (hasattr(o, '__module__') and
      hasattr(o.__module__, '_IS_TENSORFLOW_PLUGIN')):
    logging.log(2, 'Permanently allowed: %s: TensorFlow plugin', o)
    return True
  return False
def is_allowlisted(
    o, check_call_override=True, allow_namedtuple_subclass=False):
  if isinstance(o, functools.partial):
    m = functools
  else:
    m = tf_inspect.getmodule(o)
  if hasattr(m, '__name__'):
    for rule in config.CONVERSION_RULES:
      action = rule.get_action(m)
      if action == config.Action.CONVERT:
        logging.log(2, 'Not allowed: %s: %s', o, rule)
        return False
      elif action == config.Action.DO_NOT_CONVERT:
        logging.log(2, 'Allowlisted: %s: %s', o, rule)
        return True
  if hasattr(o, '__code__') and tf_inspect.isgeneratorfunction(o):
    logging.log(2, 'Allowlisted: %s: generator functions are not converted', o)
    return True
  if (check_call_override and not tf_inspect.isclass(o) and
      hasattr(o, '__call__')):
      logging.log(2, 'Allowlisted: %s: object __call__ allowed', o)
      return True
  owner_class = None
  if tf_inspect.ismethod(o):
    owner_class = inspect_utils.getmethodclass(o)
    if owner_class is function.TfMethodTarget:
      owner_class = o.__self__.target_class
    if owner_class is not None:
      if issubclass(owner_class, unittest.TestCase):
        logging.log(2, 'Allowlisted: %s: method of TestCase subclass', o)
        return True
      owner_class = inspect_utils.getdefiningclass(o, owner_class)
      if is_allowlisted(
          owner_class,
          check_call_override=False,
          allow_namedtuple_subclass=True):
        logging.log(2, 'Allowlisted: %s: owner is allowed %s', o,
                    owner_class)
        return True
  if inspect_utils.isnamedtuple(o):
    if allow_namedtuple_subclass:
      if not any(inspect_utils.isnamedtuple(base) for base in o.__bases__):
        logging.log(2, 'Allowlisted: %s: named tuple', o)
        return True
    else:
      logging.log(2, 'Allowlisted: %s: named tuple or subclass', o)
      return True
  logging.log(2, 'Not allowed: %s: default rule', o)
  return False
def is_in_allowlist_cache(entity, options):
  try:
    return _ALLOWLIST_CACHE.has(entity, options)
  except TypeError:
    return False
def cache_allowlisted(entity, options):
  try:
    _ALLOWLIST_CACHE[entity][options] = True
  except TypeError:
    pass
