
"""Decorator and context manager for saving and restoring flag values.
There are many ways to save and restore.  Always use the most convenient method
for a given use case.
Here are examples of each method.  They all call do_stuff() while FLAGS.someflag
is temporarily set to 'foo'.
  from absl.testing import flagsaver
  @flagsaver.flagsaver(someflag='foo')
  def some_func():
    do_stuff()
  @flagsaver.flagsaver((module.FOO_FLAG, 'foo'), (other_mod.BAR_FLAG, 23))
  def some_func():
    do_stuff()
  @flagsaver.flagsaver
  def some_func():
    FLAGS.someflag = 'foo'
    do_stuff()
  with flagsaver.flagsaver(someflag='foo'):
    do_stuff()
  saved_flag_values = flagsaver.save_flag_values()
  try:
    FLAGS.someflag = 'foo'
    do_stuff()
  finally:
    flagsaver.restore_flag_values(saved_flag_values)
We save and restore a shallow copy of each Flag object's __dict__ attribute.
This preserves all attributes of the flag, such as whether or not it was
overridden from its default value.
WARNING: Currently a flag that is saved and then deleted cannot be restored.  An
exception will be raised.  However if you *add* a flag after saving flag values,
and then restore flag values, the added flag will be deleted with no errors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import inspect
from absl import flags
FLAGS = flags.FLAGS
def flagsaver(*args, **kwargs):
  if not args:
    return _FlagOverrider(**kwargs)
  if len(args) == 1 and callable(args[0]):
    if kwargs:
      raise ValueError(
          "It's invalid to specify both positional and keyword parameters.")
    func = args[0]
    if inspect.isclass(func):
      raise TypeError('@flagsaver.flagsaver cannot be applied to a class.')
    return _wrap(func, {})
  for arg in args:
    if not isinstance(arg, tuple) or len(arg) != 2:
      raise ValueError('Expected (FlagHolder, value) pair, found %r' % (arg,))
    holder, value = arg
    if not isinstance(holder, flags.FlagHolder):
      raise ValueError('Expected (FlagHolder, value) pair, found %r' % (arg,))
    if holder.name in kwargs:
      raise ValueError('Cannot set --%s multiple times' % holder.name)
    kwargs[holder.name] = value
  return _FlagOverrider(**kwargs)
def save_flag_values(flag_values=FLAGS):
  return {name: _copy_flag_dict(flag_values[name]) for name in flag_values}
def restore_flag_values(saved_flag_values, flag_values=FLAGS):
  new_flag_names = list(flag_values)
  for name in new_flag_names:
    saved = saved_flag_values.get(name)
    if saved is None:
      delattr(flag_values, name)
    else:
      if flag_values[name].value != saved['_value']:
        flag_values[name].value = saved['_value']
      flag_values[name].__dict__ = saved
def _wrap(func, overrides):
  @functools.wraps(func)
  def _flagsaver_wrapper(*args, **kwargs):
    with _FlagOverrider(**overrides):
      return func(*args, **kwargs)
  return _flagsaver_wrapper
class _FlagOverrider(object):
  def __init__(self, **overrides):
    self._overrides = overrides
    self._saved_flag_values = None
  def __call__(self, func):
    if inspect.isclass(func):
      raise TypeError('flagsaver cannot be applied to a class.')
    return _wrap(func, self._overrides)
  def __enter__(self):
    self._saved_flag_values = save_flag_values(FLAGS)
    try:
      FLAGS._set_attributes(**self._overrides)
    except:
      restore_flag_values(self._saved_flag_values, FLAGS)
      raise
  def __exit__(self, exc_type, exc_value, traceback):
    restore_flag_values(self._saved_flag_values, FLAGS)
def _copy_flag_dict(flag):
  copy = flag.__dict__.copy()
  copy['_value'] = flag.value
  copy['validators'] = list(flag.validators)
  return copy
