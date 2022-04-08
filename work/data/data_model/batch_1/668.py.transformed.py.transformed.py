
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import types
import tensorflow.compat.v1 as tf
def function_with_default(f, default):
  return types.FunctionType(f.__code__, f.__globals__, f.__name__,
                            (default,), f.__closure__)
def add_functions_to_module(function_dict, module_dict=None):
  if module_dict is None:
    module_dict = globals()
  for name in function_dict:
    module_dict[name] = function_dict[name]
def merge_prefix(prefix, key):
  if prefix:
    return "/".join((prefix, key))
  else:
    return key
def has_context(key, sequence, prefix=""):
  return merge_prefix(prefix, key) in sequence.context.feature
def clear_context(key, sequence, prefix=""):
  del sequence.context.feature[merge_prefix(prefix, key)]
def set_context_float(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(prefix, key)].float_list.value[:] = (
      value,)
def get_context_float(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(
      prefix, key)].float_list.value[0]
def set_context_bytes(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].bytes_list.value[:] = (value,)
def get_context_bytes(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].bytes_list.value[0]
def set_context_int(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].int64_list.value[:] = (value,)
def get_context_int(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].int64_list.value[0]
def set_context_float_list(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].float_list.value[:] = value
def get_context_float_list(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].float_list.value
def set_context_bytes_list(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].bytes_list.value[:] = value
def get_context_bytes_list(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].bytes_list.value
def set_context_int_list(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].int64_list.value[:] = value
def get_context_int_list(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].int64_list.value
def has_feature_list(key, sequence, prefix=""):
  return merge_prefix(prefix, key) in sequence.feature_lists.feature_list
def get_feature_list_size(key, sequence, prefix=""):
  if has_feature_list(merge_prefix(prefix, key), sequence):
    return len(sequence.feature_lists.feature_list[merge_prefix(
        prefix, key)].feature)
  else:
    return 0
def clear_feature_list(key, sequence, prefix=""):
  del sequence.feature_lists.feature_list[merge_prefix(prefix, key)]
def get_float_list_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].float_list.value
def get_int_list_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].int64_list.value
def get_bytes_list_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].bytes_list.value
def add_float_list(key, values, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).float_list.value[:] = values
def add_bytes_list(key, values, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).bytes_list.value[:] = values
def add_int_list(key, values, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).int64_list.value[:] = values
def get_float_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].float_list.value[0]
def get_int_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].int64_list.value[0]
def get_bytes_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].bytes_list.value[0]
def add_float(key, value, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).float_list.value[:] = (value,)
def add_bytes(key, value, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).bytes_list.value[:] = (value,)
def add_int(key, value, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).int64_list.value[:] = (value,)
def create_bytes_list_context_feature(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)
  def _get(sequence_example, prefix=prefix):
    return get_context_bytes_list(key, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)
  def _set(value, sequence_example, prefix=prefix):
    set_context_bytes_list(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.VarLenFeature(tf.string)
  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_float_list_context_feature(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)
  def _get(sequence_example, prefix=prefix):
    return get_context_float_list(key, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)
  def _set(value, sequence_example, prefix=prefix):
    set_context_float_list(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.VarLenFeature(tf.float32)
  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_int_list_context_feature(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)
  def _get(sequence_example, prefix=prefix):
    return get_context_int_list(key, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)
  def _set(value, sequence_example, prefix=prefix):
    set_context_int_list(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.VarLenFeature(tf.int64)
  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_bytes_context_feature(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)
  def _get(sequence_example, prefix=prefix):
    return get_context_bytes(key, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)
  def _set(value, sequence_example, prefix=prefix):
    set_context_bytes(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.FixedLenFeature((), tf.string)
  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_float_context_feature(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)
  def _get(sequence_example, prefix=prefix):
    return get_context_float(key, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)
  def _set(value, sequence_example, prefix=prefix):
    set_context_float(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.FixedLenFeature((), tf.float32)
  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_int_context_feature(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)
  def _get(sequence_example, prefix=prefix):
    return get_context_int(key, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)
  def _set(value, sequence_example, prefix=prefix):
    set_context_int(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.FixedLenFeature((), tf.int64)
  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_bytes_feature_list(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)
  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)
  def _get_at(index, sequence_example, prefix=prefix):
    return get_bytes_at(key, index, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)
  def _add(value, sequence_example, prefix=prefix):
    add_bytes(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.FixedLenSequenceFeature((), tf.string)
  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_float_feature_list(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)
  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)
  def _get_at(index, sequence_example, prefix=prefix):
    return get_float_at(key, index, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)
  def _add(value, sequence_example, prefix=prefix):
    add_float(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.FixedLenSequenceFeature((), tf.float32)
  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_int_feature_list(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)
  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)
  def _get_at(index, sequence_example, prefix=prefix):
    return get_int_at(key, index, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)
  def _add(value, sequence_example, prefix=prefix):
    add_int(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.FixedLenSequenceFeature((), tf.int64)
  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_bytes_list_feature_list(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)
  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)
  def _get_at(index, sequence_example, prefix=prefix):
    return get_bytes_list_at(key, index, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)
  def _add(value, sequence_example, prefix=prefix):
    add_bytes_list(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.VarLenFeature(tf.string)
  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_float_list_feature_list(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)
  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)
  def _get_at(index, sequence_example, prefix=prefix):
    return get_float_list_at(key, index, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)
  def _add(value, sequence_example, prefix=prefix):
    add_float_list(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.VarLenFeature(tf.float32)
  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
def create_int_list_feature_list(name, key, prefix="", module_dict=None):
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)
  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)
  def _get_at(index, sequence_example, prefix=prefix):
    return get_int_list_at(key, index, sequence_example, prefix=prefix)
  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)
  def _add(value, sequence_example, prefix=prefix):
    add_int_list(key, value, sequence_example, prefix=prefix)
  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)
  def _get_default_parser():
    return tf.io.VarLenFeature(tf.int64)
  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
