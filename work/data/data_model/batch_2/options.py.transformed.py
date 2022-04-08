
import collections
from absl import logging
def _internal_attr_name(name):
  return "_" + name
class OptionsBase(object):
  def __init__(self):
    object.__setattr__(self, "_options", {})
    object.__setattr__(self, "_mutable", True)
  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return NotImplemented
      if getattr(self, name) != getattr(other, name):
        return False
    return True
  def __ne__(self, other):
    if isinstance(other, self.__class__):
      return not self.__eq__(other)
    else:
      return NotImplemented
  def __setattr__(self, name, value):
    if not self._mutable:
      raise ValueError("Mutating `tf.data.Options()` returned by "
                       "`tf.data.Dataset.options()` has no effect. Use "
                       "`tf.data.Dataset.with_options(options)` to set or "
                       "update dataset options.")
    if hasattr(self, name):
      object.__setattr__(self, name, value)
    else:
      raise AttributeError("Cannot set the property {} on {}.".format(
          name,
          type(self).__name__))
  def _set_mutable(self, mutable):
    object.__setattr__(self, "_mutable", mutable)
  def _to_proto(self):
    raise NotImplementedError("{}._to_proto()".format(type(self).__name__))
  def _from_proto(self, pb):
    raise NotImplementedError("{}._from_proto()".format(type(self).__name__))
def graph_rewrites():
  return collections.namedtuple("GraphRewrites",
                                ["enabled", "disabled", "default"])
def create_option(name, ty, docstring, default_factory=lambda: None):
  def get_fn(option):
    if name not in option._options:
      option._options[name] = default_factory()
    return option._options.get(name)
  def set_fn(option, value):
    if not isinstance(value, ty):
      raise TypeError(
          "Property \"{}\" must be of type {}, got: {} (type: {})".format(
              name, ty, value, type(value)))
  return property(get_fn, set_fn, None, docstring)
def merge_options(*options_list):
  """Merges the given options, returning the result as a new options object.
  The input arguments are expected to have a matching type that derives from
  `tf.data.OptionsBase` (and thus each represent a set of options). The method
  outputs an object of the same type created by merging the sets of options
  represented by the input arguments.
  If an option is set to different values by different options objects, the
  result will match the setting of the options object that appears in the input
  list last.
  If an option is an instance of `tf.data.OptionsBase` itself, then this method
  is applied recursively to the set of options represented by this option.
  Args:
    *options_list: options to merge
  Raises:
    TypeError: if the input arguments are incompatible or not derived from
      `tf.data.OptionsBase`
  Returns:
    A new options object which is the result of merging the given options.
  """
  if len(options_list) < 1:
    raise ValueError("At least one options should be provided")
  result_type = type(options_list[0])
  for options in options_list:
    if not isinstance(options, result_type):
      raise TypeError(
          "Could not merge incompatible options of type {} and {}.".format(
              type(options), result_type))
  if not isinstance(options_list[0], OptionsBase):
    raise TypeError(
        "All options to be merged should inherit from `OptionsBase` but found "
        "option of type {} which does not.".format(type(options_list[0])))
  default_options = result_type()
  result = result_type()
  for options in options_list:
      this = getattr(result, name)
      that = getattr(options, name)
      default = getattr(default_options, name)
      if that == default:
        continue
      elif this == default:
        setattr(result, name, that)
      elif isinstance(this, OptionsBase):
        setattr(result, name, merge_options(this, that))
      elif this != that:
        logging.warning("Changing the value of option %s from %r to %r.", name,
                        this, that)
        setattr(result, name, that)
  return result
