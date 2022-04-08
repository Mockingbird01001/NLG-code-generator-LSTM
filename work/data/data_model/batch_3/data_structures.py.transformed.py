
import collections
import copy
import operator
import sys
import six
try:
  import wrapt
except ImportError:
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import layer_utils
from tensorflow.python.util import lazy_loader
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
module = lazy_loader.LazyLoader(
    "module", globals(), "tensorflow.python.module.module")
class NoDependency(object):
  """Allows attribute assignment to `Trackable` objects with no dependency.
  Example usage:
  ```python
  obj = Trackable()
  obj.has_dependency = tf.Variable(0., name="dep")
  obj.no_dependency = NoDependency(tf.Variable(1., name="nodep"))
  assert obj.no_dependency.name == "nodep:0"
  ```
  `obj` in this example has a dependency on the variable "dep", and both
  attributes contain un-wrapped `Variable` objects.
  `NoDependency` also works with `tf.keras.Model`, but only for checkpoint
  dependencies: wrapping a `Layer` in `NoDependency` will assign the (unwrapped)
  `Layer` to the attribute without a checkpoint dependency, but the `Model` will
  still track the `Layer` (so it will appear in `Model.layers`, and its
  variables will appear in `Model.variables`).
  """
  __slots__ = ["value"]
  def __init__(self, value):
    self.value = value
def _should_wrap_tuple(t):
  for element in t:
    if isinstance(element, NoDependency):
    if isinstance(element, base.Trackable):
      return True
    if type(element) == dict:
      return True
    if type(element) == collections.OrderedDict:
      return True
    if type(element) == list:
      return True
    if isinstance(element, tuple) and _should_wrap_tuple(element):
      return True
  return False
@tf_export("__internal__.tracking.wrap", v1=[])
def wrap_or_unwrap(value):
  if isinstance(value, NoDependency):
    return value.value
  if isinstance(value, base.Trackable):
  elif type(value) == dict:
    return _DictWrapper(value)
  elif type(value) == collections.OrderedDict:
    return _DictWrapper(value)
  elif type(value) == list:
    return ListWrapper(value)
  elif isinstance(value, tuple) and _should_wrap_tuple(value):
    return _TupleWrapper(value)
  else:
    return value
@tf_export("__internal__.tracking.sticky_attribute_assignment", v1=[])
def sticky_attribute_assignment(trackable, name, value):
  """Adds dependencies, generally called from __setattr__.
  This behavior is shared between Trackable and Model.
  Respects NoDependency indicators, but otherwise makes trackable objects
  out of common data structures and tracks objects by their attribute names.
  Args:
    trackable: The object to add dependencies to (generally the one having
      an attribute assigned).
    name: The attribute name being assigned.
    value: The value being assigned. Not necessarily a trackable object.
  Returns:
    The value which should be stored in the attribute (unwrapped from a
    NoDependency object if necessary).
  """
  if isinstance(value, NoDependency):
    add_dependency = False
  else:
    add_dependency = True
  value = wrap_or_unwrap(value)
  if not add_dependency:
    return value
  if isinstance(value, base.Trackable):
        value, name=name,
        overwrite=True)
  return value
class _UntrackableError(ValueError):
    self._value = value
  def __str__(self):
    return ("Only trackable objects (such as Layers or Optimizers) may be "
            f"stored in a List object. Got {self._value}, which does not "
            "inherit from Trackable.")
@tf_export("__internal__.tracking.TrackableDataStructure", v1=[])
class TrackableDataStructure(base.Trackable):
  def __init__(self):
    self._self_trainable = True
    self._self_extra_variables = []
    self._self_attribute_sentinel = layer_utils.AttributeSentinel(True)
  @property
  def _attribute_sentinel(self):
    return self._self_attribute_sentinel
  @property
  def trainable(self):
    return self._self_trainable
  @trainable.setter
  def trainable(self, value):
    self._self_trainable = value
  def _track_value(self, value, name):
    value = sticky_attribute_assignment(
        trackable=self, value=value, name=name)
    if isinstance(value, variables.Variable):
      self._self_extra_variables.append(value)
    if not isinstance(value, base.Trackable):
      raise _UntrackableError(value)
    if hasattr(value, "_use_resource_variables"):
    value_attribute_sentinel = getattr(value, "_attribute_sentinel", None)
    if value_attribute_sentinel:
      value_attribute_sentinel.add_parent(self._attribute_sentinel)
    return value
  @property
  def _values(self):
    raise NotImplementedError("Abstract method")
  @property
  def _layers(self):
    collected = []
    for obj in self._values:
      if (isinstance(obj, TrackableDataStructure)
          or layer_utils.is_layer(obj)
          or layer_utils.has_weights(obj)):
        collected.append(obj)
    return collected
  @property
  def layers(self):
    return list(layer_utils.filter_empty_layer_containers(self._layers))
  @property
  def trainable_weights(self):
    if not self._self_trainable:
      return []
    trainable_variables = []
    for obj in self._values:
      if isinstance(obj, (TrackableDataStructure, module.Module)):
        trainable_variables += obj.trainable_variables
    trainable_extra_variables = [
        v for v in self._self_extra_variables if v.trainable
    ]
    return trainable_variables + trainable_extra_variables
  @property
  def non_trainable_weights(self):
    trainable_extra_variables = [
        v for v in self._self_extra_variables if v.trainable
    ]
    non_trainable_extra_variables = [
        v for v in self._self_extra_variables if not v.trainable
    ]
    non_trainable_variables = []
    for obj in self._values:
      if isinstance(obj, (TrackableDataStructure, module.Module)):
        non_trainable_variables += obj.non_trainable_variables
    if not self._self_trainable:
      trainable_variables = []
      for obj in self._values:
        if isinstance(obj, (TrackableDataStructure, module.Module)):
          trainable_variables += obj.trainable_variables
      non_trainable_variables = (
          trainable_variables + trainable_extra_variables +
          non_trainable_variables + non_trainable_extra_variables)
    else:
      non_trainable_variables = (
          non_trainable_variables + non_trainable_extra_variables)
    return non_trainable_variables
  @property
  def weights(self):
    return self.trainable_weights + self.non_trainable_weights
  @property
  def trainable_variables(self):
    return self.trainable_weights
  @property
  def non_trainable_variables(self):
    return self.non_trainable_weights
  @property
  def variables(self):
    return self.weights
  @property
  def updates(self):
    aggregated = []
    for layer in self.layers:
      if hasattr(layer, "updates"):
        aggregated += layer.updates
    return aggregated
  @property
  def losses(self):
    aggregated = []
    for layer in self.layers:
      if hasattr(layer, "losses"):
        aggregated += layer.losses
    return aggregated
  def __hash__(self):
    return id(self)
  def __eq__(self, other):
    return self is other
class List(TrackableDataStructure, collections_abc.Sequence):
  """An append-only sequence type which is trackable.
  Maintains checkpoint dependencies on its contents (which must also be
  trackable), and forwards any `Layer` metadata such as updates and losses.
  Note that `List` is purely a container. It lets a `tf.keras.Model` or
  other trackable object know about its contents, but does not call any
  `Layer` instances which are added to it. To indicate a sequence of `Layer`
  instances which should be called sequentially, use `tf.keras.Sequential`.
  Example usage:
  ```python
  class HasList(tf.keras.Model):
    def __init__(self):
      super(HasList, self).__init__()
      self.layer_list = List([layers.Dense(3)])
      self.layer_list.append(layers.Dense(4))
    def call(self, x):
      aggregation = 0.
      for l in self.layer_list:
        x = l(x)
        aggregation += tf.reduce_sum(x)
      return aggregation
  ```
  This kind of wrapping is necessary because `Trackable` objects do not
  (yet) deeply inspect regular Python data structures, so for example assigning
  a regular list (`self.layer_list = [layers.Dense(3)]`) does not create a
  checkpoint dependency and does not add the `Layer` instance's weights to its
  parent `Model`.
  """
  def __init__(self, *args, **kwargs):
    super(List, self).__init__()
    self._storage = self._make_storage(*args, **kwargs)
    for index, element in enumerate(self._storage):
      self._storage[index] = self._track_value(
          element, name=self._name_element(index))
  def copy(self):
    return type(self)(copy.copy(self._storage))
  def __copy__(self):
    return self.copy()
  def __deepcopy__(self, memo):
    return type(self)(copy.deepcopy(self._storage, memo))
  def _make_storage(self, *args, **kwargs):
    return list(*args, **kwargs)
  def _name_element(self, index):
    return "%d" % (index,)
  @property
  def _values(self):
    return self
  def append(self, value):
    value = self._track_value(value, self._name_element(len(self._storage)))
    self._storage.append(value)
  def extend(self, values):
    for value in values:
      self.append(value)
  def __iadd__(self, values):
    self.extend(values)
    return self
  def __add__(self, other):
    return self._storage + getattr(other, "_storage", other)
  def __imul__(self, y):
    if y <= 0:
      raise ValueError(
          f"List only supports append, multiplying in place by {y} removes "
          "elements.")
    n = len(self._storage)
    for _ in range(y - 1):
      for i in range(n):
        self.append(self._storage[i])
    return self
  def __mul__(self, n):
    return self._storage * n
  def __rmul__(self, n):
    return self * n
  def __radd__(self, other):
    return other + self._storage
  def __getitem__(self, key):
    return self._storage[key]
  def __getslice__(self, i, j):
    return self._storage[slice(i, j)]
  def __len__(self):
    return len(self._storage)
  def __repr__(self):
    return "List(%s)" % (repr(self._storage),)
  def __sizeof__(self):
    return super(List, self).__sizeof__() + sys.getsizeof(self._storage)
class ListWrapper(
    List,
    collections_abc.MutableSequence,
    list):
  """Wraps the built-in `list` to support restore-on-create for variables.
  Unlike `List`, this sequence type is mutable in the same ways built-in lists
  are. Instead of throwing an error immediately like `List`, it records
  problematic mutations (e.g. assigning a new element to a position already
  occupied, meaning both elements get the same names at different times) and
  refuses to save.
  On assignment to an attribute of a Model or Trackable object, Python
  lists are replaced with ListWrapper. Wrapping a list in a
  `NoDependency` object prevents this.
  """
  def __init__(self, wrapped_list):
    self._non_append_mutation_value = False
    self._external_modification_value = False
    super(ListWrapper, self).__init__(wrapped_list)
    self._last_wrapped_list_snapshot = list(self._storage)
  @property
  def _non_append_mutation(self):
    return self._non_append_mutation_value
  @_non_append_mutation.setter
  def _non_append_mutation(self, value):
    self._attribute_sentinel.invalidate_all()
    self._non_append_mutation_value = value
  @property
  def _external_modification(self):
    return self._external_modification_value
  @_external_modification.setter
  def _external_modification(self, value):
    self._attribute_sentinel.invalidate_all()
    self._external_modification_value = value
  def __copy__(self):
    copied = super(ListWrapper, self).__copy__()
    copied._non_append_mutation = self._non_append_mutation
    copied._external_modification = self._external_modification
    return copied
  def __deepcopy__(self, memo):
    copied = super(ListWrapper, self).__deepcopy__(memo)
    copied._non_append_mutation = self._non_append_mutation
    copied._external_modification = self._external_modification
    return copied
  def __reduce_ex__(self, protocol):
    return (self.__class__,
            (self._storage,))
  def _make_storage(self, wrapped_list):
    return wrapped_list
  def _check_external_modification(self):
    if self._external_modification or self._non_append_mutation:
      return
    if self._storage != self._last_wrapped_list_snapshot:
      self._external_modification = True
      self._last_wrapped_list_snapshot = None
  def _update_snapshot(self):
    self._attribute_sentinel.invalidate_all()
    if self._external_modification or self._non_append_mutation:
      return
    self._last_wrapped_list_snapshot = list(self._storage)
  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    self._check_external_modification()
    if self._non_append_mutation:
      raise ValueError(
          f"Unable to save the object {self} (a list wrapper constructed to "
          "track trackable TensorFlow objects). A list element was replaced "
          "(__setitem__, __setslice__), deleted (__delitem__, __delslice__), "
          "or moved (sort). In order to support restoration on object "
          "creation, tracking is exclusively for append-only data structures."
          "\n\nIf you don't need this list checkpointed, wrap it in a "
          "non-trackable object; it will be subsequently ignored.")
    if self._external_modification:
      raise ValueError(
          f"Unable to save the object {self} (a list wrapper constructed to "
          "track trackable TensorFlow objects). The wrapped list was modified "
          f"outside the wrapper (its final value was {self._storage}, its value"
          " when a checkpoint dependency was added was "
          f"{self._last_wrapped_list_snapshot}), which breaks "
          "restoration on object creation.\n\nIf you don't need this list "
          "checkpointed, wrap it in a NoDependency object; it will be "
          "subsequently ignored.")
    children = super(ListWrapper, self)._trackable_children(save_type, **kwargs)
    if save_type == base.SaveType.SAVEDMODEL:
      children.update({
          str(key): value
          for key, value in enumerate(self)
          if _is_function(value)
      })
    return children
  def _has_mutation_or_trackable(self):
    if self._non_append_mutation:
      return True
    return any(isinstance(element, base.Trackable) for element in self._storage)
  def __delitem__(self, key):
    self._check_external_modification()
    if self._has_mutation_or_trackable():
      self._non_append_mutation = True
    del self._storage[key]
    self._update_snapshot()
  def __setitem__(self, key, value):
    self._check_external_modification()
    if isinstance(key, slice):
      storage_copy = list(self._storage)
      self._storage[key] = value
      len_before = len(storage_copy)
      len_now = len(self._storage)
      for i in range(max(len_before, len_now)):
        value_now = self._storage[i] if i < len_now else None
        value_before = storage_copy[i] if i < len_before else None
        if isinstance(value_before, base.Trackable):
          self._non_append_mutation = True
        if value_now is not None and value_now != value_before:
          self._storage[i] = self._track_value(self._storage[i],
                                               self._name_element(i))
    else:
      if isinstance(self._storage[key], base.Trackable):
        self._non_append_mutation = True
      self._storage[key] = self._track_value(value, self._name_element(key))
    self._update_snapshot()
  def append(self, value):
    self._check_external_modification()
    super(ListWrapper, self).append(value)
    self._update_snapshot()
  def extend(self, values):
    self._check_external_modification()
    super(ListWrapper, self).extend(values)
    self._update_snapshot()
  def __imul__(self, y):
    if y <= 0:
      self._check_external_modification()
      if self._has_mutation_or_trackable():
        self._non_append_mutation = True
      self._storage *= y
      self._update_snapshot()
      return self
    return super(ListWrapper, self).__imul__(y)
  def __eq__(self, other):
    return self._storage == getattr(other, "_storage", other)
  def __ne__(self, other):
    return self._storage != getattr(other, "_storage", other)
  def __lt__(self, other):
    return self._storage < getattr(other, "_storage", other)
  def __le__(self, other):
    return self._storage <= getattr(other, "_storage", other)
  def __gt__(self, other):
    return self._storage > getattr(other, "_storage", other)
  def __ge__(self, other):
    return self._storage >= getattr(other, "_storage", other)
  def __hash__(self):
    raise TypeError("unhashable type: 'ListWrapper'")
  def insert(self, index, obj):
    self._check_external_modification()
    if (self._has_mutation_or_trackable() or isinstance(obj, base.Trackable)):
      self._non_append_mutation = True
    self._storage.insert(index, obj)
    self._update_snapshot()
  def sort(self):
    self._check_external_modification()
    if self._has_mutation_or_trackable():
      self._non_append_mutation = True
    self._storage.sort()
    self._update_snapshot()
  def __setslice__(self, i, j, y):
    self.__setitem__(slice(i, j), y)
  def __delslice__(self, i, j):
    self._check_external_modification()
    if self._has_mutation_or_trackable():
      self._non_append_mutation = True
    del self._storage[slice(i, j)]
    self._update_snapshot()
  def _track_value(self, value, name):
    try:
      value = super(ListWrapper, self)._track_value(value=value, name=name)
    except ValueError:
      value = sticky_attribute_assignment(
          trackable=self, value=value, name=name)
    return value
  def __repr__(self):
    return "ListWrapper(%s)" % (repr(self._storage),)
class Mapping(TrackableDataStructure, collections_abc.Mapping):
  """An append-only trackable mapping data structure with string keys.
  Maintains checkpoint dependencies on its contents (which must also be
  trackable), named based on its keys.
  Note that once a key has been added, it may not be deleted or replaced.
  """
  def __init__(self, *args, **kwargs):
    super(Mapping, self).__init__()
    self._storage = self._make_storage(*args, **kwargs)
    self._storage.update(
        {key: self._track_value(
            value, name=self._name_element(key))
         for key, value in self._storage.items()})
  def __copy__(self):
    return type(self)(copy.copy(self._storage))
  def __deepcopy__(self, memo):
    return type(self)(copy.deepcopy(self._storage, memo))
  def _make_storage(self, *args, **kwargs):
    return dict(*args, **kwargs)
  @property
  def _values(self):
    ordered = list(zip(*sorted(self.items(), key=lambda it: it[0])))
    if ordered:
      return ordered[1]
    return []
  def _name_element(self, key):
    if not isinstance(key, six.string_types):
      raise TypeError(
          f"Mapping accepts only string keys, but got a key {repr(key)}.")
    return str(key)
  def __setitem__(self, key, value):
    name = self._name_element(key)
    value = self._track_value(value, name=name)
    current_value = self._storage.setdefault(key, value)
    if current_value is not value:
      raise ValueError(
          "Mappings are an append-only data structure. Tried to overwrite the "
          f"key '{key}' with value {value}, but it already contains "
          f"{current_value}")
  def update(self, *args, **kwargs):
    for key, value in dict(*args, **kwargs).items():
      self[key] = value
  def __getitem__(self, key):
    return self._storage[key]
  def __len__(self):
    return len(self._storage)
  def __repr__(self):
    return "Mapping(%s)" % (repr(self._storage),)
  def __iter__(self):
    return iter(self._storage)
class _DictWrapper(TrackableDataStructure, wrapt.ObjectProxy):
  """Wraps built-in dicts to support restore-on-create for variables.
  _DictWrapper is to Mapping as ListWrapper is to List. Unlike Mapping,
  _DictWrapper allows non-string keys and values and arbitrary mutations (delete
  keys, reassign values). Like ListWrapper, these mutations mean that
  _DictWrapper will raise an exception on save.
  """
  def __init__(self, wrapped_dict=None):
    if wrapped_dict is None:
      wrapped_dict = {}
    if not isinstance(wrapped_dict, collections_abc.Mapping):
      wrapped_dict = dict(wrapped_dict)
    wrapt.ObjectProxy.__init__(self, wrapped_dict)
    TrackableDataStructure.__init__(self)
    self._self_non_string_key = False
    self._self_external_modification = False
    self.__wrapped__.update(
        {key: self._track_value(
            value, name=self._name_element(key))
         for key, value in self.__wrapped__.items()})
    self._update_snapshot()
  def __reduce_ex__(self, protocol):
    return (self.__class__,
            (self.__wrapped__,))
  def __getattribute__(self, name):
    if (hasattr(type(self), name)
        and isinstance(getattr(type(self), name), property)):
      return object.__getattribute__(self, name)
    else:
      return super(_DictWrapper, self).__getattribute__(name)
  def copy(self):
    return copy.copy(self)
  def __copy__(self):
    copied = _DictWrapper(copy.copy(self.__wrapped__))
    copied._self_external_modification = self._self_external_modification
    copied._self_non_string_key = self._self_non_string_key
    return copied
  def __deepcopy__(self, memo):
    copied = _DictWrapper(copy.deepcopy(self.__wrapped__, memo))
    copied._self_external_modification = self._self_external_modification
    copied._self_non_string_key = self._self_non_string_key
    return copied
  @property
  def _values(self):
    ordered = list(zip(*sorted(self.items(), key=lambda it: it[0])))
    if ordered:
      return ordered[1]
    return []
  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    self._check_self_external_modification()
    if self._self_non_string_key:
      raise ValueError(
          f"Unable to save the object {self} (a dictionary wrapper constructed "
          "automatically on attribute assignment). The wrapped dictionary "
          "contains a non-string key which maps to a trackable object or "
          "mutable data structure.\n\nIf you don't need this dictionary "
          "checkpointed, wrap it in a non-trackable "
          "object; it will be subsequently ignored.")
    if self._self_external_modification:
      raise ValueError(
          f"Unable to save the object {self} (a dictionary wrapper constructed "
          "automatically on attribute assignment). The wrapped dictionary was "
          f"modified outside the wrapper (its final value was {self}, its value"
          " when a checkpoint dependency was added was "
          f"{self._self_last_wrapped_dict_snapshot}), which breaks "
          "restoration on object creation.\n\nIf you don't need this "
          "dictionary checkpointed, wrap it in a "
          "non-trackable object; it will be subsequently ignored.")
    children = super(_DictWrapper,
                     self)._trackable_children(save_type, **kwargs)
    if save_type == base.SaveType.SAVEDMODEL:
      children.update(
          {key: value for key, value in self.items() if _is_function(value)})
    return children
  @property
  def _dirty(self):
    return (self._self_external_modification
            or self._self_non_string_key)
  def _check_self_external_modification(self):
    if self._dirty:
      return
    if self != self._self_last_wrapped_dict_snapshot:
      self._self_external_modification = True
      self._self_last_wrapped_dict_snapshot = None
  def _update_snapshot(self):
    self._attribute_sentinel.invalidate_all()
    if self._dirty:
      return
    self._self_last_wrapped_dict_snapshot = dict(self)
  def _track_value(self, value, name):
    if isinstance(name, six.string_types):
      string_key = True
    else:
      name = "-non_string_key"
      string_key = False
    try:
      no_dependency = isinstance(value, NoDependency)
      value = super(_DictWrapper, self)._track_value(value=value, name=name)
      if not (string_key or no_dependency):
        self._self_non_string_key = True
      return value
    except ValueError:
      return sticky_attribute_assignment(
          trackable=self, value=value, name=name)
  def _name_element(self, key):
    return key
  def __setitem__(self, key, value):
    self._check_self_external_modification()
    self._maybe_initialize_trackable()
    no_dep = isinstance(value, NoDependency)
    if isinstance(key, six.string_types):
      value = self._track_value(value, name=key)
    else:
      value = wrap_or_unwrap(value)
      if not no_dep and isinstance(value, base.Trackable):
        self._self_non_string_key = True
    self.__wrapped__[key] = value
    self._update_snapshot()
  def __delitem__(self, key):
    self._check_self_external_modification()
    del self.__wrapped__[key]
    self._update_snapshot()
  def __repr__(self):
    return "DictWrapper(%s)" % (repr(self.__wrapped__),)
  def __hash__(self):
    raise TypeError("unhashable type: 'DictWrapper'")
  def __eq__(self, other):
    return self.__wrapped__ == other
  def update(self, *args, **kwargs):
    for key, value in six.iteritems(dict(*args, **kwargs)):
      self[key] = value
class _TupleWrapper(TrackableDataStructure, wrapt.ObjectProxy):
  def __init__(self, original_wrapped_tuple=()):
    add_dependency = []
    substituted_wrapped_tuple = []
    for element in original_wrapped_tuple:
      if isinstance(element, NoDependency):
        add_dependency.append(False)
      else:
        add_dependency.append(True)
      substituted_wrapped_tuple.append(wrap_or_unwrap(element))
    try:
      fields = original_wrapped_tuple._fields
    except AttributeError:
      is_namedtuple = False
    else:
      is_namedtuple = True
    original_type = type(original_wrapped_tuple)
    self._self_tuple_is_constructable = True
    if is_namedtuple:
      try:
        substituted_wrapped_tuple = original_type(
            **dict(zip(fields, substituted_wrapped_tuple)))
      except TypeError:
        wrapt.ObjectProxy.__init__(self, original_wrapped_tuple)
        TrackableDataStructure.__init__(self)
        self._self_tuple_is_constructable = False
        return
    else:
      substituted_wrapped_tuple = original_type(substituted_wrapped_tuple)
    wrapt.ObjectProxy.__init__(self, substituted_wrapped_tuple)
    TrackableDataStructure.__init__(self)
    if is_namedtuple:
      for name, should_depend, element in zip(
          fields, add_dependency, substituted_wrapped_tuple):
        if should_depend:
          self._track_value(element, name=name)
    for index, (should_depend, element) in enumerate(
        zip(add_dependency, substituted_wrapped_tuple)):
      if should_depend:
        self._track_value(element, name="%d" % (index,))
  @property
  def _values(self):
    return self
  def _track_value(self, value, name):
    try:
      value = super(_TupleWrapper, self)._track_value(value=value, name=name)
    except ValueError:
      value = sticky_attribute_assignment(
          trackable=self, value=value, name=name)
    return value
  def __repr__(self):
    return "_TupleWrapper(%s)" % (repr(self.__wrapped__),)
  def __hash__(self):
    return hash(self.__wrapped__)
  def __eq__(self, other):
    return self.__wrapped__ == other
  def __copy__(self):
    return _TupleWrapper(copy.copy(self.__wrapped__))
  def __deepcopy__(self, memo):
    return _TupleWrapper(copy.deepcopy(self.__wrapped__, memo))
  def __reduce_ex__(self, protocol):
    return (self.__class__,
            (self.__wrapped__,))
  def __imul__(self, y):
    return self.__wrapped__ * y
  def __iadd__(self, y):
    return self.__wrapped__ + y
  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    if not self._self_tuple_is_constructable:
      raise ValueError(
          f"Unable to save because the namedtuple {self.__wrapped__} is not "
          "constructable from its _fields (i.e. __new__ is overridden). "
          f"Expected keyword arguments {self.__wrapped__._fields}. If you do "
          "not need to save this object, consider wrapping it in a custom "
          "object that does not inherit from tuple.")
    return super(_TupleWrapper, self)._trackable_children(save_type, **kwargs)
  def __getattribute__(self, name):
    if name != "__wrapped__" and hasattr(self.__wrapped__, name):
      return getattr(self.__wrapped__, name)
    if (hasattr(type(self), name)
        and isinstance(getattr(type(self), name), property)):
      return object.__getattribute__(self, name)
    else:
      return super(_TupleWrapper, self).__getattribute__(name)
def _is_function(x):
  return isinstance(x, (def_function.Function, defun.ConcreteFunction))
revived_types.register_revived_type(
    "trackable_dict_wrapper",
    lambda obj: isinstance(obj, _DictWrapper),
    versions=[revived_types.VersionedTypeRegistration(
        object_factory=lambda proto: _DictWrapper({}),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
        setter=operator.setitem)])
def _set_list_item(list_object, index_string, value):
  item_index = int(index_string)
  if len(list_object) <= item_index:
    list_object.extend([None] * (1 + item_index - len(list_object)))
  list_object[item_index] = value
revived_types.register_revived_type(
    "trackable_list_wrapper",
    lambda obj: isinstance(obj, ListWrapper),
    versions=[revived_types.VersionedTypeRegistration(
        object_factory=lambda proto: ListWrapper([]),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
        setter=_set_list_item)])
def _set_tuple_item(list_object, index_string, value):
  try:
    item_index = int(index_string)
  except ValueError:
    return
  if len(list_object) <= item_index:
    list_object.extend([None] * (1 + item_index - len(list_object)))
  list_object[item_index] = value
revived_types.register_revived_type(
    "trackable_tuple_wrapper",
    lambda obj: isinstance(obj, _TupleWrapper),
    versions=[revived_types.VersionedTypeRegistration(
        object_factory=lambda proto: ListWrapper([]),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
        setter=_set_tuple_item)])
