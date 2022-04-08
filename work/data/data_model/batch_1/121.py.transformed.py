
"""Contains container classes to represent different protocol buffer types.
This file defines container classes which represent categories of protocol
buffer field types which need extra maintenance. Currently these categories
are:
-   Repeated scalar fields - These are all repeated fields which aren't
    composite (e.g. they are of simple types like int32, string, etc).
-   Repeated composite fields - Repeated fields which are composite. This
    includes groups and nested messages.
"""
__author__ = 'petar@google.com (Petar Petrov)'
import sys
try:
  import collections.abc as collections_abc
except ImportError:
  import collections as collections_abc
if sys.version_info[0] < 3:
  class Mapping(object):
    __slots__ = ()
    def get(self, key, default=None):
      try:
        return self[key]
      except KeyError:
        return default
    def __contains__(self, key):
      try:
        self[key]
      except KeyError:
        return False
      else:
        return True
    def iterkeys(self):
      return iter(self)
    def itervalues(self):
      for key in self:
        yield self[key]
    def iteritems(self):
      for key in self:
        yield (key, self[key])
    def keys(self):
      return list(self)
    def items(self):
      return [(key, self[key]) for key in self]
    def values(self):
      return [self[key] for key in self]
    __hash__ = None
    def __eq__(self, other):
      if not isinstance(other, collections_abc.Mapping):
        return NotImplemented
      return dict(self.items()) == dict(other.items())
    def __ne__(self, other):
      return not (self == other)
  class MutableMapping(Mapping):
    __slots__ = ()
    __marker = object()
    def pop(self, key, default=__marker):
      try:
        value = self[key]
      except KeyError:
        if default is self.__marker:
          raise
        return default
      else:
        del self[key]
        return value
    def popitem(self):
      try:
        key = next(iter(self))
      except StopIteration:
        raise KeyError
      value = self[key]
      del self[key]
      return key, value
    def clear(self):
      try:
        while True:
          self.popitem()
      except KeyError:
        pass
    def update(*args, **kwds):
      if len(args) > 2:
        raise TypeError("update() takes at most 2 positional "
                        "arguments ({} given)".format(len(args)))
      elif not args:
        raise TypeError("update() takes at least 1 argument (0 given)")
      self = args[0]
      other = args[1] if len(args) >= 2 else ()
      if isinstance(other, Mapping):
        for key in other:
          self[key] = other[key]
      elif hasattr(other, "keys"):
        for key in other.keys():
          self[key] = other[key]
      else:
        for key, value in other:
          self[key] = value
      for key, value in kwds.items():
        self[key] = value
    def setdefault(self, key, default=None):
      try:
        return self[key]
      except KeyError:
        self[key] = default
      return default
  collections_abc.Mapping.register(Mapping)
  collections_abc.MutableMapping.register(MutableMapping)
else:
  MutableMapping = collections_abc.MutableMapping
class BaseContainer(object):
  __slots__ = ['_message_listener', '_values']
  def __init__(self, message_listener):
    self._message_listener = message_listener
    self._values = []
  def __getitem__(self, key):
    return self._values[key]
  def __len__(self):
    return len(self._values)
  def __ne__(self, other):
    return not self == other
  def __hash__(self):
    raise TypeError('unhashable object')
  def __repr__(self):
    return repr(self._values)
  def sort(self, *args, **kwargs):
    if 'sort_function' in kwargs:
      kwargs['cmp'] = kwargs.pop('sort_function')
    self._values.sort(*args, **kwargs)
  def reverse(self):
    self._values.reverse()
collections_abc.MutableSequence.register(BaseContainer)
class RepeatedScalarFieldContainer(BaseContainer):
  __slots__ = ['_type_checker']
  def __init__(self, message_listener, type_checker):
    super(RepeatedScalarFieldContainer, self).__init__(message_listener)
    self._type_checker = type_checker
  def append(self, value):
    self._values.append(self._type_checker.CheckValue(value))
    if not self._message_listener.dirty:
      self._message_listener.Modified()
  def insert(self, key, value):
    self._values.insert(key, self._type_checker.CheckValue(value))
    if not self._message_listener.dirty:
      self._message_listener.Modified()
  def extend(self, elem_seq):
    if elem_seq is None:
      return
    try:
      elem_seq_iter = iter(elem_seq)
    except TypeError:
      if not elem_seq:
        return
      raise
    new_values = [self._type_checker.CheckValue(elem) for elem in elem_seq_iter]
    if new_values:
      self._values.extend(new_values)
    self._message_listener.Modified()
  def MergeFrom(self, other):
    self._values.extend(other._values)
    self._message_listener.Modified()
  def remove(self, elem):
    self._values.remove(elem)
    self._message_listener.Modified()
  def pop(self, key=-1):
    value = self._values[key]
    self.__delitem__(key)
    return value
  def __setitem__(self, key, value):
    if isinstance(key, slice):
      if key.step is not None:
        raise ValueError('Extended slices not supported')
      self.__setslice__(key.start, key.stop, value)
    else:
      self._values[key] = self._type_checker.CheckValue(value)
      self._message_listener.Modified()
  def __getslice__(self, start, stop):
    return self._values[start:stop]
  def __setslice__(self, start, stop, values):
    new_values = []
    for value in values:
      new_values.append(self._type_checker.CheckValue(value))
    self._values[start:stop] = new_values
    self._message_listener.Modified()
  def __delitem__(self, key):
    del self._values[key]
    self._message_listener.Modified()
  def __delslice__(self, start, stop):
    del self._values[start:stop]
    self._message_listener.Modified()
  def __eq__(self, other):
    if self is other:
      return True
    if isinstance(other, self.__class__):
      return other._values == self._values
    return other == self._values
class RepeatedCompositeFieldContainer(BaseContainer):
  __slots__ = ['_message_descriptor']
  def __init__(self, message_listener, message_descriptor):
    super(RepeatedCompositeFieldContainer, self).__init__(message_listener)
    self._message_descriptor = message_descriptor
  def add(self, **kwargs):
    new_element = self._message_descriptor._concrete_class(**kwargs)
    new_element._SetListener(self._message_listener)
    self._values.append(new_element)
    if not self._message_listener.dirty:
      self._message_listener.Modified()
    return new_element
  def append(self, value):
    new_element = self._message_descriptor._concrete_class()
    new_element._SetListener(self._message_listener)
    new_element.CopyFrom(value)
    self._values.append(new_element)
    if not self._message_listener.dirty:
      self._message_listener.Modified()
  def insert(self, key, value):
    new_element = self._message_descriptor._concrete_class()
    new_element._SetListener(self._message_listener)
    new_element.CopyFrom(value)
    self._values.insert(key, new_element)
    if not self._message_listener.dirty:
      self._message_listener.Modified()
  def extend(self, elem_seq):
    message_class = self._message_descriptor._concrete_class
    listener = self._message_listener
    values = self._values
    for message in elem_seq:
      new_element = message_class()
      new_element._SetListener(listener)
      new_element.MergeFrom(message)
      values.append(new_element)
    listener.Modified()
  def MergeFrom(self, other):
    self.extend(other._values)
  def remove(self, elem):
    self._values.remove(elem)
    self._message_listener.Modified()
  def pop(self, key=-1):
    value = self._values[key]
    self.__delitem__(key)
    return value
  def __getslice__(self, start, stop):
    return self._values[start:stop]
  def __delitem__(self, key):
    del self._values[key]
    self._message_listener.Modified()
  def __delslice__(self, start, stop):
    del self._values[start:stop]
    self._message_listener.Modified()
  def __eq__(self, other):
    if self is other:
      return True
    if not isinstance(other, self.__class__):
      raise TypeError('Can only compare repeated composite fields against '
                      'other repeated composite fields.')
    return self._values == other._values
class ScalarMap(MutableMapping):
  __slots__ = ['_key_checker', '_value_checker', '_values', '_message_listener',
               '_entry_descriptor']
  def __init__(self, message_listener, key_checker, value_checker,
               entry_descriptor):
    self._message_listener = message_listener
    self._key_checker = key_checker
    self._value_checker = value_checker
    self._entry_descriptor = entry_descriptor
    self._values = {}
  def __getitem__(self, key):
    try:
      return self._values[key]
    except KeyError:
      key = self._key_checker.CheckValue(key)
      val = self._value_checker.DefaultValue()
      self._values[key] = val
      return val
  def __contains__(self, item):
    self._key_checker.CheckValue(item)
    return item in self._values
  def get(self, key, default=None):
    if key in self:
      return self[key]
    else:
      return default
  def __setitem__(self, key, value):
    checked_key = self._key_checker.CheckValue(key)
    checked_value = self._value_checker.CheckValue(value)
    self._values[checked_key] = checked_value
    self._message_listener.Modified()
  def __delitem__(self, key):
    del self._values[key]
    self._message_listener.Modified()
  def __len__(self):
    return len(self._values)
  def __iter__(self):
    return iter(self._values)
  def __repr__(self):
    return repr(self._values)
  def MergeFrom(self, other):
    self._values.update(other._values)
    self._message_listener.Modified()
  def InvalidateIterators(self):
    original = self._values
    self._values = original.copy()
    original[None] = None
  def clear(self):
    self._values.clear()
    self._message_listener.Modified()
  def GetEntryClass(self):
    return self._entry_descriptor._concrete_class
class MessageMap(MutableMapping):
  __slots__ = ['_key_checker', '_values', '_message_listener',
               '_message_descriptor', '_entry_descriptor']
  def __init__(self, message_listener, message_descriptor, key_checker,
               entry_descriptor):
    self._message_listener = message_listener
    self._message_descriptor = message_descriptor
    self._key_checker = key_checker
    self._entry_descriptor = entry_descriptor
    self._values = {}
  def __getitem__(self, key):
    key = self._key_checker.CheckValue(key)
    try:
      return self._values[key]
    except KeyError:
      new_element = self._message_descriptor._concrete_class()
      new_element._SetListener(self._message_listener)
      self._values[key] = new_element
      self._message_listener.Modified()
      return new_element
  def get_or_create(self, key):
    return self[key]
  def get(self, key, default=None):
    if key in self:
      return self[key]
    else:
      return default
  def __contains__(self, item):
    item = self._key_checker.CheckValue(item)
    return item in self._values
  def __setitem__(self, key, value):
    raise ValueError('May not set values directly, call my_map[key].foo = 5')
  def __delitem__(self, key):
    key = self._key_checker.CheckValue(key)
    del self._values[key]
    self._message_listener.Modified()
  def __len__(self):
    return len(self._values)
  def __iter__(self):
    return iter(self._values)
  def __repr__(self):
    return repr(self._values)
  def MergeFrom(self, other):
    for key in other._values:
      if key in self:
        del self[key]
      self[key].CopyFrom(other[key])
  def InvalidateIterators(self):
    original = self._values
    self._values = original.copy()
    original[None] = None
  def clear(self):
    self._values.clear()
    self._message_listener.Modified()
  def GetEntryClass(self):
    return self._entry_descriptor._concrete_class
class _UnknownField(object):
  __slots__ = ['_field_number', '_wire_type', '_data']
  def __init__(self, field_number, wire_type, data):
    self._field_number = field_number
    self._wire_type = wire_type
    self._data = data
    return
  def __lt__(self, other):
    return self._field_number < other._field_number
  def __eq__(self, other):
    if self is other:
      return True
    return (self._field_number == other._field_number and
            self._wire_type == other._wire_type and
            self._data == other._data)
class UnknownFieldRef(object):
  def __init__(self, parent, index):
    self._parent = parent
    self._index = index
    return
  def _check_valid(self):
    if not self._parent:
      raise ValueError('UnknownField does not exist. '
                       'The parent message might be cleared.')
    if self._index >= len(self._parent):
      raise ValueError('UnknownField does not exist. '
                       'The parent message might be cleared.')
  @property
  def field_number(self):
    self._check_valid()
    return self._parent._internal_get(self._index)._field_number
  @property
  def wire_type(self):
    self._check_valid()
    return self._parent._internal_get(self._index)._wire_type
  @property
  def data(self):
    self._check_valid()
    return self._parent._internal_get(self._index)._data
class UnknownFieldSet(object):
  __slots__ = ['_values']
  def __init__(self):
    self._values = []
  def __getitem__(self, index):
    if self._values is None:
      raise ValueError('UnknownFields does not exist. '
                       'The parent message might be cleared.')
    size = len(self._values)
    if index < 0:
      index += size
    if index < 0 or index >= size:
      raise IndexError('index %d out of range'.index)
    return UnknownFieldRef(self, index)
  def _internal_get(self, index):
    return self._values[index]
  def __len__(self):
    if self._values is None:
      raise ValueError('UnknownFields does not exist. '
                       'The parent message might be cleared.')
    return len(self._values)
  def _add(self, field_number, wire_type, data):
    unknown_field = _UnknownField(field_number, wire_type, data)
    self._values.append(unknown_field)
    return unknown_field
  def __iter__(self):
    for i in range(len(self)):
      yield UnknownFieldRef(self, i)
  def _extend(self, other):
    if other is None:
      return
    self._values.extend(other._values)
  def __eq__(self, other):
    if self is other:
      return True
    values = list(self._values)
    if other is None:
      return not values
    values.sort()
    other_values = sorted(other._values)
    return values == other_values
  def _clear(self):
    for value in self._values:
      if isinstance(value._data, UnknownFieldSet):
        value._data._clear()
    self._values = None
