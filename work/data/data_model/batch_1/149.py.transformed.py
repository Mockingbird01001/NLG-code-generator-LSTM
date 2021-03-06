import itertools
import weakref
import six
from mongoengine.common import _import_class
from mongoengine.errors import DoesNotExist, MultipleObjectsReturned
__all__ = ('BaseDict', 'BaseList', 'EmbeddedDocumentList')
class BaseDict(dict):
    _dereferenced = False
    _instance = None
    _name = None
    def __init__(self, dict_items, instance, name):
        Document = _import_class('Document')
        EmbeddedDocument = _import_class('EmbeddedDocument')
        if isinstance(instance, (Document, EmbeddedDocument)):
            self._instance = weakref.proxy(instance)
        self._name = name
        super(BaseDict, self).__init__(dict_items)
    def __getitem__(self, key, *args, **kwargs):
        value = super(BaseDict, self).__getitem__(key)
        EmbeddedDocument = _import_class('EmbeddedDocument')
        if isinstance(value, EmbeddedDocument) and value._instance is None:
            value._instance = self._instance
        elif not isinstance(value, BaseDict) and isinstance(value, dict):
            value = BaseDict(value, None, '%s.%s' % (self._name, key))
            super(BaseDict, self).__setitem__(key, value)
            value._instance = self._instance
        elif not isinstance(value, BaseList) and isinstance(value, list):
            value = BaseList(value, None, '%s.%s' % (self._name, key))
            super(BaseDict, self).__setitem__(key, value)
            value._instance = self._instance
        return value
    def __setitem__(self, key, value, *args, **kwargs):
        self._mark_as_changed(key)
        return super(BaseDict, self).__setitem__(key, value)
    def __delete__(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseDict, self).__delete__(*args, **kwargs)
    def __delitem__(self, key, *args, **kwargs):
        self._mark_as_changed(key)
        return super(BaseDict, self).__delitem__(key)
    def __delattr__(self, key, *args, **kwargs):
        self._mark_as_changed(key)
        return super(BaseDict, self).__delattr__(key)
    def __getstate__(self):
        self.instance = None
        self._dereferenced = False
        return self
    def __setstate__(self, state):
        self = state
        return self
    def clear(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseDict, self).clear()
    def pop(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseDict, self).pop(*args, **kwargs)
    def popitem(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseDict, self).popitem()
    def setdefault(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseDict, self).setdefault(*args, **kwargs)
    def update(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseDict, self).update(*args, **kwargs)
    def _mark_as_changed(self, key=None):
        if hasattr(self._instance, '_mark_as_changed'):
            if key:
                self._instance._mark_as_changed('%s.%s' % (self._name, key))
            else:
                self._instance._mark_as_changed(self._name)
class BaseList(list):
    _dereferenced = False
    _instance = None
    _name = None
    def __init__(self, list_items, instance, name):
        Document = _import_class('Document')
        EmbeddedDocument = _import_class('EmbeddedDocument')
        if isinstance(instance, (Document, EmbeddedDocument)):
            self._instance = weakref.proxy(instance)
        self._name = name
        super(BaseList, self).__init__(list_items)
    def __getitem__(self, key, *args, **kwargs):
        value = super(BaseList, self).__getitem__(key)
        EmbeddedDocument = _import_class('EmbeddedDocument')
        if isinstance(value, EmbeddedDocument) and value._instance is None:
            value._instance = self._instance
        elif not isinstance(value, BaseDict) and isinstance(value, dict):
            value = BaseDict(value, None, '%s.%s' % (self._name, key))
            super(BaseList, self).__setitem__(key, value)
            value._instance = self._instance
        elif not isinstance(value, BaseList) and isinstance(value, list):
            value = BaseList(value, None, '%s.%s' % (self._name, key))
            super(BaseList, self).__setitem__(key, value)
            value._instance = self._instance
        return value
    def __iter__(self):
        for i in six.moves.range(self.__len__()):
            yield self[i]
    def __setitem__(self, key, value, *args, **kwargs):
        if isinstance(key, slice):
            self._mark_as_changed()
        else:
            self._mark_as_changed(key)
        return super(BaseList, self).__setitem__(key, value)
    def __delitem__(self, key, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).__delitem__(key)
    def __setslice__(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).__setslice__(*args, **kwargs)
    def __delslice__(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).__delslice__(*args, **kwargs)
    def __getstate__(self):
        self.instance = None
        self._dereferenced = False
        return self
    def __setstate__(self, state):
        self = state
        return self
    def __iadd__(self, other):
        self._mark_as_changed()
        return super(BaseList, self).__iadd__(other)
    def __imul__(self, other):
        self._mark_as_changed()
        return super(BaseList, self).__imul__(other)
    def append(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).append(*args, **kwargs)
    def extend(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).extend(*args, **kwargs)
    def insert(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).insert(*args, **kwargs)
    def pop(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).pop(*args, **kwargs)
    def remove(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).remove(*args, **kwargs)
    def reverse(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).reverse()
    def sort(self, *args, **kwargs):
        self._mark_as_changed()
        return super(BaseList, self).sort(*args, **kwargs)
    def _mark_as_changed(self, key=None):
        if hasattr(self._instance, '_mark_as_changed'):
            if key:
                self._instance._mark_as_changed(
                    '%s.%s' % (self._name, key % len(self))
                )
            else:
                self._instance._mark_as_changed(self._name)
class EmbeddedDocumentList(BaseList):
    @classmethod
    def __match_all(cls, embedded_doc, kwargs):
        for key, expected_value in kwargs.items():
            doc_val = getattr(embedded_doc, key)
            if doc_val != expected_value and six.text_type(doc_val) != expected_value:
                return False
        return True
    @classmethod
    def __only_matches(cls, embedded_docs, kwargs):
        if not kwargs:
            return embedded_docs
        return [doc for doc in embedded_docs if cls.__match_all(doc, kwargs)]
    def __init__(self, list_items, instance, name):
        super(EmbeddedDocumentList, self).__init__(list_items, instance, name)
        self._instance = instance
    def filter(self, **kwargs):
        values = self.__only_matches(self, kwargs)
        return EmbeddedDocumentList(values, self._instance, self._name)
    def exclude(self, **kwargs):
        exclude = self.__only_matches(self, kwargs)
        values = [item for item in self if item not in exclude]
        return EmbeddedDocumentList(values, self._instance, self._name)
    def count(self):
        return len(self)
    def get(self, **kwargs):
        values = self.__only_matches(self, kwargs)
        if len(values) == 0:
            raise DoesNotExist(
                '%s matching query does not exist.' % self._name
            )
        elif len(values) > 1:
            raise MultipleObjectsReturned(
                '%d items returned, instead of 1' % len(values)
            )
        return values[0]
    def first(self):
        if len(self) > 0:
            return self[0]
    def create(self, **values):
        name = self._name
        EmbeddedClass = self._instance._fields[name].field.document_type_obj
        self._instance[self._name].append(EmbeddedClass(**values))
        return self._instance[self._name][-1]
    def save(self, *args, **kwargs):
        self._instance.save(*args, **kwargs)
    def delete(self):
        values = list(self)
        for item in values:
            self._instance[self._name].remove(item)
        return len(values)
    def update(self, **update):
        if len(update) == 0:
            return 0
        values = list(self)
        for item in values:
            for k, v in update.items():
                setattr(item, k, v)
        return len(values)
class StrictDict(object):
    __slots__ = ()
    _special_fields = set(['get', 'pop', 'iteritems', 'items', 'keys', 'create'])
    _classes = {}
    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
    def __getitem__(self, key):
        key = '_reserved_' + key if key in self._special_fields else key
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
    def __setitem__(self, key, value):
        key = '_reserved_' + key if key in self._special_fields else key
        return setattr(self, key, value)
    def __contains__(self, key):
        return hasattr(self, key)
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    def pop(self, key, default=None):
        v = self.get(key, default)
        try:
            delattr(self, key)
        except AttributeError:
            pass
        return v
    def iteritems(self):
        for key in self:
            yield key, self[key]
    def items(self):
        return [(k, self[k]) for k in iter(self)]
    def iterkeys(self):
        return iter(self)
    def keys(self):
        return list(iter(self))
    def __iter__(self):
        return (key for key in self.__slots__ if hasattr(self, key))
    def __len__(self):
        return len(list(self.iteritems()))
    def __eq__(self, other):
        return self.items() == other.items()
    def __ne__(self, other):
        return self.items() != other.items()
    @classmethod
    def create(cls, allowed_keys):
        allowed_keys_tuple = tuple(('_reserved_' + k if k in cls._special_fields else k) for k in allowed_keys)
        allowed_keys = frozenset(allowed_keys_tuple)
        if allowed_keys not in cls._classes:
            class SpecificStrictDict(cls):
                __slots__ = allowed_keys_tuple
                def __repr__(self):
                    return '{%s}' % ', '.join('"{0!s}": {1!r}'.format(k, v) for k, v in self.items())
            cls._classes[allowed_keys] = SpecificStrictDict
        return cls._classes[allowed_keys]
