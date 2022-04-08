from mongoengine.common import _import_class
from mongoengine.connection import DEFAULT_CONNECTION_NAME, get_db
__all__ = ('switch_db', 'switch_collection', 'no_dereference',
           'no_sub_classes', 'query_counter')
class switch_db(object):
    def __init__(self, cls, db_alias):
        self.cls = cls
        self.collection = cls._get_collection()
        self.db_alias = db_alias
        self.ori_db_alias = cls._meta.get('db_alias', DEFAULT_CONNECTION_NAME)
    def __enter__(self):
        self.cls._meta['db_alias'] = self.db_alias
        self.cls._collection = None
        return self.cls
    def __exit__(self, t, value, traceback):
        self.cls._meta['db_alias'] = self.ori_db_alias
        self.cls._collection = self.collection
class switch_collection(object):
    def __init__(self, cls, collection_name):
        self.cls = cls
        self.ori_collection = cls._get_collection()
        self.ori_get_collection_name = cls._get_collection_name
        self.collection_name = collection_name
    def __enter__(self):
        @classmethod
        def _get_collection_name(cls):
            return self.collection_name
        self.cls._get_collection_name = _get_collection_name
        self.cls._collection = None
        return self.cls
    def __exit__(self, t, value, traceback):
        self.cls._collection = self.ori_collection
        self.cls._get_collection_name = self.ori_get_collection_name
class no_dereference(object):
    def __init__(self, cls):
        self.cls = cls
        ReferenceField = _import_class('ReferenceField')
        GenericReferenceField = _import_class('GenericReferenceField')
        ComplexBaseField = _import_class('ComplexBaseField')
        self.deref_fields = [k for k, v in self.cls._fields.iteritems()
                             if isinstance(v, (ReferenceField,
                                               GenericReferenceField,
                                               ComplexBaseField))]
    def __enter__(self):
        for field in self.deref_fields:
            self.cls._fields[field]._auto_dereference = False
        return self.cls
    def __exit__(self, t, value, traceback):
        for field in self.deref_fields:
            self.cls._fields[field]._auto_dereference = True
        return self.cls
class no_sub_classes(object):
    def __init__(self, cls):
        self.cls = cls
    def __enter__(self):
        self.cls._all_subclasses = self.cls._subclasses
        self.cls._subclasses = (self.cls,)
        return self.cls
    def __exit__(self, t, value, traceback):
        self.cls._subclasses = self.cls._all_subclasses
        delattr(self.cls, '_all_subclasses')
        return self.cls
class query_counter(object):
    def __init__(self):
        self.counter = 0
        self.db = get_db()
    def __enter__(self):
        self.db.set_profiling_level(0)
        self.db.system.profile.drop()
        self.db.set_profiling_level(2)
        return self
    def __exit__(self, t, value, traceback):
        self.db.set_profiling_level(0)
    def __eq__(self, value):
        counter = self._get_count()
        return value == counter
    def __ne__(self, value):
        return not self.__eq__(value)
    def __lt__(self, value):
        return self._get_count() < value
    def __le__(self, value):
        return self._get_count() <= value
    def __gt__(self, value):
        return self._get_count() > value
    def __ge__(self, value):
        return self._get_count() >= value
    def __int__(self):
        return self._get_count()
    def __repr__(self):
        return u"%s" % self._get_count()
    def _get_count(self):
        ignore_query = {'ns': {'$ne': '%s.system.indexes' % self.db.name}}
        count = self.db.system.profile.find(ignore_query).count() - self.counter
        self.counter += 1
        return count
