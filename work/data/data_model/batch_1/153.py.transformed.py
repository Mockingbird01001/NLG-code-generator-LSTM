
from copy import deepcopy
from bson.py3compat import iteritems, string_type
from bson.son import SON
class DBRef(object):
    _type_marker = 100
    def __init__(self, collection, id, database=None, _extra={}, **kwargs):
        if not isinstance(collection, string_type):
            raise TypeError("collection must be an "
                            "instance of %s" % string_type.__name__)
        if database is not None and not isinstance(database, string_type):
            raise TypeError("database must be an "
                            "instance of %s" % string_type.__name__)
        self.__collection = collection
        self.__id = id
        self.__database = database
        kwargs.update(_extra)
        self.__kwargs = kwargs
    @property
    def collection(self):
        return self.__collection
    @property
    def id(self):
        return self.__id
    @property
    def database(self):
        return self.__database
    def __getattr__(self, key):
        try:
            return self.__kwargs[key]
        except KeyError:
            raise AttributeError(key)
    def __setstate__(self, state):
        self.__dict__.update(state)
    def as_doc(self):
        doc = SON([("$ref", self.collection),
                   ("$id", self.id)])
        if self.database is not None:
            doc["$db"] = self.database
        doc.update(self.__kwargs)
        return doc
    def __repr__(self):
        extra = "".join([", %s=%r" % (k, v)
                         for k, v in iteritems(self.__kwargs)])
        if self.database is None:
            return "DBRef(%r, %r%s)" % (self.collection, self.id, extra)
        return "DBRef(%r, %r, %r%s)" % (self.collection, self.id,
                                        self.database, extra)
    def __eq__(self, other):
        if isinstance(other, DBRef):
            us = (self.__database, self.__collection,
                  self.__id, self.__kwargs)
            them = (other.__database, other.__collection,
                    other.__id, other.__kwargs)
            return us == them
        return NotImplemented
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash((self.__collection, self.__id, self.__database,
                     tuple(sorted(self.__kwargs.items()))))
    def __deepcopy__(self, memo):
        return DBRef(deepcopy(self.__collection, memo),
                     deepcopy(self.__id, memo),
                     deepcopy(self.__database, memo),
                     deepcopy(self.__kwargs, memo))
