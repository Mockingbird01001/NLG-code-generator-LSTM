
import collections
from bson.dbref import DBRef
from bson.objectid import ObjectId
from bson.son import SON
class SONManipulator(object):
    def will_copy(self):
        return False
    def transform_incoming(self, son, collection):
        if self.will_copy():
            return SON(son)
        return son
    def transform_outgoing(self, son, collection):
        if self.will_copy():
            return SON(son)
        return son
class ObjectIdInjector(SONManipulator):
    def transform_incoming(self, son, collection):
        if not "_id" in son:
            son["_id"] = ObjectId()
        return son
class ObjectIdShuffler(SONManipulator):
    def will_copy(self):
        return True
    def transform_incoming(self, son, collection):
        if not "_id" in son:
            return son
        transformed = SON({"_id": son["_id"]})
        transformed.update(son)
        return transformed
class NamespaceInjector(SONManipulator):
    def transform_incoming(self, son, collection):
        son["_ns"] = collection.name
        return son
class AutoReference(SONManipulator):
    def __init__(self, db):
        self.database = db
    def will_copy(self):
        return True
    def transform_incoming(self, son, collection):
        def transform_value(value):
            if isinstance(value, collections.MutableMapping):
                if "_id" in value and "_ns" in value:
                    return DBRef(value["_ns"], transform_value(value["_id"]))
                else:
                    return transform_dict(SON(value))
            elif isinstance(value, list):
                return [transform_value(v) for v in value]
            return value
        def transform_dict(object):
            for (key, value) in object.items():
                object[key] = transform_value(value)
            return object
        return transform_dict(SON(son))
    def transform_outgoing(self, son, collection):
        def transform_value(value):
            if isinstance(value, DBRef):
                return self.database.dereference(value)
            elif isinstance(value, list):
                return [transform_value(v) for v in value]
            elif isinstance(value, collections.MutableMapping):
                return transform_dict(SON(value))
            return value
        def transform_dict(object):
            for (key, value) in object.items():
                object[key] = transform_value(value)
            return object
        return transform_dict(SON(son))
