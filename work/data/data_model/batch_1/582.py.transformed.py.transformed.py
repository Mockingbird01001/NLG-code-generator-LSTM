from bson import json_util
from flask.json import JSONEncoder
from mongoengine.base import BaseDocument
from mongoengine.queryset import QuerySet
def _make_encoder(superclass):
    class MongoEngineJSONEncoder(superclass):
        def default(self, obj):
            if isinstance(obj, BaseDocument):
                return json_util._json_convert(obj.to_mongo())
            elif isinstance(obj, QuerySet):
                return json_util._json_convert(obj.as_pymongo())
            return superclass.default(self, obj)
    return MongoEngineJSONEncoder
MongoEngineJSONEncoder = _make_encoder(JSONEncoder)
def override_json_encoder(app):
    app.json_encoder = _make_encoder(app.json_encoder)
