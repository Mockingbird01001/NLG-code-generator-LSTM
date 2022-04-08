
import warnings
from pymongo import mongo_client
class MongoReplicaSetClient(mongo_client.MongoClient):
    def __init__(self, *args, **kwargs):
        warnings.warn('MongoReplicaSetClient is deprecated, use MongoClient'
                      ' to connect to a replica set',
                      DeprecationWarning, stacklevel=2)
        super(MongoReplicaSetClient, self).__init__(*args, **kwargs)
    def __repr__(self):
        return "MongoReplicaSetClient(%s)" % (self._repr_helper(),)
