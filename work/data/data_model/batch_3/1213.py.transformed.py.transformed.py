
from bson.py3compat import integer_types, string_type
from pymongo.errors import ConfigurationError
class WriteConcern(object):
    __slots__ = ("__document", "__acknowledged")
    def __init__(self, w=None, wtimeout=None, j=None, fsync=None):
        self.__document = {}
        self.__acknowledged = True
        if wtimeout is not None:
            if not isinstance(wtimeout, integer_types):
                raise TypeError("wtimeout must be an integer")
            self.__document["wtimeout"] = wtimeout
        if j is not None:
            if not isinstance(j, bool):
                raise TypeError("j must be True or False")
            self.__document["j"] = j
        if fsync is not None:
            if not isinstance(fsync, bool):
                raise TypeError("fsync must be True or False")
            if j and fsync:
                raise ConfigurationError("Can't set both j "
                                         "and fsync at the same time")
            self.__document["fsync"] = fsync
        if self.__document and w == 0:
            raise ConfigurationError("Can not use w value "
                                     "of 0 with other options")
        if w is not None:
            if isinstance(w, integer_types):
                self.__acknowledged = w > 0
            elif not isinstance(w, string_type):
                raise TypeError("w must be an integer or string")
            self.__document["w"] = w
    @property
    def document(self):
        return self.__document.copy()
    @property
    def acknowledged(self):
        return self.__acknowledged
    def __repr__(self):
        return ("WriteConcern(%s)" % (
            ", ".join("%s=%s" % kvt for kvt in self.document.items()),))
    def __eq__(self, other):
        return self.document == other.document
    def __ne__(self, other):
        return self.document != other.document
    def __bool__(self):
        return bool(self.document)
