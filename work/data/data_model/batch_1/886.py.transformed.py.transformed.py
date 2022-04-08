
from bson.py3compat import string_type
class ReadConcern(object):
    def __init__(self, level=None):
        if level is None or isinstance(level, string_type):
            self.__level = level
        else:
            raise TypeError(
                'level must be a string or None.')
    @property
    def level(self):
        return self.__level
    @property
    def ok_for_legacy(self):
        return self.level is None or self.level == 'local'
    @property
    def document(self):
        doc = {}
        if self.__level:
            doc['level'] = self.level
        return doc
    def __eq__(self, other):
        if isinstance(other, ReadConcern):
            return self.document == other.document
        return NotImplemented
    def __repr__(self):
        if self.level:
            return 'ReadConcern(%s)' % self.level
        return 'ReadConcern()'
DEFAULT_READ_CONCERN = ReadConcern()
