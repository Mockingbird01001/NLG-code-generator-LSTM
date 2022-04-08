
import collections
from bson import _UNPACK_INT, _iterate_elements
from bson.py3compat import iteritems
from bson.codec_options import (
    CodecOptions, DEFAULT_CODEC_OPTIONS as DEFAULT, _RAW_BSON_DOCUMENT_MARKER)
from bson.errors import InvalidBSON
class RawBSONDocument(collections.Mapping):
    __slots__ = ('__raw', '__inflated_doc', '__codec_options')
    _type_marker = _RAW_BSON_DOCUMENT_MARKER
    def __init__(self, bson_bytes, codec_options=None):
        self.__raw = bson_bytes
        self.__inflated_doc = None
        if codec_options is None:
            codec_options = DEFAULT_RAW_BSON_OPTIONS
        elif codec_options.document_class is not RawBSONDocument:
            raise TypeError(
                "RawBSONDocument cannot use CodecOptions with document "
                "class %s" % (codec_options.document_class, ))
        self.__codec_options = codec_options
    @property
    def raw(self):
        return self.__raw
    def items(self):
        return iteritems(self.__inflated)
    @property
    def __inflated(self):
        if self.__inflated_doc is None:
            object_size = _UNPACK_INT(self.__raw[:4])[0] - 1
            position = 0
            self.__inflated_doc = {}
            for key, value, position in _iterate_elements(
                    self.__raw, 4, object_size, self.__codec_options):
                self.__inflated_doc[key] = value
            if position != object_size:
                raise InvalidBSON('bad object or element length')
        return self.__inflated_doc
    def __getitem__(self, item):
        return self.__inflated[item]
    def __iter__(self):
        return iter(self.__inflated)
    def __len__(self):
        return len(self.__inflated)
    def __eq__(self, other):
        if isinstance(other, RawBSONDocument):
            return self.__raw == other.raw
        return NotImplemented
    def __repr__(self):
        return ("RawBSONDocument(%r, codec_options=%r)"
                % (self.raw, self.__codec_options))
DEFAULT_RAW_BSON_OPTIONS = DEFAULT.with_options(document_class=RawBSONDocument)
