
import datetime
from collections import MutableMapping, namedtuple
from bson.py3compat import string_type
from bson.binary import (ALL_UUID_REPRESENTATIONS,
                         PYTHON_LEGACY,
                         UUID_REPRESENTATION_NAMES)
_RAW_BSON_DOCUMENT_MARKER = 101
def _raw_document_class(document_class):
    marker = getattr(document_class, '_type_marker', None)
    return marker == _RAW_BSON_DOCUMENT_MARKER
_options_base = namedtuple(
    'CodecOptions',
    ('document_class', 'tz_aware', 'uuid_representation',
     'unicode_decode_error_handler', 'tzinfo'))
class CodecOptions(_options_base):
    def __new__(cls, document_class=dict,
                tz_aware=False, uuid_representation=PYTHON_LEGACY,
                unicode_decode_error_handler="strict",
                tzinfo=None):
        if not (issubclass(document_class, MutableMapping) or
                _raw_document_class(document_class)):
            raise TypeError("document_class must be dict, bson.son.SON, "
                            "bson.raw_bson.RawBSONDocument, or a "
                            "sublass of collections.MutableMapping")
        if not isinstance(tz_aware, bool):
            raise TypeError("tz_aware must be True or False")
        if uuid_representation not in ALL_UUID_REPRESENTATIONS:
            raise ValueError("uuid_representation must be a value "
                             "from bson.binary.ALL_UUID_REPRESENTATIONS")
        if not isinstance(unicode_decode_error_handler, (string_type, None)):
            raise ValueError("unicode_decode_error_handler must be a string "
                             "or None")
        if tzinfo is not None:
            if not isinstance(tzinfo, datetime.tzinfo):
                raise TypeError(
                    "tzinfo must be an instance of datetime.tzinfo")
            if not tz_aware:
                raise ValueError(
                    "cannot specify tzinfo without also setting tz_aware=True")
        return tuple.__new__(
            cls, (document_class, tz_aware, uuid_representation,
                  unicode_decode_error_handler, tzinfo))
    def _arguments_repr(self):
        document_class_repr = (
            'dict' if self.document_class is dict
            else repr(self.document_class))
        uuid_rep_repr = UUID_REPRESENTATION_NAMES.get(self.uuid_representation,
                                                      self.uuid_representation)
        return ('document_class=%s, tz_aware=%r, uuid_representation='
                '%s, unicode_decode_error_handler=%r, tzinfo=%r' %
                (document_class_repr, self.tz_aware, uuid_rep_repr,
                 self.unicode_decode_error_handler, self.tzinfo))
    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._arguments_repr())
    def with_options(self, **kwargs):
        return CodecOptions(
            kwargs.get('document_class', self.document_class),
            kwargs.get('tz_aware', self.tz_aware),
            kwargs.get('uuid_representation', self.uuid_representation),
            kwargs.get('unicode_decode_error_handler',
                       self.unicode_decode_error_handler),
            kwargs.get('tzinfo', self.tzinfo))
DEFAULT_CODEC_OPTIONS = CodecOptions()
def _parse_codec_options(options):
    return CodecOptions(
        document_class=options.get(
            'document_class', DEFAULT_CODEC_OPTIONS.document_class),
        tz_aware=options.get(
            'tz_aware', DEFAULT_CODEC_OPTIONS.tz_aware),
        uuid_representation=options.get(
            'uuidrepresentation', DEFAULT_CODEC_OPTIONS.uuid_representation),
        unicode_decode_error_handler=options.get(
            'unicode_decode_error_handler',
            DEFAULT_CODEC_OPTIONS.unicode_decode_error_handler),
        tzinfo=options.get('tzinfo', DEFAULT_CODEC_OPTIONS.tzinfo))
