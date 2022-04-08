
"""
    itsdangerous
    ~~~~~~~~~~~~
    A module that implements various functions to deal with untrusted
    sources.  Mainly useful for web applications.
    :copyright: (c) 2014 by Armin Ronacher and the Django Software Foundation.
    :license: BSD, see LICENSE for more details.
"""
import sys
import hmac
import zlib
import time
import base64
import hashlib
import operator
from datetime import datetime
PY2 = sys.version_info[0] == 2
if PY2:
    from itertools import izip
    text_type = unicode
    int_to_byte = chr
    number_types = (int, long, float)
else:
    from functools import reduce
    izip = zip
    text_type = str
    int_to_byte = operator.methodcaller('to_bytes', 1, 'big')
    number_types = (int, float)
try:
    import simplejson as json
except ImportError:
    import json
class _CompactJSON(object):
    def loads(self, payload):
        return json.loads(payload)
    def dumps(self, obj):
        return json.dumps(obj, separators=(',', ':'))
compact_json = _CompactJSON()
EPOCH = 1293840000
def want_bytes(s, encoding='utf-8', errors='strict'):
    if isinstance(s, text_type):
        s = s.encode(encoding, errors)
    return s
def is_text_serializer(serializer):
    return isinstance(serializer.dumps({}), text_type)
_builtin_constant_time_compare = getattr(hmac, 'compare_digest', None)
def constant_time_compare(val1, val2):
    if _builtin_constant_time_compare is not None:
        return _builtin_constant_time_compare(val1, val2)
    len_eq = len(val1) == len(val2)
    if len_eq:
        result = 0
        left = val1
    else:
        result = 1
        left = val2
    for x, y in izip(bytearray(left), bytearray(val2)):
        result |= x ^ y
    return result == 0
class BadData(Exception):
    message = None
    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message
    def __str__(self):
        return text_type(self.message)
    if PY2:
        __unicode__ = __str__
        def __str__(self):
            return self.__unicode__().encode('utf-8')
class BadPayload(BadData):
    def __init__(self, message, original_error=None):
        BadData.__init__(self, message)
        self.original_error = original_error
class BadSignature(BadData):
    def __init__(self, message, payload=None):
        BadData.__init__(self, message)
        self.payload = payload
class BadTimeSignature(BadSignature):
    def __init__(self, message, payload=None, date_signed=None):
        BadSignature.__init__(self, message, payload)
        self.date_signed = date_signed
class BadHeader(BadSignature):
    def __init__(self, message, payload=None, header=None,
                 original_error=None):
        BadSignature.__init__(self, message, payload)
        self.header = header
        self.original_error = original_error
class SignatureExpired(BadTimeSignature):
def base64_encode(string):
    string = want_bytes(string)
    return base64.urlsafe_b64encode(string).strip(b'=')
def base64_decode(string):
    string = want_bytes(string, encoding='ascii', errors='ignore')
    return base64.urlsafe_b64decode(string + b'=' * (-len(string) % 4))
def int_to_bytes(num):
    assert num >= 0
    rv = []
    while num:
        rv.append(int_to_byte(num & 0xff))
        num >>= 8
    return b''.join(reversed(rv))
def bytes_to_int(bytestr):
    return reduce(lambda a, b: a << 8 | b, bytearray(bytestr), 0)
class SigningAlgorithm(object):
    def get_signature(self, key, value):
        raise NotImplementedError()
    def verify_signature(self, key, value, sig):
        return constant_time_compare(sig, self.get_signature(key, value))
class NoneAlgorithm(SigningAlgorithm):
    def get_signature(self, key, value):
        return b''
class HMACAlgorithm(SigningAlgorithm):
    default_digest_method = staticmethod(hashlib.sha1)
    def __init__(self, digest_method=None):
        if digest_method is None:
            digest_method = self.default_digest_method
        self.digest_method = digest_method
    def get_signature(self, key, value):
        mac = hmac.new(key, msg=value, digestmod=self.digest_method)
        return mac.digest()
class Signer(object):
    default_digest_method = staticmethod(hashlib.sha1)
    default_key_derivation = 'django-concat'
    def __init__(self, secret_key, salt=None, sep='.', key_derivation=None,
                 digest_method=None, algorithm=None):
        self.secret_key = want_bytes(secret_key)
        self.sep = sep
        self.salt = 'itsdangerous.Signer' if salt is None else salt
        if key_derivation is None:
            key_derivation = self.default_key_derivation
        self.key_derivation = key_derivation
        if digest_method is None:
            digest_method = self.default_digest_method
        self.digest_method = digest_method
        if algorithm is None:
            algorithm = HMACAlgorithm(self.digest_method)
        self.algorithm = algorithm
    def derive_key(self):
        salt = want_bytes(self.salt)
        if self.key_derivation == 'concat':
            return self.digest_method(salt + self.secret_key).digest()
        elif self.key_derivation == 'django-concat':
            return self.digest_method(salt + b'signer' +
                self.secret_key).digest()
        elif self.key_derivation == 'hmac':
            mac = hmac.new(self.secret_key, digestmod=self.digest_method)
            mac.update(salt)
            return mac.digest()
        elif self.key_derivation == 'none':
            return self.secret_key
        else:
            raise TypeError('Unknown key derivation method')
    def get_signature(self, value):
        value = want_bytes(value)
        key = self.derive_key()
        sig = self.algorithm.get_signature(key, value)
        return base64_encode(sig)
    def sign(self, value):
        return value + want_bytes(self.sep) + self.get_signature(value)
    def verify_signature(self, value, sig):
        key = self.derive_key()
        try:
            sig = base64_decode(sig)
        except Exception:
            return False
        return self.algorithm.verify_signature(key, value, sig)
    def unsign(self, signed_value):
        signed_value = want_bytes(signed_value)
        sep = want_bytes(self.sep)
        if sep not in signed_value:
            raise BadSignature('No %r found in value' % self.sep)
        value, sig = signed_value.rsplit(sep, 1)
        if self.verify_signature(value, sig):
            return value
        raise BadSignature('Signature %r does not match' % sig,
                           payload=value)
    def validate(self, signed_value):
        try:
            self.unsign(signed_value)
            return True
        except BadSignature:
            return False
class TimestampSigner(Signer):
    def get_timestamp(self):
        return int(time.time() - EPOCH)
    def timestamp_to_datetime(self, ts):
        return datetime.utcfromtimestamp(ts + EPOCH)
    def sign(self, value):
        value = want_bytes(value)
        timestamp = base64_encode(int_to_bytes(self.get_timestamp()))
        sep = want_bytes(self.sep)
        value = value + sep + timestamp
        return value + sep + self.get_signature(value)
    def unsign(self, value, max_age=None, return_timestamp=False):
        try:
            result = Signer.unsign(self, value)
            sig_error = None
        except BadSignature as e:
            sig_error = e
            result = e.payload or b''
        sep = want_bytes(self.sep)
        if not sep in result:
            if sig_error:
                raise sig_error
            raise BadTimeSignature('timestamp missing', payload=result)
        value, timestamp = result.rsplit(sep, 1)
        try:
            timestamp = bytes_to_int(base64_decode(timestamp))
        except Exception:
            timestamp = None
        if sig_error is not None:
            raise BadTimeSignature(text_type(sig_error), payload=value,
                                   date_signed=timestamp)
        if timestamp is None:
            raise BadTimeSignature('Malformed timestamp', payload=value)
        if max_age is not None:
            age = self.get_timestamp() - timestamp
            if age > max_age:
                raise SignatureExpired(
                    'Signature age %s > %s seconds' % (age, max_age),
                    payload=value,
                    date_signed=self.timestamp_to_datetime(timestamp))
        if return_timestamp:
            return value, self.timestamp_to_datetime(timestamp)
        return value
    def validate(self, signed_value, max_age=None):
        try:
            self.unsign(signed_value, max_age=max_age)
            return True
        except BadSignature:
            return False
class Serializer(object):
    default_serializer = json
    default_signer = Signer
    def __init__(self, secret_key, salt=b'itsdangerous', serializer=None,
                 signer=None, signer_kwargs=None):
        self.secret_key = want_bytes(secret_key)
        self.salt = want_bytes(salt)
        if serializer is None:
            serializer = self.default_serializer
        self.serializer = serializer
        self.is_text_serializer = is_text_serializer(serializer)
        if signer is None:
            signer = self.default_signer
        self.signer = signer
        self.signer_kwargs = signer_kwargs or {}
    def load_payload(self, payload, serializer=None):
        if serializer is None:
            serializer = self.serializer
            is_text = self.is_text_serializer
        else:
            is_text = is_text_serializer(serializer)
        try:
            if is_text:
                payload = payload.decode('utf-8')
            return serializer.loads(payload)
        except Exception as e:
            raise BadPayload('Could not load the payload because an '
                'exception occurred on unserializing the data',
                original_error=e)
    def dump_payload(self, obj):
        return want_bytes(self.serializer.dumps(obj))
    def make_signer(self, salt=None):
        if salt is None:
            salt = self.salt
        return self.signer(self.secret_key, salt=salt, **self.signer_kwargs)
    def dumps(self, obj, salt=None):
        payload = want_bytes(self.dump_payload(obj))
        rv = self.make_signer(salt).sign(payload)
        if self.is_text_serializer:
            rv = rv.decode('utf-8')
        return rv
    def dump(self, obj, f, salt=None):
        f.write(self.dumps(obj, salt))
    def loads(self, s, salt=None):
        s = want_bytes(s)
        return self.load_payload(self.make_signer(salt).unsign(s))
    def load(self, f, salt=None):
        return self.loads(f.read(), salt)
    def loads_unsafe(self, s, salt=None):
        return self._loads_unsafe_impl(s, salt)
    def _loads_unsafe_impl(self, s, salt, load_kwargs=None,
                           load_payload_kwargs=None):
        try:
            return True, self.loads(s, salt=salt, **(load_kwargs or {}))
        except BadSignature as e:
            if e.payload is None:
                return False, None
            try:
                return False, self.load_payload(e.payload,
                    **(load_payload_kwargs or {}))
            except BadPayload:
                return False, None
    def load_unsafe(self, f, *args, **kwargs):
        return self.loads_unsafe(f.read(), *args, **kwargs)
class TimedSerializer(Serializer):
    default_signer = TimestampSigner
    def loads(self, s, max_age=None, return_timestamp=False, salt=None):
        base64d, timestamp = self.make_signer(salt)            .unsign(s, max_age, return_timestamp=True)
        payload = self.load_payload(base64d)
        if return_timestamp:
            return payload, timestamp
        return payload
    def loads_unsafe(self, s, max_age=None, salt=None):
        load_kwargs = {'max_age': max_age}
        load_payload_kwargs = {}
        return self._loads_unsafe_impl(s, salt, load_kwargs, load_payload_kwargs)
class JSONWebSignatureSerializer(Serializer):
    jws_algorithms = {
        'HS256': HMACAlgorithm(hashlib.sha256),
        'HS384': HMACAlgorithm(hashlib.sha384),
        'HS512': HMACAlgorithm(hashlib.sha512),
        'none': NoneAlgorithm(),
    }
    default_algorithm = 'HS256'
    default_serializer = compact_json
    def __init__(self, secret_key, salt=None, serializer=None,
                 signer=None, signer_kwargs=None, algorithm_name=None):
        Serializer.__init__(self, secret_key, salt, serializer,
                            signer, signer_kwargs)
        if algorithm_name is None:
            algorithm_name = self.default_algorithm
        self.algorithm_name = algorithm_name
        self.algorithm = self.make_algorithm(algorithm_name)
    def load_payload(self, payload, return_header=False):
        payload = want_bytes(payload)
        if b'.' not in payload:
            raise BadPayload('No "." found in value')
        base64d_header, base64d_payload = payload.split(b'.', 1)
        try:
            json_header = base64_decode(base64d_header)
        except Exception as e:
            raise BadHeader('Could not base64 decode the header because of '
                'an exception', original_error=e)
        try:
            json_payload = base64_decode(base64d_payload)
        except Exception as e:
            raise BadPayload('Could not base64 decode the payload because of '
                'an exception', original_error=e)
        try:
            header = Serializer.load_payload(self, json_header,
                serializer=json)
        except BadData as e:
            raise BadHeader('Could not unserialize header because it was '
                'malformed', original_error=e)
        if not isinstance(header, dict):
            raise BadHeader('Header payload is not a JSON object',
                header=header)
        payload = Serializer.load_payload(self, json_payload)
        if return_header:
            return payload, header
        return payload
    def dump_payload(self, header, obj):
        base64d_header = base64_encode(self.serializer.dumps(header))
        base64d_payload = base64_encode(self.serializer.dumps(obj))
        return base64d_header + b'.' + base64d_payload
    def make_algorithm(self, algorithm_name):
        try:
            return self.jws_algorithms[algorithm_name]
        except KeyError:
            raise NotImplementedError('Algorithm not supported')
    def make_signer(self, salt=None, algorithm=None):
        if salt is None:
            salt = self.salt
        key_derivation = 'none' if salt is None else None
        if algorithm is None:
            algorithm = self.algorithm
        return self.signer(self.secret_key, salt=salt, sep='.',
            key_derivation=key_derivation, algorithm=algorithm)
    def make_header(self, header_fields):
        header = header_fields.copy() if header_fields else {}
        header['alg'] = self.algorithm_name
        return header
    def dumps(self, obj, salt=None, header_fields=None):
        header = self.make_header(header_fields)
        signer = self.make_signer(salt, self.algorithm)
        return signer.sign(self.dump_payload(header, obj))
    def loads(self, s, salt=None, return_header=False):
        payload, header = self.load_payload(
            self.make_signer(salt, self.algorithm).unsign(want_bytes(s)),
            return_header=True)
        if header.get('alg') != self.algorithm_name:
            raise BadHeader('Algorithm mismatch', header=header,
                            payload=payload)
        if return_header:
            return payload, header
        return payload
    def loads_unsafe(self, s, salt=None, return_header=False):
        kwargs = {'return_header': return_header}
        return self._loads_unsafe_impl(s, salt, kwargs, kwargs)
class TimedJSONWebSignatureSerializer(JSONWebSignatureSerializer):
    DEFAULT_EXPIRES_IN = 3600
    def __init__(self, secret_key, expires_in=None, **kwargs):
        JSONWebSignatureSerializer.__init__(self, secret_key, **kwargs)
        if expires_in is None:
            expires_in = self.DEFAULT_EXPIRES_IN
        self.expires_in = expires_in
    def make_header(self, header_fields):
        header = JSONWebSignatureSerializer.make_header(self, header_fields)
        iat = self.now()
        exp = iat + self.expires_in
        header['iat'] = iat
        header['exp'] = exp
        return header
    def loads(self, s, salt=None, return_header=False):
        payload, header = JSONWebSignatureSerializer.loads(
            self, s, salt, return_header=True)
        if 'exp' not in header:
            raise BadSignature('Missing expiry date', payload=payload)
        if not (isinstance(header['exp'], number_types)
                and header['exp'] > 0):
            raise BadSignature('expiry date is not an IntDate',
                               payload=payload)
        if header['exp'] < self.now():
            raise SignatureExpired('Signature expired', payload=payload,
                                   date_signed=self.get_issue_date(header))
        if return_header:
            return payload, header
        return payload
    def get_issue_date(self, header):
        rv = header.get('iat')
        if isinstance(rv, number_types):
            return datetime.utcfromtimestamp(int(rv))
    def now(self):
        return int(time.time())
class URLSafeSerializerMixin(object):
    def load_payload(self, payload):
        decompress = False
        if payload.startswith(b'.'):
            payload = payload[1:]
            decompress = True
        try:
            json = base64_decode(payload)
        except Exception as e:
            raise BadPayload('Could not base64 decode the payload because of '
                'an exception', original_error=e)
        if decompress:
            try:
                json = zlib.decompress(json)
            except Exception as e:
                raise BadPayload('Could not zlib decompress the payload before '
                    'decoding the payload', original_error=e)
        return super(URLSafeSerializerMixin, self).load_payload(json)
    def dump_payload(self, obj):
        json = super(URLSafeSerializerMixin, self).dump_payload(obj)
        is_compressed = False
        compressed = zlib.compress(json)
        if len(compressed) < (len(json) - 1):
            json = compressed
            is_compressed = True
        base64d = base64_encode(json)
        if is_compressed:
            base64d = b'.' + base64d
        return base64d
class URLSafeSerializer(URLSafeSerializerMixin, Serializer):
    default_serializer = compact_json
class URLSafeTimedSerializer(URLSafeSerializerMixin, TimedSerializer):
    default_serializer = compact_json
