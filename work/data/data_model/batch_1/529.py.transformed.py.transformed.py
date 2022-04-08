
"""
    werkzeug.http
    ~~~~~~~~~~~~~
    Werkzeug comes with a bunch of utilities that help Werkzeug to deal with
    HTTP data.  Most of the classes and functions provided by this module are
    used by the wrappers, but they are useful on their own, too, especially if
    the response and request objects are not used.
    This covers some of the more HTTP centric features of WSGI, some other
    utilities such as cookie handling are documented in the `werkzeug.utils`
    module.
    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import re
from time import time, gmtime
try:
    from email.utils import parsedate_tz
except ImportError:
    from email.Utils import parsedate_tz
try:
    from urllib.request import parse_http_list as _parse_list_header
    from urllib.parse import unquote_to_bytes as _unquote
except ImportError:
    from urllib2 import parse_http_list as _parse_list_header,        unquote as _unquote
from datetime import datetime, timedelta
from hashlib import md5
import base64
from werkzeug._internal import _cookie_quote, _make_cookie_domain,    _cookie_parse_impl
from werkzeug._compat import to_unicode, iteritems, text_type,    string_types, try_coerce_native, to_bytes, PY2,    integer_types
_cookie_charset = 'latin1'
_accept_re = re.compile(
                )
        ''', re.VERBOSE)
                         '^_`abcdefghijklmnopqrstuvwxyz|~')
_etag_re = re.compile(r'([Ww]/)?(?:"(.*?)"|(.*?))(?:\s*,\s*|$)')
_unsafe_header_chars = set('()<>@,;:\"/[]?={} \t')
_quoted_string_re = r'"[^"\\]*(?:\\.[^"\\]*)*"'
_option_header_piece_re = re.compile(
    r';\s*(%s|[^\s;,=\*]+)\s*'
    r'(?:\*?=\s*(?:([^\s]+?)\'([^\s]*?)\')?(%s|[^;,]+)?)?\s*' %
    (_quoted_string_re, _quoted_string_re)
)
_option_header_start_mime_type = re.compile(r',\s*([^;,\s]+)([;,]\s*.+)?')
_entity_headers = frozenset([
    'allow', 'content-encoding', 'content-language', 'content-length',
    'content-location', 'content-md5', 'content-range', 'content-type',
    'expires', 'last-modified'
])
_hop_by_hop_headers = frozenset([
    'connection', 'keep-alive', 'proxy-authenticate',
    'proxy-authorization', 'te', 'trailer', 'transfer-encoding',
    'upgrade'
])
HTTP_STATUS_CODES = {
    100:    'Continue',
    101:    'Switching Protocols',
    102:    'Processing',
    200:    'OK',
    201:    'Created',
    202:    'Accepted',
    203:    'Non Authoritative Information',
    204:    'No Content',
    205:    'Reset Content',
    206:    'Partial Content',
    207:    'Multi Status',
    226:    'IM Used',
    300:    'Multiple Choices',
    301:    'Moved Permanently',
    302:    'Found',
    303:    'See Other',
    304:    'Not Modified',
    305:    'Use Proxy',
    307:    'Temporary Redirect',
    400:    'Bad Request',
    401:    'Unauthorized',
    402:    'Payment Required',
    403:    'Forbidden',
    404:    'Not Found',
    405:    'Method Not Allowed',
    406:    'Not Acceptable',
    407:    'Proxy Authentication Required',
    408:    'Request Timeout',
    409:    'Conflict',
    410:    'Gone',
    411:    'Length Required',
    412:    'Precondition Failed',
    413:    'Request Entity Too Large',
    414:    'Request URI Too Long',
    415:    'Unsupported Media Type',
    416:    'Requested Range Not Satisfiable',
    417:    'Expectation Failed',
    418:    'I\'m a teapot',
    422:    'Unprocessable Entity',
    423:    'Locked',
    424:    'Failed Dependency',
    426:    'Upgrade Required',
    428:    'Precondition Required',
    429:    'Too Many Requests',
    431:    'Request Header Fields Too Large',
    449:    'Retry With',
    451:    'Unavailable For Legal Reasons',
    500:    'Internal Server Error',
    501:    'Not Implemented',
    502:    'Bad Gateway',
    503:    'Service Unavailable',
    504:    'Gateway Timeout',
    505:    'HTTP Version Not Supported',
    507:    'Insufficient Storage',
    510:    'Not Extended'
}
def wsgi_to_bytes(data):
    if isinstance(data, bytes):
        return data
    return data.encode('latin1')
def bytes_to_wsgi(data):
    assert isinstance(data, bytes), 'data must be bytes'
    if isinstance(data, str):
        return data
    else:
        return data.decode('latin1')
def quote_header_value(value, extra_chars='', allow_token=True):
    if isinstance(value, bytes):
        value = bytes_to_wsgi(value)
    value = str(value)
    if allow_token:
        token_chars = _token_chars | set(extra_chars)
        if set(value).issubset(token_chars):
            return value
    return '"%s"' % value.replace('\\', '\\\\').replace('"', '\\"')
def unquote_header_value(value, is_filename=False):
    if value and value[0] == value[-1] == '"':
        value = value[1:-1]
        if not is_filename or value[:2] != '\\\\':
            return value.replace('\\\\', '\\').replace('\\"', '"')
    return value
def dump_options_header(header, options):
    segments = []
    if header is not None:
        segments.append(header)
    for key, value in iteritems(options):
        if value is None:
            segments.append(key)
        else:
            segments.append('%s=%s' % (key, quote_header_value(value)))
    return '; '.join(segments)
def dump_header(iterable, allow_token=True):
    if isinstance(iterable, dict):
        items = []
        for key, value in iteritems(iterable):
            if value is None:
                items.append(key)
            else:
                items.append('%s=%s' % (
                    key,
                    quote_header_value(value, allow_token=allow_token)
                ))
    else:
        items = [quote_header_value(x, allow_token=allow_token)
                 for x in iterable]
    return ', '.join(items)
def parse_list_header(value):
    result = []
    for item in _parse_list_header(value):
        if item[:1] == item[-1:] == '"':
            item = unquote_header_value(item[1:-1])
        result.append(item)
    return result
def parse_dict_header(value, cls=dict):
    result = cls()
    if not isinstance(value, text_type):
        value = bytes_to_wsgi(value)
    for item in _parse_list_header(value):
        if '=' not in item:
            result[item] = None
            continue
        name, value = item.split('=', 1)
        if value[:1] == value[-1:] == '"':
            value = unquote_header_value(value[1:-1])
        result[name] = value
    return result
def parse_options_header(value, multiple=False):
    if not value:
        return '', {}
    result = []
    value = "," + value.replace("\n", ",")
    while value:
        match = _option_header_start_mime_type.match(value)
        if not match:
            break
        result.append(match.group(1))
        options = {}
        rest = match.group(2)
        while rest:
            optmatch = _option_header_piece_re.match(rest)
            if not optmatch:
                break
            option, encoding, _, option_value = optmatch.groups()
            option = unquote_header_value(option)
            if option_value is not None:
                option_value = unquote_header_value(
                    option_value,
                    option == 'filename')
                if encoding is not None:
                    option_value = _unquote(option_value).decode(encoding)
            options[option] = option_value
            rest = rest[optmatch.end():]
        result.append(options)
        if multiple is False:
            return tuple(result)
        value = rest
    return tuple(result) if result else ('', {})
def parse_accept_header(value, cls=None):
    if cls is None:
        cls = Accept
    if not value:
        return cls(None)
    result = []
    for match in _accept_re.finditer(value):
        quality = match.group(2)
        if not quality:
            quality = 1
        else:
            quality = max(min(float(quality), 1), 0)
        result.append((match.group(1), quality))
    return cls(result)
def parse_cache_control_header(value, on_update=None, cls=None):
    if cls is None:
        cls = RequestCacheControl
    if not value:
        return cls(None, on_update)
    return cls(parse_dict_header(value), on_update)
def parse_set_header(value, on_update=None):
    if not value:
        return HeaderSet(None, on_update)
    return HeaderSet(parse_list_header(value), on_update)
def parse_authorization_header(value):
    if not value:
        return
    value = wsgi_to_bytes(value)
    try:
        auth_type, auth_info = value.split(None, 1)
        auth_type = auth_type.lower()
    except ValueError:
        return
    if auth_type == b'basic':
        try:
            username, password = base64.b64decode(auth_info).split(b':', 1)
        except Exception:
            return
        return Authorization('basic', {'username':  bytes_to_wsgi(username),
                                       'password': bytes_to_wsgi(password)})
    elif auth_type == b'digest':
        auth_map = parse_dict_header(auth_info)
        for key in 'username', 'realm', 'nonce', 'uri', 'response':
            if key not in auth_map:
                return
        if 'qop' in auth_map:
            if not auth_map.get('nc') or not auth_map.get('cnonce'):
                return
        return Authorization('digest', auth_map)
def parse_www_authenticate_header(value, on_update=None):
    if not value:
        return WWWAuthenticate(on_update=on_update)
    try:
        auth_type, auth_info = value.split(None, 1)
        auth_type = auth_type.lower()
    except (ValueError, AttributeError):
        return WWWAuthenticate(value.strip().lower(), on_update=on_update)
    return WWWAuthenticate(auth_type, parse_dict_header(auth_info),
                           on_update)
def parse_if_range_header(value):
    if not value:
        return IfRange()
    date = parse_date(value)
    if date is not None:
        return IfRange(date=date)
    return IfRange(unquote_etag(value)[0])
def parse_range_header(value, make_inclusive=True):
    if not value or '=' not in value:
        return None
    ranges = []
    last_end = 0
    units, rng = value.split('=', 1)
    units = units.strip().lower()
    for item in rng.split(','):
        item = item.strip()
        if '-' not in item:
            return None
        if item.startswith('-'):
            if last_end < 0:
                return None
            try:
                begin = int(item)
            except ValueError:
                return None
            end = None
            last_end = -1
        elif '-' in item:
            begin, end = item.split('-', 1)
            begin = begin.strip()
            end = end.strip()
            if not begin.isdigit():
                return None
            begin = int(begin)
            if begin < last_end or last_end < 0:
                return None
            if end:
                if not end.isdigit():
                    return None
                end = int(end) + 1
                if begin >= end:
                    return None
            else:
                end = None
            last_end = end
        ranges.append((begin, end))
    return Range(units, ranges)
def parse_content_range_header(value, on_update=None):
    if value is None:
        return None
    try:
        units, rangedef = (value or '').strip().split(None, 1)
    except ValueError:
        return None
    if '/' not in rangedef:
        return None
    rng, length = rangedef.split('/', 1)
    if length == '*':
        length = None
    elif length.isdigit():
        length = int(length)
    else:
        return None
    if rng == '*':
        return ContentRange(units, None, None, length, on_update=on_update)
    elif '-' not in rng:
        return None
    start, stop = rng.split('-', 1)
    try:
        start = int(start)
        stop = int(stop) + 1
    except ValueError:
        return None
    if is_byte_range_valid(start, stop, length):
        return ContentRange(units, start, stop, length, on_update=on_update)
def quote_etag(etag, weak=False):
    if '"' in etag:
        raise ValueError('invalid etag')
    etag = '"%s"' % etag
    if weak:
        etag = 'W/' + etag
    return etag
def unquote_etag(etag):
    if not etag:
        return None, None
    etag = etag.strip()
    weak = False
    if etag.startswith(('W/', 'w/')):
        weak = True
        etag = etag[2:]
    if etag[:1] == etag[-1:] == '"':
        etag = etag[1:-1]
    return etag, weak
def parse_etags(value):
    if not value:
        return ETags()
    strong = []
    weak = []
    end = len(value)
    pos = 0
    while pos < end:
        match = _etag_re.match(value, pos)
        if match is None:
            break
        is_weak, quoted, raw = match.groups()
        if raw == '*':
            return ETags(star_tag=True)
        elif quoted:
            raw = quoted
        if is_weak:
            weak.append(raw)
        else:
            strong.append(raw)
        pos = match.end()
    return ETags(strong, weak)
def generate_etag(data):
    return md5(data).hexdigest()
def parse_date(value):
    if value:
        t = parsedate_tz(value.strip())
        if t is not None:
            try:
                year = t[0]
                if year >= 0 and year <= 68:
                    year += 2000
                elif year >= 69 and year <= 99:
                    year += 1900
                return datetime(*((year,) + t[1:7])) -                    timedelta(seconds=t[-1] or 0)
            except (ValueError, OverflowError):
                return None
def _dump_date(d, delim):
    if d is None:
        d = gmtime()
    elif isinstance(d, datetime):
        d = d.utctimetuple()
    elif isinstance(d, (integer_types, float)):
        d = gmtime(d)
    return '%s, %02d%s%s%s%s %02d:%02d:%02d GMT' % (
        ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')[d.tm_wday],
        d.tm_mday, delim,
        ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
         'Oct', 'Nov', 'Dec')[d.tm_mon - 1],
        delim, str(d.tm_year), d.tm_hour, d.tm_min, d.tm_sec
    )
def cookie_date(expires=None):
    return _dump_date(expires, '-')
def http_date(timestamp=None):
    return _dump_date(timestamp, ' ')
def is_resource_modified(environ, etag=None, data=None, last_modified=None,
                         ignore_if_range=True):
    if etag is None and data is not None:
        etag = generate_etag(data)
    elif data is not None:
        raise TypeError('both data and etag given')
    if environ['REQUEST_METHOD'] not in ('GET', 'HEAD'):
        return False
    unmodified = False
    if isinstance(last_modified, string_types):
        last_modified = parse_date(last_modified)
    if last_modified is not None:
        last_modified = last_modified.replace(microsecond=0)
    if_range = None
    if not ignore_if_range and 'HTTP_RANGE' in environ:
        if_range = parse_if_range_header(environ.get('HTTP_IF_RANGE'))
    if if_range is not None and if_range.date is not None:
        modified_since = if_range.date
    else:
        modified_since = parse_date(environ.get('HTTP_IF_MODIFIED_SINCE'))
    if modified_since and last_modified and last_modified <= modified_since:
        unmodified = True
    if etag:
        etag, _ = unquote_etag(etag)
        if if_range is not None and if_range.etag is not None:
            unmodified = parse_etags(if_range.etag).contains(etag)
        else:
            if_none_match = parse_etags(environ.get('HTTP_IF_NONE_MATCH'))
            if if_none_match:
                unmodified = if_none_match.contains_weak(etag)
    return not unmodified
def remove_entity_headers(headers, allowed=('expires', 'content-location')):
    allowed = set(x.lower() for x in allowed)
    headers[:] = [(key, value) for key, value in headers if
                  not is_entity_header(key) or key.lower() in allowed]
def remove_hop_by_hop_headers(headers):
    headers[:] = [(key, value) for key, value in headers if
                  not is_hop_by_hop_header(key)]
def is_entity_header(header):
    return header.lower() in _entity_headers
def is_hop_by_hop_header(header):
    return header.lower() in _hop_by_hop_headers
def parse_cookie(header, charset='utf-8', errors='replace', cls=None):
    if isinstance(header, dict):
        header = header.get('HTTP_COOKIE', '')
    elif header is None:
        header = ''
    if isinstance(header, text_type):
        header = header.encode('latin1', 'replace')
    if cls is None:
        cls = TypeConversionDict
    def _parse_pairs():
        for key, val in _cookie_parse_impl(header):
            key = to_unicode(key, charset, errors, allow_none_charset=True)
            val = to_unicode(val, charset, errors, allow_none_charset=True)
            yield try_coerce_native(key), val
    return cls(_parse_pairs())
def dump_cookie(key, value='', max_age=None, expires=None, path='/',
                domain=None, secure=False, httponly=False,
                charset='utf-8', sync_expires=True):
    key = to_bytes(key, charset)
    value = to_bytes(value, charset)
    if path is not None:
        path = iri_to_uri(path, charset)
    domain = _make_cookie_domain(domain)
    if isinstance(max_age, timedelta):
        max_age = (max_age.days * 60 * 60 * 24) + max_age.seconds
    if expires is not None:
        if not isinstance(expires, string_types):
            expires = cookie_date(expires)
    elif max_age is not None and sync_expires:
        expires = to_bytes(cookie_date(time() + max_age))
    buf = [key + b'=' + _cookie_quote(value)]
    for k, v, q in ((b'Domain', domain, True),
                    (b'Expires', expires, False,),
                    (b'Max-Age', max_age, False),
                    (b'Secure', secure, None),
                    (b'HttpOnly', httponly, None),
                    (b'Path', path, False)):
        if q is None:
            if v:
                buf.append(k)
            continue
        if v is None:
            continue
        tmp = bytearray(k)
        if not isinstance(v, (bytes, bytearray)):
            v = to_bytes(text_type(v), charset)
        if q:
            v = _cookie_quote(v)
        tmp += b'=' + v
        buf.append(bytes(tmp))
    rv = b'; '.join(buf)
    if not PY2:
        rv = rv.decode('latin1')
    return rv
def is_byte_range_valid(start, stop, length):
    if (start is None) != (stop is None):
        return False
    elif start is None:
        return length is None or length >= 0
    elif length is None:
        return 0 <= start < stop
    elif start >= stop:
        return False
    return 0 <= start < length
from werkzeug.datastructures import Accept, HeaderSet, ETags, Authorization,    WWWAuthenticate, TypeConversionDict, IfRange, Range, ContentRange,    RequestCacheControl
from werkzeug.datastructures import (
    MIMEAccept, CharsetAccept, LanguageAccept, Headers
)
from werkzeug.urls import iri_to_uri
