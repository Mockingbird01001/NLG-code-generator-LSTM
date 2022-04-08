
"""
    werkzeug.wsgi
    ~~~~~~~~~~~~~
    This module implements WSGI related helpers.
    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import re
import os
import posixpath
import mimetypes
from itertools import chain
from zlib import adler32
from time import time, mktime
from datetime import datetime
from functools import partial, update_wrapper
from werkzeug._compat import iteritems, text_type, string_types,    implements_iterator, make_literal_wrapper, to_unicode, to_bytes,    wsgi_get_bytes, try_coerce_native, PY2, BytesIO
from werkzeug._internal import _empty_stream, _encode_idna
from werkzeug.http import is_resource_modified, http_date
from werkzeug.urls import uri_to_iri, url_quote, url_parse, url_join
from werkzeug.filesystem import get_filesystem_encoding
def responder(f):
    return update_wrapper(lambda *a: f(*a)(*a[-2:]), f)
def get_current_url(environ, root_only=False, strip_querystring=False,
                    host_only=False, trusted_hosts=None):
    tmp = [environ['wsgi.url_scheme'], '://', get_host(environ, trusted_hosts)]
    cat = tmp.append
    if host_only:
        return uri_to_iri(''.join(tmp) + '/')
    cat(url_quote(wsgi_get_bytes(environ.get('SCRIPT_NAME', ''))).rstrip('/'))
    cat('/')
    if not root_only:
        cat(url_quote(wsgi_get_bytes(environ.get('PATH_INFO', '')).lstrip(b'/')))
        if not strip_querystring:
            qs = get_query_string(environ)
            if qs:
                cat('?' + qs)
    return uri_to_iri(''.join(tmp))
def host_is_trusted(hostname, trusted_list):
    if not hostname:
        return False
    if isinstance(trusted_list, string_types):
        trusted_list = [trusted_list]
    def _normalize(hostname):
        if ':' in hostname:
            hostname = hostname.rsplit(':', 1)[0]
        return _encode_idna(hostname)
    try:
        hostname = _normalize(hostname)
    except UnicodeError:
        return False
    for ref in trusted_list:
        if ref.startswith('.'):
            ref = ref[1:]
            suffix_match = True
        else:
            suffix_match = False
        try:
            ref = _normalize(ref)
        except UnicodeError:
            return False
        if ref == hostname:
            return True
        if suffix_match and hostname.endswith('.' + ref):
            return True
    return False
def get_host(environ, trusted_hosts=None):
    if 'HTTP_X_FORWARDED_HOST' in environ:
        rv = environ['HTTP_X_FORWARDED_HOST'].split(',', 1)[0].strip()
    elif 'HTTP_HOST' in environ:
        rv = environ['HTTP_HOST']
    else:
        rv = environ['SERVER_NAME']
        if (environ['wsgi.url_scheme'], environ['SERVER_PORT']) not           in (('https', '443'), ('http', '80')):
            rv += ':' + environ['SERVER_PORT']
    if trusted_hosts is not None:
        if not host_is_trusted(rv, trusted_hosts):
            from werkzeug.exceptions import SecurityError
            raise SecurityError('Host "%s" is not trusted' % rv)
    return rv
def get_content_length(environ):
    content_length = environ.get('CONTENT_LENGTH')
    if content_length is not None:
        try:
            return max(0, int(content_length))
        except (ValueError, TypeError):
            pass
def get_input_stream(environ, safe_fallback=True):
    stream = environ['wsgi.input']
    content_length = get_content_length(environ)
    if environ.get('wsgi.input_terminated'):
        return stream
    if content_length is None:
        return safe_fallback and _empty_stream or stream
    return LimitedStream(stream, content_length)
def get_query_string(environ):
    qs = wsgi_get_bytes(environ.get('QUERY_STRING', ''))
    return try_coerce_native(url_quote(qs, safe=':&%=+$!*\'(),'))
def get_path_info(environ, charset='utf-8', errors='replace'):
    path = wsgi_get_bytes(environ.get('PATH_INFO', ''))
    return to_unicode(path, charset, errors, allow_none_charset=True)
def get_script_name(environ, charset='utf-8', errors='replace'):
    path = wsgi_get_bytes(environ.get('SCRIPT_NAME', ''))
    return to_unicode(path, charset, errors, allow_none_charset=True)
def pop_path_info(environ, charset='utf-8', errors='replace'):
    path = environ.get('PATH_INFO')
    if not path:
        return None
    script_name = environ.get('SCRIPT_NAME', '')
    old_path = path
    path = path.lstrip('/')
    if path != old_path:
        script_name += '/' * (len(old_path) - len(path))
    if '/' not in path:
        environ['PATH_INFO'] = ''
        environ['SCRIPT_NAME'] = script_name + path
        rv = wsgi_get_bytes(path)
    else:
        segment, path = path.split('/', 1)
        environ['PATH_INFO'] = '/' + path
        environ['SCRIPT_NAME'] = script_name + segment
        rv = wsgi_get_bytes(segment)
    return to_unicode(rv, charset, errors, allow_none_charset=True)
def peek_path_info(environ, charset='utf-8', errors='replace'):
    segments = environ.get('PATH_INFO', '').lstrip('/').split('/', 1)
    if segments:
        return to_unicode(wsgi_get_bytes(segments[0]),
                          charset, errors, allow_none_charset=True)
def extract_path_info(environ_or_baseurl, path_or_url, charset='utf-8',
                      errors='replace', collapse_http_schemes=True):
    def _normalize_netloc(scheme, netloc):
        parts = netloc.split(u'@', 1)[-1].split(u':', 1)
        if len(parts) == 2:
            netloc, port = parts
            if (scheme == u'http' and port == u'80') or               (scheme == u'https' and port == u'443'):
                port = None
        else:
            netloc = parts[0]
            port = None
        if port is not None:
            netloc += u':' + port
        return netloc
    path = uri_to_iri(path_or_url, charset, errors)
    if isinstance(environ_or_baseurl, dict):
        environ_or_baseurl = get_current_url(environ_or_baseurl,
                                             root_only=True)
    base_iri = uri_to_iri(environ_or_baseurl, charset, errors)
    base_scheme, base_netloc, base_path = url_parse(base_iri)[:3]
    cur_scheme, cur_netloc, cur_path, =        url_parse(url_join(base_iri, path))[:3]
    base_netloc = _normalize_netloc(base_scheme, base_netloc)
    cur_netloc = _normalize_netloc(cur_scheme, cur_netloc)
    if collapse_http_schemes:
        for scheme in base_scheme, cur_scheme:
            if scheme not in (u'http', u'https'):
                return None
    else:
        if not (base_scheme in (u'http', u'https') and
                base_scheme == cur_scheme):
            return None
    if base_netloc != cur_netloc:
        return None
    base_path = base_path.rstrip(u'/')
    if not cur_path.startswith(base_path):
        return None
    return u'/' + cur_path[len(base_path):].lstrip(u'/')
class SharedDataMiddleware(object):
    def __init__(self, app, exports, disallow=None, cache=True,
                 cache_timeout=60 * 60 * 12, fallback_mimetype='text/plain'):
        self.app = app
        self.exports = {}
        self.cache = cache
        self.cache_timeout = cache_timeout
        for key, value in iteritems(exports):
            if isinstance(value, tuple):
                loader = self.get_package_loader(*value)
            elif isinstance(value, string_types):
                if os.path.isfile(value):
                    loader = self.get_file_loader(value)
                else:
                    loader = self.get_directory_loader(value)
            else:
                raise TypeError('unknown def %r' % value)
            self.exports[key] = loader
        if disallow is not None:
            from fnmatch import fnmatch
            self.is_allowed = lambda x: not fnmatch(x, disallow)
        self.fallback_mimetype = fallback_mimetype
    def is_allowed(self, filename):
        return True
    def _opener(self, filename):
        return lambda: (
            open(filename, 'rb'),
            datetime.utcfromtimestamp(os.path.getmtime(filename)),
            int(os.path.getsize(filename))
        )
    def get_file_loader(self, filename):
        return lambda x: (os.path.basename(filename), self._opener(filename))
    def get_package_loader(self, package, package_path):
        from pkg_resources import DefaultProvider, ResourceManager,            get_provider
        loadtime = datetime.utcnow()
        provider = get_provider(package)
        manager = ResourceManager()
        filesystem_bound = isinstance(provider, DefaultProvider)
        def loader(path):
            if path is None:
                return None, None
            path = posixpath.join(package_path, path)
            if not provider.has_resource(path):
                return None, None
            basename = posixpath.basename(path)
            if filesystem_bound:
                return basename, self._opener(
                    provider.get_resource_filename(manager, path))
            s = provider.get_resource_string(manager, path)
            return basename, lambda: (
                BytesIO(s),
                loadtime,
                len(s)
            )
        return loader
    def get_directory_loader(self, directory):
        def loader(path):
            if path is not None:
                path = os.path.join(directory, path)
            else:
                path = directory
            if os.path.isfile(path):
                return os.path.basename(path), self._opener(path)
            return None, None
        return loader
    def generate_etag(self, mtime, file_size, real_filename):
        if not isinstance(real_filename, bytes):
            real_filename = real_filename.encode(get_filesystem_encoding())
        return 'wzsdm-%d-%s-%s' % (
            mktime(mtime.timetuple()),
            file_size,
            adler32(real_filename) & 0xffffffff
        )
    def __call__(self, environ, start_response):
        cleaned_path = get_path_info(environ)
        if PY2:
            cleaned_path = cleaned_path.encode(get_filesystem_encoding())
        cleaned_path = cleaned_path.strip('/')
        for sep in os.sep, os.altsep:
            if sep and sep != '/':
                cleaned_path = cleaned_path.replace(sep, '/')
        path = '/' + '/'.join(x for x in cleaned_path.split('/')
                              if x and x != '..')
        file_loader = None
        for search_path, loader in iteritems(self.exports):
            if search_path == path:
                real_filename, file_loader = loader(None)
                if file_loader is not None:
                    break
            if not search_path.endswith('/'):
                search_path += '/'
            if path.startswith(search_path):
                real_filename, file_loader = loader(path[len(search_path):])
                if file_loader is not None:
                    break
        if file_loader is None or not self.is_allowed(real_filename):
            return self.app(environ, start_response)
        guessed_type = mimetypes.guess_type(real_filename)
        mime_type = guessed_type[0] or self.fallback_mimetype
        f, mtime, file_size = file_loader()
        headers = [('Date', http_date())]
        if self.cache:
            timeout = self.cache_timeout
            etag = self.generate_etag(mtime, file_size, real_filename)
            headers += [
                ('Etag', '"%s"' % etag),
                ('Cache-Control', 'max-age=%d, public' % timeout)
            ]
            if not is_resource_modified(environ, etag, last_modified=mtime):
                f.close()
                start_response('304 Not Modified', headers)
                return []
            headers.append(('Expires', http_date(time() + timeout)))
        else:
            headers.append(('Cache-Control', 'public'))
        headers.extend((
            ('Content-Type', mime_type),
            ('Content-Length', str(file_size)),
            ('Last-Modified', http_date(mtime))
        ))
        start_response('200 OK', headers)
        return wrap_file(environ, f)
class DispatcherMiddleware(object):
    def __init__(self, app, mounts=None):
        self.app = app
        self.mounts = mounts or {}
    def __call__(self, environ, start_response):
        script = environ.get('PATH_INFO', '')
        path_info = ''
        while '/' in script:
            if script in self.mounts:
                app = self.mounts[script]
                break
            script, last_item = script.rsplit('/', 1)
            path_info = '/%s%s' % (last_item, path_info)
        else:
            app = self.mounts.get(script, self.app)
        original_script_name = environ.get('SCRIPT_NAME', '')
        environ['SCRIPT_NAME'] = original_script_name + script
        environ['PATH_INFO'] = path_info
        return app(environ, start_response)
@implements_iterator
class ClosingIterator(object):
    def __init__(self, iterable, callbacks=None):
        iterator = iter(iterable)
        self._next = partial(next, iterator)
        if callbacks is None:
            callbacks = []
        elif callable(callbacks):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)
        iterable_close = getattr(iterator, 'close', None)
        if iterable_close:
            callbacks.insert(0, iterable_close)
        self._callbacks = callbacks
    def __iter__(self):
        return self
    def __next__(self):
        return self._next()
    def close(self):
        for callback in self._callbacks:
            callback()
def wrap_file(environ, file, buffer_size=8192):
    return environ.get('wsgi.file_wrapper', FileWrapper)(file, buffer_size)
@implements_iterator
class FileWrapper(object):
    def __init__(self, file, buffer_size=8192):
        self.file = file
        self.buffer_size = buffer_size
    def close(self):
        if hasattr(self.file, 'close'):
            self.file.close()
    def seekable(self):
        if hasattr(self.file, 'seekable'):
            return self.file.seekable()
        if hasattr(self.file, 'seek'):
            return True
        return False
    def seek(self, *args):
        if hasattr(self.file, 'seek'):
            self.file.seek(*args)
    def tell(self):
        if hasattr(self.file, 'tell'):
            return self.file.tell()
        return None
    def __iter__(self):
        return self
    def __next__(self):
        data = self.file.read(self.buffer_size)
        if data:
            return data
        raise StopIteration()
@implements_iterator
class _RangeWrapper(object):
    def __init__(self, iterable, start_byte=0, byte_range=None):
        self.iterable = iter(iterable)
        self.byte_range = byte_range
        self.start_byte = start_byte
        self.end_byte = None
        if byte_range is not None:
            self.end_byte = self.start_byte + self.byte_range
        self.read_length = 0
        self.seekable = hasattr(iterable, 'seekable') and iterable.seekable()
        self.end_reached = False
    def __iter__(self):
        return self
    def _next_chunk(self):
        try:
            chunk = next(self.iterable)
            self.read_length += len(chunk)
            return chunk
        except StopIteration:
            self.end_reached = True
            raise
    def _first_iteration(self):
        chunk = None
        if self.seekable:
            self.iterable.seek(self.start_byte)
            self.read_length = self.iterable.tell()
            contextual_read_length = self.read_length
        else:
            while self.read_length <= self.start_byte:
                chunk = self._next_chunk()
            if chunk is not None:
                chunk = chunk[self.start_byte - self.read_length:]
            contextual_read_length = self.start_byte
        return chunk, contextual_read_length
    def _next(self):
        if self.end_reached:
            raise StopIteration()
        chunk = None
        contextual_read_length = self.read_length
        if self.read_length == 0:
            chunk, contextual_read_length = self._first_iteration()
        if chunk is None:
            chunk = self._next_chunk()
        if self.end_byte is not None and self.read_length >= self.end_byte:
            self.end_reached = True
            return chunk[:self.end_byte - contextual_read_length]
        return chunk
    def __next__(self):
        chunk = self._next()
        if chunk:
            return chunk
        self.end_reached = True
        raise StopIteration()
    def close(self):
        if hasattr(self.iterable, 'close'):
            self.iterable.close()
def _make_chunk_iter(stream, limit, buffer_size):
    if isinstance(stream, (bytes, bytearray, text_type)):
        raise TypeError('Passed a string or byte object instead of '
                        'true iterator or stream.')
    if not hasattr(stream, 'read'):
        for item in stream:
            if item:
                yield item
        return
    if not isinstance(stream, LimitedStream) and limit is not None:
        stream = LimitedStream(stream, limit)
    _read = stream.read
    while 1:
        item = _read(buffer_size)
        if not item:
            break
        yield item
def make_line_iter(stream, limit=None, buffer_size=10 * 1024,
                   cap_at_buffer=False):
    _iter = _make_chunk_iter(stream, limit, buffer_size)
    first_item = next(_iter, '')
    if not first_item:
        return
    s = make_literal_wrapper(first_item)
    empty = s('')
    cr = s('\r')
    lf = s('\n')
    crlf = s('\r\n')
    _iter = chain((first_item,), _iter)
    def _iter_basic_lines():
        _join = empty.join
        buffer = []
        while 1:
            new_data = next(_iter, '')
            if not new_data:
                break
            new_buf = []
            buf_size = 0
            for item in chain(buffer, new_data.splitlines(True)):
                new_buf.append(item)
                buf_size += len(item)
                if item and item[-1:] in crlf:
                    yield _join(new_buf)
                    new_buf = []
                elif cap_at_buffer and buf_size >= buffer_size:
                    rv = _join(new_buf)
                    while len(rv) >= buffer_size:
                        yield rv[:buffer_size]
                        rv = rv[buffer_size:]
                    new_buf = [rv]
            buffer = new_buf
        if buffer:
            yield _join(buffer)
    previous = empty
    for item in _iter_basic_lines():
        if item == lf and previous[-1:] == cr:
            previous += item
            item = empty
        if previous:
            yield previous
        previous = item
    if previous:
        yield previous
def make_chunk_iter(stream, separator, limit=None, buffer_size=10 * 1024,
                    cap_at_buffer=False):
    _iter = _make_chunk_iter(stream, limit, buffer_size)
    first_item = next(_iter, '')
    if not first_item:
        return
    _iter = chain((first_item,), _iter)
    if isinstance(first_item, text_type):
        separator = to_unicode(separator)
        _split = re.compile(r'(%s)' % re.escape(separator)).split
        _join = u''.join
    else:
        separator = to_bytes(separator)
        _split = re.compile(b'(' + re.escape(separator) + b')').split
        _join = b''.join
    buffer = []
    while 1:
        new_data = next(_iter, '')
        if not new_data:
            break
        chunks = _split(new_data)
        new_buf = []
        buf_size = 0
        for item in chain(buffer, chunks):
            if item == separator:
                yield _join(new_buf)
                new_buf = []
                buf_size = 0
            else:
                buf_size += len(item)
                new_buf.append(item)
                if cap_at_buffer and buf_size >= buffer_size:
                    rv = _join(new_buf)
                    while len(rv) >= buffer_size:
                        yield rv[:buffer_size]
                        rv = rv[buffer_size:]
                    new_buf = [rv]
                    buf_size = len(rv)
        buffer = new_buf
    if buffer:
        yield _join(buffer)
@implements_iterator
class LimitedStream(object):
    def __init__(self, stream, limit):
        self._read = stream.read
        self._readline = stream.readline
        self._pos = 0
        self.limit = limit
    def __iter__(self):
        return self
    @property
    def is_exhausted(self):
        return self._pos >= self.limit
    def on_exhausted(self):
        return self._read(0)
    def on_disconnect(self):
        from werkzeug.exceptions import ClientDisconnected
        raise ClientDisconnected()
    def exhaust(self, chunk_size=1024 * 64):
        to_read = self.limit - self._pos
        chunk = chunk_size
        while to_read > 0:
            chunk = min(to_read, chunk)
            self.read(chunk)
            to_read -= chunk
    def read(self, size=None):
        if self._pos >= self.limit:
            return self.on_exhausted()
        if size is None or size == -1:
            size = self.limit
        to_read = min(self.limit - self._pos, size)
        try:
            read = self._read(to_read)
        except (IOError, ValueError):
            return self.on_disconnect()
        if to_read and len(read) != to_read:
            return self.on_disconnect()
        self._pos += len(read)
        return read
    def readline(self, size=None):
        if self._pos >= self.limit:
            return self.on_exhausted()
        if size is None:
            size = self.limit - self._pos
        else:
            size = min(size, self.limit - self._pos)
        try:
            line = self._readline(size)
        except (ValueError, IOError):
            return self.on_disconnect()
        if size and not line:
            return self.on_disconnect()
        self._pos += len(line)
        return line
    def readlines(self, size=None):
        last_pos = self._pos
        result = []
        if size is not None:
            end = min(self.limit, last_pos + size)
        else:
            end = self.limit
        while 1:
            if size is not None:
                size -= last_pos - self._pos
            if self._pos >= end:
                break
            result.append(self.readline(size))
            if size is not None:
                last_pos = self._pos
        return result
    def tell(self):
        return self._pos
    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration()
        return line
