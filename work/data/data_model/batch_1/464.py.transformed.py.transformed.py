
"""
    werkzeug.formparser
    ~~~~~~~~~~~~~~~~~~~
    This module implements the form parsing.  It supports url-encoded forms
    as well as non-nested multipart uploads.
    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import re
import codecs
from io import BytesIO
from tempfile import TemporaryFile
from itertools import chain, repeat, tee
from functools import update_wrapper
from werkzeug._compat import to_native, text_type
from werkzeug.urls import url_decode_stream
from werkzeug.wsgi import make_line_iter,    get_input_stream, get_content_length
from werkzeug.datastructures import Headers, FileStorage, MultiDict
from werkzeug.http import parse_options_header
_empty_string_iter = repeat('')
_multipart_boundary_re = re.compile('^[ -~]{0,200}[!-~]$')
_supported_multipart_encodings = frozenset(['base64', 'quoted-printable'])
def default_stream_factory(total_content_length, filename, content_type,
                           content_length=None):
    if total_content_length > 1024 * 500:
        return TemporaryFile('wb+')
    return BytesIO()
def parse_form_data(environ, stream_factory=None, charset='utf-8',
                    errors='replace', max_form_memory_size=None,
                    max_content_length=None, cls=None,
                    silent=True):
    return FormDataParser(stream_factory, charset, errors,
                          max_form_memory_size, max_content_length,
                          cls, silent).parse_from_environ(environ)
def exhaust_stream(f):
    def wrapper(self, stream, *args, **kwargs):
        try:
            return f(self, stream, *args, **kwargs)
        finally:
            exhaust = getattr(stream, 'exhaust', None)
            if exhaust is not None:
                exhaust()
            else:
                while 1:
                    chunk = stream.read(1024 * 64)
                    if not chunk:
                        break
    return update_wrapper(wrapper, f)
class FormDataParser(object):
    def __init__(self, stream_factory=None, charset='utf-8',
                 errors='replace', max_form_memory_size=None,
                 max_content_length=None, cls=None,
                 silent=True):
        if stream_factory is None:
            stream_factory = default_stream_factory
        self.stream_factory = stream_factory
        self.charset = charset
        self.errors = errors
        self.max_form_memory_size = max_form_memory_size
        self.max_content_length = max_content_length
        if cls is None:
            cls = MultiDict
        self.cls = cls
        self.silent = silent
    def get_parse_func(self, mimetype, options):
        return self.parse_functions.get(mimetype)
    def parse_from_environ(self, environ):
        content_type = environ.get('CONTENT_TYPE', '')
        content_length = get_content_length(environ)
        mimetype, options = parse_options_header(content_type)
        return self.parse(get_input_stream(environ), mimetype,
                          content_length, options)
    def parse(self, stream, mimetype, content_length, options=None):
        if self.max_content_length is not None and           content_length is not None and           content_length > self.max_content_length:
            raise exceptions.RequestEntityTooLarge()
        if options is None:
            options = {}
        parse_func = self.get_parse_func(mimetype, options)
        if parse_func is not None:
            try:
                return parse_func(self, stream, mimetype,
                                  content_length, options)
            except ValueError:
                if not self.silent:
                    raise
        return stream, self.cls(), self.cls()
    @exhaust_stream
    def _parse_multipart(self, stream, mimetype, content_length, options):
        parser = MultiPartParser(self.stream_factory, self.charset, self.errors,
                                 max_form_memory_size=self.max_form_memory_size,
                                 cls=self.cls)
        boundary = options.get('boundary')
        if boundary is None:
            raise ValueError('Missing boundary')
        if isinstance(boundary, text_type):
            boundary = boundary.encode('ascii')
        form, files = parser.parse(stream, boundary, content_length)
        return stream, form, files
    @exhaust_stream
    def _parse_urlencoded(self, stream, mimetype, content_length, options):
        if self.max_form_memory_size is not None and           content_length is not None and           content_length > self.max_form_memory_size:
            raise exceptions.RequestEntityTooLarge()
        form = url_decode_stream(stream, self.charset,
                                 errors=self.errors, cls=self.cls)
        return stream, form, self.cls()
    parse_functions = {
        'multipart/form-data':                  _parse_multipart,
        'application/x-www-form-urlencoded':    _parse_urlencoded,
        'application/x-url-encoded':            _parse_urlencoded
    }
def is_valid_multipart_boundary(boundary):
    return _multipart_boundary_re.match(boundary) is not None
def _line_parse(line):
    if line[-2:] in ['\r\n', b'\r\n']:
        return line[:-2], True
    elif line[-1:] in ['\r', '\n', b'\r', b'\n']:
        return line[:-1], True
    return line, False
def parse_multipart_headers(iterable):
    result = []
    for line in iterable:
        line = to_native(line)
        line, line_terminated = _line_parse(line)
        if not line_terminated:
            raise ValueError('unexpected end of line in multipart header')
        if not line:
            break
        elif line[0] in ' \t' and result:
            key, value = result[-1]
            result[-1] = (key, value + '\n ' + line[1:])
        else:
            parts = line.split(':', 1)
            if len(parts) == 2:
                result.append((parts[0].strip(), parts[1].strip()))
    return Headers(result)
_begin_form = 'begin_form'
_begin_file = 'begin_file'
_cont = 'cont'
_end = 'end'
class MultiPartParser(object):
    def __init__(self, stream_factory=None, charset='utf-8', errors='replace',
                 max_form_memory_size=None, cls=None, buffer_size=64 * 1024):
        self.charset = charset
        self.errors = errors
        self.max_form_memory_size = max_form_memory_size
        self.stream_factory = default_stream_factory if stream_factory is None else stream_factory
        self.cls = MultiDict if cls is None else cls
        assert buffer_size % 4 == 0, 'buffer size has to be divisible by 4'
        assert buffer_size >= 1024, 'buffer size has to be at least 1KB'
        self.buffer_size = buffer_size
    def _fix_ie_filename(self, filename):
        if filename[1:3] == ':\\' or filename[:2] == '\\\\':
            return filename.split('\\')[-1]
        return filename
    def _find_terminator(self, iterator):
        for line in iterator:
            if not line:
                break
            line = line.strip()
            if line:
                return line
        return b''
    def fail(self, message):
        raise ValueError(message)
    def get_part_encoding(self, headers):
        transfer_encoding = headers.get('content-transfer-encoding')
        if transfer_encoding is not None and           transfer_encoding in _supported_multipart_encodings:
            return transfer_encoding
    def get_part_charset(self, headers):
        content_type = headers.get('content-type')
        if content_type:
            mimetype, ct_params = parse_options_header(content_type)
            return ct_params.get('charset', self.charset)
        return self.charset
    def start_file_streaming(self, filename, headers, total_content_length):
        if isinstance(filename, bytes):
            filename = filename.decode(self.charset, self.errors)
        filename = self._fix_ie_filename(filename)
        content_type = headers.get('content-type')
        try:
            content_length = int(headers['content-length'])
        except (KeyError, ValueError):
            content_length = 0
        container = self.stream_factory(total_content_length, content_type,
                                        filename, content_length)
        return filename, container
    def in_memory_threshold_reached(self, bytes):
        raise exceptions.RequestEntityTooLarge()
    def validate_boundary(self, boundary):
        if not boundary:
            self.fail('Missing boundary')
        if not is_valid_multipart_boundary(boundary):
            self.fail('Invalid boundary: %s' % boundary)
        if len(boundary) > self.buffer_size:
            self.fail('Boundary longer than buffer size')
    def parse_lines(self, file, boundary, content_length, cap_at_buffer=True):
        next_part = b'--' + boundary
        last_part = next_part + b'--'
        iterator = chain(make_line_iter(file, limit=content_length,
                                        buffer_size=self.buffer_size,
                                        cap_at_buffer=cap_at_buffer),
                         _empty_string_iter)
        terminator = self._find_terminator(iterator)
        if terminator == last_part:
            return
        elif terminator != next_part:
            self.fail('Expected boundary at start of multipart data')
        while terminator != last_part:
            headers = parse_multipart_headers(iterator)
            disposition = headers.get('content-disposition')
            if disposition is None:
                self.fail('Missing Content-Disposition header')
            disposition, extra = parse_options_header(disposition)
            transfer_encoding = self.get_part_encoding(headers)
            name = extra.get('name')
            filename = extra.get('filename')
            if filename is None:
                yield _begin_form, (headers, name)
            else:
                yield _begin_file, (headers, name, filename)
            buf = b''
            for line in iterator:
                if not line:
                    self.fail('unexpected end of stream')
                if line[:2] == b'--':
                    terminator = line.rstrip()
                    if terminator in (next_part, last_part):
                        break
                if transfer_encoding is not None:
                    if transfer_encoding == 'base64':
                        transfer_encoding = 'base64_codec'
                    try:
                        line = codecs.decode(line, transfer_encoding)
                    except Exception:
                        self.fail('could not decode transfer encoded chunk')
                if buf:
                    yield _cont, buf
                    buf = b''
                if line[-2:] == b'\r\n':
                    buf = b'\r\n'
                    cutoff = -2
                else:
                    buf = line[-1:]
                    cutoff = -1
                yield _cont, line[:cutoff]
            else:
                raise ValueError('unexpected end of part')
            if buf not in (b'', b'\r', b'\n', b'\r\n'):
                yield _cont, buf
            yield _end, None
    def parse_parts(self, file, boundary, content_length):
        in_memory = 0
        for ellt, ell in self.parse_lines(file, boundary, content_length):
            if ellt == _begin_file:
                headers, name, filename = ell
                is_file = True
                guard_memory = False
                filename, container = self.start_file_streaming(
                    filename, headers, content_length)
                _write = container.write
            elif ellt == _begin_form:
                headers, name = ell
                is_file = False
                container = []
                _write = container.append
                guard_memory = self.max_form_memory_size is not None
            elif ellt == _cont:
                _write(ell)
                if guard_memory:
                    in_memory += len(ell)
                    if in_memory > self.max_form_memory_size:
                        self.in_memory_threshold_reached(in_memory)
            elif ellt == _end:
                if is_file:
                    container.seek(0)
                    yield ('file',
                           (name, FileStorage(container, filename, name,
                                              headers=headers)))
                else:
                    part_charset = self.get_part_charset(headers)
                    yield ('form',
                           (name, b''.join(container).decode(
                               part_charset, self.errors)))
    def parse(self, file, boundary, content_length):
        formstream, filestream = tee(
            self.parse_parts(file, boundary, content_length), 2)
        form = (p[1] for p in formstream if p[0] == 'form')
        files = (p[1] for p in filestream if p[0] == 'file')
        return self.cls(form), self.cls(files)
from werkzeug import exceptions
