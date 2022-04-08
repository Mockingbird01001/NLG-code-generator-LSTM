
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import bytes, chr, dict, int, range, super
__all__ = [
    'body_decode',
    'body_encode',
    'body_length',
    'decode',
    'decodestring',
    'header_decode',
    'header_encode',
    'header_length',
    'quote',
    'unquote',
    ]
import re
import io
from string import ascii_letters, digits, hexdigits
CRLF = '\r\n'
NL = '\n'
EMPTYSTRING = ''
_QUOPRI_HEADER_MAP = dict((c, '=%02X' % c) for c in range(256))
_QUOPRI_BODY_MAP = _QUOPRI_HEADER_MAP.copy()
for c in bytes(b'-!*+/' + ascii_letters.encode('ascii') + digits.encode('ascii')):
    _QUOPRI_HEADER_MAP[c] = chr(c)
_QUOPRI_HEADER_MAP[ord(' ')] = '_'
               b'?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`'
               b'abcdefghijklmnopqrstuvwxyz{|}~\t'):
    _QUOPRI_BODY_MAP[c] = chr(c)
def header_check(octet):
    return chr(octet) != _QUOPRI_HEADER_MAP[octet]
def body_check(octet):
    return chr(octet) != _QUOPRI_BODY_MAP[octet]
def header_length(bytearray):
    return sum(len(_QUOPRI_HEADER_MAP[octet]) for octet in bytearray)
def body_length(bytearray):
    return sum(len(_QUOPRI_BODY_MAP[octet]) for octet in bytearray)
def _max_append(L, s, maxlen, extra=''):
    if not isinstance(s, str):
        s = chr(s)
    if not L:
        L.append(s.lstrip())
    elif len(L[-1]) + len(s) <= maxlen:
        L[-1] += extra + s
    else:
        L.append(s.lstrip())
def unquote(s):
    return chr(int(s[1:3], 16))
def quote(c):
    return '=%02X' % ord(c)
def header_encode(header_bytes, charset='iso-8859-1'):
    if not header_bytes:
        return ''
    encoded = []
    for octet in header_bytes:
        encoded.append(_QUOPRI_HEADER_MAP[octet])
    return '=?%s?q?%s?=' % (charset, EMPTYSTRING.join(encoded))
class _body_accumulator(io.StringIO):
    def __init__(self, maxlinelen, eol, *args, **kw):
        super().__init__(*args, **kw)
        self.eol = eol
        self.maxlinelen = self.room = maxlinelen
    def write_str(self, s):
        self.write(s)
        self.room -= len(s)
    def newline(self):
        self.write_str(self.eol)
        self.room = self.maxlinelen
    def write_soft_break(self):
        self.write_str('=')
        self.newline()
    def write_wrapped(self, s, extra_room=0):
        if self.room < len(s) + extra_room:
            self.write_soft_break()
        self.write_str(s)
    def write_char(self, c, is_last_char):
        if not is_last_char:
            self.write_wrapped(c, extra_room=1)
        elif c not in ' \t':
            self.write_wrapped(c)
        elif self.room >= 3:
            self.write(quote(c))
        elif self.room == 2:
            self.write(c)
            self.write_soft_break()
        else:
            self.write_soft_break()
            self.write(quote(c))
def body_encode(body, maxlinelen=76, eol=NL):
    if maxlinelen < 4:
        raise ValueError("maxlinelen must be at least 4")
    if not body:
        return body
    last_has_eol = (body[-1] in '\r\n')
    encoded_body = _body_accumulator(maxlinelen, eol)
    lines = body.splitlines()
    last_line_no = len(lines) - 1
    for line_no, line in enumerate(lines):
        last_char_index = len(line) - 1
        for i, c in enumerate(line):
            if body_check(ord(c)):
                c = quote(c)
            encoded_body.write_char(c, i==last_char_index)
        if line_no < last_line_no or last_has_eol:
            encoded_body.newline()
    return encoded_body.getvalue()
def decode(encoded, eol=NL):
    if not encoded:
        return encoded
    decoded = ''
    for line in encoded.splitlines():
        line = line.rstrip()
        if not line:
            decoded += eol
            continue
        i = 0
        n = len(line)
        while i < n:
            c = line[i]
            if c != '=':
                decoded += c
                i += 1
            elif i+1 == n:
                i += 1
                continue
            elif i+2 < n and line[i+1] in hexdigits and line[i+2] in hexdigits:
                decoded += unquote(line[i:i+3])
                i += 3
            else:
                decoded += c
                i += 1
            if i == n:
                decoded += eol
    if encoded[-1] not in '\r\n' and decoded.endswith(eol):
        decoded = decoded[:-1]
    return decoded
body_decode = decode
decodestring = decode
def _unquote_match(match):
    s = match.group(0)
    return unquote(s)
def header_decode(s):
    s = s.replace('_', ' ')
    return re.sub(r'=[a-fA-F0-9]{2}', _unquote_match, s, re.ASCII)
