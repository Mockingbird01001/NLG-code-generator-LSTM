
"""FeedParser - An email feed parser.
The feed parser implements an interface for incrementally parsing an email
message, line by line.  This has advantages for certain applications, such as
those reading email messages off a socket.
FeedParser.feed() is the primary interface for pushing new data into the
parser.  It returns when there's nothing more it can do with the available
data.  When you have no more data to push into the parser, call .close().
This completes the parsing and returns the root message object.
The other advantage of this parser is that it will never raise a parsing
exception.  Instead, when it finds something unexpected, it adds a 'defect' to
the current message.  Defects are just instances that live on the message
object's .defects attribute.
"""
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import object, range, super
from future.utils import implements_iterator, PY3
__all__ = ['FeedParser', 'BytesFeedParser']
import re
from future.backports.email import errors
from future.backports.email import message
from future.backports.email._policybase import compat32
NLCRE = re.compile('\r\n|\r|\n')
NLCRE_bol = re.compile('(\r\n|\r|\n)')
NLCRE_eol = re.compile('(\r\n|\r|\n)\Z')
NLCRE_crack = re.compile('(\r\n|\r|\n)')
headerRE = re.compile(r'^(From |[\041-\071\073-\176]{1,}:|[\t ])')
EMPTYSTRING = ''
NL = '\n'
NeedMoreData = object()
class BufferedSubFile(object):
    def __init__(self):
        self._partial = ''
        self._lines = []
        self._eofstack = []
        self._closed = False
    def push_eof_matcher(self, pred):
        self._eofstack.append(pred)
    def pop_eof_matcher(self):
        return self._eofstack.pop()
    def close(self):
        self._lines.append(self._partial)
        self._partial = ''
        self._closed = True
    def readline(self):
        if not self._lines:
            if self._closed:
                return ''
            return NeedMoreData
        line = self._lines.pop()
        for ateof in self._eofstack[::-1]:
            if ateof(line):
                self._lines.append(line)
                return ''
        return line
    def unreadline(self, line):
        assert line is not NeedMoreData
        self._lines.append(line)
    def push(self, data):
        data, self._partial = self._partial + data, ''
        parts = NLCRE_crack.split(data)
        self._partial = parts.pop()
        if not self._partial and parts and parts[-1].endswith('\r'):
            self._partial = parts.pop(-2)+parts.pop()
        lines = []
        for i in range(len(parts) // 2):
            lines.append(parts[i*2] + parts[i*2+1])
        self.pushlines(lines)
    def pushlines(self, lines):
        self._lines[:0] = lines[::-1]
    def __iter__(self):
        return self
    def __next__(self):
        line = self.readline()
        if line == '':
            raise StopIteration
        return line
class FeedParser(object):
    def __init__(self, _factory=message.Message, **_3to2kwargs):
        if 'policy' in _3to2kwargs: policy = _3to2kwargs['policy']; del _3to2kwargs['policy']
        else: policy = compat32
        self._factory = _factory
        self.policy = policy
        try:
            _factory(policy=self.policy)
            self._factory_kwds = lambda: {'policy': self.policy}
        except TypeError:
            self._factory_kwds = lambda: {}
        self._input = BufferedSubFile()
        self._msgstack = []
        if PY3:
            self._parse = self._parsegen().__next__
        else:
            self._parse = self._parsegen().next
        self._cur = None
        self._last = None
        self._headersonly = False
    def _set_headersonly(self):
        self._headersonly = True
    def feed(self, data):
        self._input.push(data)
        self._call_parse()
    def _call_parse(self):
        try:
            self._parse()
        except StopIteration:
            pass
    def close(self):
        self._input.close()
        self._call_parse()
        root = self._pop_message()
        assert not self._msgstack
        if root.get_content_maintype() == 'multipart'               and not root.is_multipart():
            defect = errors.MultipartInvariantViolationDefect()
            self.policy.handle_defect(root, defect)
        return root
    def _new_message(self):
        msg = self._factory(**self._factory_kwds())
        if self._cur and self._cur.get_content_type() == 'multipart/digest':
            msg.set_default_type('message/rfc822')
        if self._msgstack:
            self._msgstack[-1].attach(msg)
        self._msgstack.append(msg)
        self._cur = msg
        self._last = msg
    def _pop_message(self):
        retval = self._msgstack.pop()
        if self._msgstack:
            self._cur = self._msgstack[-1]
        else:
            self._cur = None
        return retval
    def _parsegen(self):
        self._new_message()
        headers = []
        for line in self._input:
            if line is NeedMoreData:
                yield NeedMoreData
                continue
            if not headerRE.match(line):
                if not NLCRE.match(line):
                    defect = errors.MissingHeaderBodySeparatorDefect()
                    self.policy.handle_defect(self._cur, defect)
                    self._input.unreadline(line)
                break
            headers.append(line)
        self._parse_headers(headers)
        if self._headersonly:
            lines = []
            while True:
                line = self._input.readline()
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                if line == '':
                    break
                lines.append(line)
            self._cur.set_payload(EMPTYSTRING.join(lines))
            return
        if self._cur.get_content_type() == 'message/delivery-status':
            while True:
                self._input.push_eof_matcher(NLCRE.match)
                for retval in self._parsegen():
                    if retval is NeedMoreData:
                        yield NeedMoreData
                        continue
                    break
                msg = self._pop_message()
                self._input.pop_eof_matcher()
                while True:
                    line = self._input.readline()
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                    break
                while True:
                    line = self._input.readline()
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                    break
                if line == '':
                    break
                self._input.unreadline(line)
            return
        if self._cur.get_content_maintype() == 'message':
            for retval in self._parsegen():
                if retval is NeedMoreData:
                    yield NeedMoreData
                    continue
                break
            self._pop_message()
            return
        if self._cur.get_content_maintype() == 'multipart':
            boundary = self._cur.get_boundary()
            if boundary is None:
                defect = errors.NoBoundaryInMultipartDefect()
                self.policy.handle_defect(self._cur, defect)
                lines = []
                for line in self._input:
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                    lines.append(line)
                self._cur.set_payload(EMPTYSTRING.join(lines))
                return
            if (self._cur.get('content-transfer-encoding', '8bit').lower()
                    not in ('7bit', '8bit', 'binary')):
                defect = errors.InvalidMultipartContentTransferEncodingDefect()
                self.policy.handle_defect(self._cur, defect)
            separator = '--' + boundary
            boundaryre = re.compile(
                '(?P<sep>' + re.escape(separator) +
                r')(?P<end>--)?(?P<ws>[ \t]*)(?P<linesep>\r\n|\r|\n)?$')
            capturing_preamble = True
            preamble = []
            linesep = False
            close_boundary_seen = False
            while True:
                line = self._input.readline()
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                if line == '':
                    break
                mo = boundaryre.match(line)
                if mo:
                    if mo.group('end'):
                        close_boundary_seen = True
                        linesep = mo.group('linesep')
                        break
                    if capturing_preamble:
                        if preamble:
                            lastline = preamble[-1]
                            eolmo = NLCRE_eol.search(lastline)
                            if eolmo:
                                preamble[-1] = lastline[:-len(eolmo.group(0))]
                            self._cur.preamble = EMPTYSTRING.join(preamble)
                        capturing_preamble = False
                        self._input.unreadline(line)
                        continue
                    while True:
                        line = self._input.readline()
                        if line is NeedMoreData:
                            yield NeedMoreData
                            continue
                        mo = boundaryre.match(line)
                        if not mo:
                            self._input.unreadline(line)
                            break
                    self._input.push_eof_matcher(boundaryre.match)
                    for retval in self._parsegen():
                        if retval is NeedMoreData:
                            yield NeedMoreData
                            continue
                        break
                    if self._last.get_content_maintype() == 'multipart':
                        epilogue = self._last.epilogue
                        if epilogue == '':
                            self._last.epilogue = None
                        elif epilogue is not None:
                            mo = NLCRE_eol.search(epilogue)
                            if mo:
                                end = len(mo.group(0))
                                self._last.epilogue = epilogue[:-end]
                    else:
                        payload = self._last._payload
                        if isinstance(payload, str):
                            mo = NLCRE_eol.search(payload)
                            if mo:
                                payload = payload[:-len(mo.group(0))]
                                self._last._payload = payload
                    self._input.pop_eof_matcher()
                    self._pop_message()
                    self._last = self._cur
                else:
                    assert capturing_preamble
                    preamble.append(line)
            if capturing_preamble:
                defect = errors.StartBoundaryNotFoundDefect()
                self.policy.handle_defect(self._cur, defect)
                self._cur.set_payload(EMPTYSTRING.join(preamble))
                epilogue = []
                for line in self._input:
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                self._cur.epilogue = EMPTYSTRING.join(epilogue)
                return
            if not close_boundary_seen:
                defect = errors.CloseBoundaryNotFoundDefect()
                self.policy.handle_defect(self._cur, defect)
                return
            if linesep:
                epilogue = ['']
            else:
                epilogue = []
            for line in self._input:
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                epilogue.append(line)
            if epilogue:
                firstline = epilogue[0]
                bolmo = NLCRE_bol.match(firstline)
                if bolmo:
                    epilogue[0] = firstline[len(bolmo.group(0)):]
            self._cur.epilogue = EMPTYSTRING.join(epilogue)
            return
        lines = []
        for line in self._input:
            if line is NeedMoreData:
                yield NeedMoreData
                continue
            lines.append(line)
        self._cur.set_payload(EMPTYSTRING.join(lines))
    def _parse_headers(self, lines):
        lastheader = ''
        lastvalue = []
        for lineno, line in enumerate(lines):
            if line[0] in ' \t':
                if not lastheader:
                    defect = errors.FirstHeaderLineIsContinuationDefect(line)
                    self.policy.handle_defect(self._cur, defect)
                    continue
                lastvalue.append(line)
                continue
            if lastheader:
                self._cur.set_raw(*self.policy.header_source_parse(lastvalue))
                lastheader, lastvalue = '', []
            if line.startswith('From '):
                if lineno == 0:
                    mo = NLCRE_eol.search(line)
                    if mo:
                        line = line[:-len(mo.group(0))]
                    self._cur.set_unixfrom(line)
                    continue
                elif lineno == len(lines) - 1:
                    self._input.unreadline(line)
                    return
                else:
                    defect = errors.MisplacedEnvelopeHeaderDefect(line)
                    self._cur.defects.append(defect)
                    continue
            i = line.find(':')
            assert i>0, "_parse_headers fed line with no : and no leading WS"
            lastheader = line[:i]
            lastvalue = [line]
        if lastheader:
            self._cur.set_raw(*self.policy.header_source_parse(lastvalue))
class BytesFeedParser(FeedParser):
    def feed(self, data):
        super().feed(data.decode('ascii', 'surrogateescape'))
