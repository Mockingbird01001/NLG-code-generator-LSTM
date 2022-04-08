
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import super
class MessageError(Exception):
class MessageParseError(MessageError):
class HeaderParseError(MessageParseError):
class BoundaryError(MessageParseError):
class MultipartConversionError(MessageError, TypeError):
class CharsetError(MessageError):
class MessageDefect(ValueError):
    def __init__(self, line=None):
        if line is not None:
            super().__init__(line)
        self.line = line
class NoBoundaryInMultipartDefect(MessageDefect):
class StartBoundaryNotFoundDefect(MessageDefect):
class CloseBoundaryNotFoundDefect(MessageDefect):
class FirstHeaderLineIsContinuationDefect(MessageDefect):
class MisplacedEnvelopeHeaderDefect(MessageDefect):
class MissingHeaderBodySeparatorDefect(MessageDefect):
MalformedHeaderDefect = MissingHeaderBodySeparatorDefect
class MultipartInvariantViolationDefect(MessageDefect):
class InvalidMultipartContentTransferEncodingDefect(MessageDefect):
class UndecodableBytesDefect(MessageDefect):
class InvalidBase64PaddingDefect(MessageDefect):
class InvalidBase64CharactersDefect(MessageDefect):
class HeaderDefect(MessageDefect):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
class InvalidHeaderDefect(HeaderDefect):
class HeaderMissingRequiredValue(HeaderDefect):
class NonPrintableDefect(HeaderDefect):
    def __init__(self, non_printables):
        super().__init__(non_printables)
        self.non_printables = non_printables
    def __str__(self):
        return ("the following ASCII non-printables found in header: "
            "{}".format(self.non_printables))
class ObsoleteHeaderDefect(HeaderDefect):
class NonASCIILocalPartDefect(HeaderDefect):
