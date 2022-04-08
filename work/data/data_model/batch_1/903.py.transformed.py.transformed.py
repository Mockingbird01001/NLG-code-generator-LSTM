
import re
from bson.son import RE_TYPE
from bson.py3compat import string_type, text_type
def str_flags_to_int(str_flags):
    flags = 0
    if "i" in str_flags:
        flags |= re.IGNORECASE
    if "l" in str_flags:
        flags |= re.LOCALE
    if "m" in str_flags:
        flags |= re.MULTILINE
    if "s" in str_flags:
        flags |= re.DOTALL
    if "u" in str_flags:
        flags |= re.UNICODE
    if "x" in str_flags:
        flags |= re.VERBOSE
    return flags
class Regex(object):
    _type_marker = 11
    @classmethod
    def from_native(cls, regex):
        if not isinstance(regex, RE_TYPE):
            raise TypeError(
                "regex must be a compiled regular expression, not %s"
                % type(regex))
        return Regex(regex.pattern, regex.flags)
    def __init__(self, pattern, flags=0):
        if not isinstance(pattern, (text_type, bytes)):
            raise TypeError("pattern must be a string, not %s" % type(pattern))
        self.pattern = pattern
        if isinstance(flags, string_type):
            self.flags = str_flags_to_int(flags)
        elif isinstance(flags, int):
            self.flags = flags
        else:
            raise TypeError(
                "flags must be a string or int, not %s" % type(flags))
    def __eq__(self, other):
        if isinstance(other, Regex):
            return self.pattern == self.pattern and self.flags == other.flags
        else:
            return NotImplemented
    __hash__ = None
    def __ne__(self, other):
        return not self == other
    def __repr__(self):
        return "Regex(%r, %r)" % (self.pattern, self.flags)
    def try_compile(self):
        return re.compile(self.pattern, self.flags)
