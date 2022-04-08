
from collections import namedtuple
import datetime
import sys
import struct
PY2 = sys.version_info[0] == 2
if PY2:
    int_types = (int, long)
    _utc = None
else:
    int_types = int
    try:
        _utc = datetime.timezone.utc
    except AttributeError:
        _utc = datetime.timezone(datetime.timedelta(0))
class ExtType(namedtuple("ExtType", "code data")):
    def __new__(cls, code, data):
        if not isinstance(code, int):
            raise TypeError("code must be int")
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not 0 <= code <= 127:
            raise ValueError("code must be 0~127")
        return super(ExtType, cls).__new__(cls, code, data)
class Timestamp(object):
    __slots__ = ["seconds", "nanoseconds"]
    def __init__(self, seconds, nanoseconds=0):
        if not isinstance(seconds, int_types):
            raise TypeError("seconds must be an interger")
        if not isinstance(nanoseconds, int_types):
            raise TypeError("nanoseconds must be an integer")
        if not (0 <= nanoseconds < 10 ** 9):
            raise ValueError(
                "nanoseconds must be a non-negative integer less than 999999999."
            )
        self.seconds = seconds
        self.nanoseconds = nanoseconds
    def __repr__(self):
        return "Timestamp(seconds={0}, nanoseconds={1})".format(
            self.seconds, self.nanoseconds
        )
    def __eq__(self, other):
        if type(other) is self.__class__:
            return (
                self.seconds == other.seconds and self.nanoseconds == other.nanoseconds
            )
        return False
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return hash((self.seconds, self.nanoseconds))
    @staticmethod
    def from_bytes(b):
        if len(b) == 4:
            seconds = struct.unpack("!L", b)[0]
            nanoseconds = 0
        elif len(b) == 8:
            data64 = struct.unpack("!Q", b)[0]
            seconds = data64 & 0x00000003FFFFFFFF
            nanoseconds = data64 >> 34
        elif len(b) == 12:
            nanoseconds, seconds = struct.unpack("!Iq", b)
        else:
            raise ValueError(
                "Timestamp type can only be created from 32, 64, or 96-bit byte objects"
            )
        return Timestamp(seconds, nanoseconds)
    def to_bytes(self):
        if (self.seconds >> 34) == 0:
            data64 = self.nanoseconds << 34 | self.seconds
            if data64 & 0xFFFFFFFF00000000 == 0:
                data = struct.pack("!L", data64)
            else:
                data = struct.pack("!Q", data64)
        else:
            data = struct.pack("!Iq", self.nanoseconds, self.seconds)
        return data
    @staticmethod
    def from_unix(unix_sec):
        seconds = int(unix_sec // 1)
        nanoseconds = int((unix_sec % 1) * 10 ** 9)
        return Timestamp(seconds, nanoseconds)
    def to_unix(self):
        return self.seconds + self.nanoseconds / 1e9
    @staticmethod
    def from_unix_nano(unix_ns):
        return Timestamp(*divmod(unix_ns, 10 ** 9))
    def to_unix_nano(self):
        return self.seconds * 10 ** 9 + self.nanoseconds
    def to_datetime(self):
        return datetime.datetime.fromtimestamp(0, _utc) + datetime.timedelta(
            seconds=self.to_unix()
        )
    @staticmethod
    def from_datetime(dt):
        return Timestamp.from_unix(dt.timestamp())
