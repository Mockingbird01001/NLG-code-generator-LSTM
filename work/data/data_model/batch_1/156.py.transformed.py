
import decimal
import struct
import sys
from bson.py3compat import (PY3 as _PY3,
                            string_type as _string_type)
if _PY3:
    _from_bytes = int.from_bytes
else:
    import binascii
    def _from_bytes(value, dummy, _int=int, _hexlify=binascii.hexlify):
        return _int(_hexlify(value), 16)
if sys.version_info[:2] == (2, 6):
    def _bit_length(num):
        if num:
            return len(bin(num)) - 2
        return 0
else:
    def _bit_length(num):
        return num.bit_length()
_PACK_64 = struct.Struct("<Q").pack
_UNPACK_64 = struct.Struct("<Q").unpack
_EXPONENT_MASK = 3 << 61
_EXPONENT_BIAS = 6176
_EXPONENT_MAX = 6144
_EXPONENT_MIN = -6143
_MAX_DIGITS = 34
_INF = 0x7800000000000000
_NAN = 0x7c00000000000000
_SNAN = 0x7e00000000000000
_SIGN = 0x8000000000000000
_NINF = (_INF + _SIGN, 0)
_PINF = (_INF, 0)
_NNAN = (_NAN + _SIGN, 0)
_PNAN = (_NAN, 0)
_NSNAN = (_SNAN + _SIGN, 0)
_PSNAN = (_SNAN, 0)
_CTX_OPTIONS = {
    'prec': _MAX_DIGITS,
    'rounding': decimal.ROUND_HALF_EVEN,
    'Emin': _EXPONENT_MIN,
    'Emax': _EXPONENT_MAX,
    'capitals': 1,
    'flags': [],
    'traps': [decimal.InvalidOperation,
              decimal.Overflow,
              decimal.Inexact]
}
try:
    decimal.Context(clamp=1)
    _CTX_OPTIONS['clamp'] = 1
except TypeError:
    _CTX_OPTIONS['_clamp'] = 1
_DEC128_CTX = decimal.Context(**_CTX_OPTIONS.copy())
def create_decimal128_context():
    opts = _CTX_OPTIONS.copy()
    opts['traps'] = []
    return decimal.Context(**opts)
def _decimal_to_128(value):
    with decimal.localcontext(_DEC128_CTX) as ctx:
        value = ctx.create_decimal(value)
    if value.is_infinite():
        return _NINF if value.is_signed() else _PINF
    sign, digits, exponent = value.as_tuple()
    if value.is_nan():
        if digits:
            raise ValueError("NaN with debug payload is not supported")
        if value.is_snan():
            return _NSNAN if value.is_signed() else _PSNAN
        return _NNAN if value.is_signed() else _PNAN
    significand = int("".join([str(digit) for digit in digits]))
    bit_length = _bit_length(significand)
    high = 0
    low = 0
    for i in range(min(64, bit_length)):
        if significand & (1 << i):
            low |= 1 << i
    for i in range(64, bit_length):
        if significand & (1 << i):
            high |= 1 << (i - 64)
    biased_exponent = exponent + _EXPONENT_BIAS
    if high >> 49 == 1:
        high = high & 0x7fffffffffff
        high |= _EXPONENT_MASK
        high |= (biased_exponent & 0x3fff) << 47
    else:
        high |= biased_exponent << 49
    if sign:
        high |= _SIGN
    return high, low
class Decimal128(object):
    __slots__ = ('__high', '__low')
    _type_marker = 19
    def __init__(self, value):
        if isinstance(value, (_string_type, decimal.Decimal)):
            self.__high, self.__low = _decimal_to_128(value)
        elif isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError('Invalid size for creation of Decimal128 '
                                 'from list or tuple. Must have exactly 2 '
                                 'elements.')
            self.__high, self.__low = value
        else:
            raise TypeError("Cannot convert %r to Decimal128" % (value,))
    def to_decimal(self):
        high = self.__high
        low = self.__low
        sign = 1 if (high & _SIGN) else 0
        if (high & _SNAN) == _SNAN:
            return decimal.Decimal((sign, (), 'N'))
        elif (high & _NAN) == _NAN:
            return decimal.Decimal((sign, (), 'n'))
        elif (high & _INF) == _INF:
            return decimal.Decimal((sign, (), 'F'))
        if (high & _EXPONENT_MASK) == _EXPONENT_MASK:
            exponent = ((high & 0x1fffe00000000000) >> 47) - _EXPONENT_BIAS
            return decimal.Decimal((sign, (0,), exponent))
        else:
            exponent = ((high & 0x7fff800000000000) >> 49) - _EXPONENT_BIAS
        arr = bytearray(15)
        mask = 0x00000000000000ff
        for i in range(14, 6, -1):
            arr[i] = (low & mask) >> ((14 - i) << 3)
            mask = mask << 8
        mask = 0x00000000000000ff
        for i in range(6, 0, -1):
            arr[i] = (high & mask) >> ((6 - i) << 3)
            mask = mask << 8
        mask = 0x0001000000000000
        arr[0] = (high & mask) >> 48
        digits = tuple(
            int(digit) for digit in str(_from_bytes(bytes(arr), 'big')))
        with decimal.localcontext(_DEC128_CTX) as ctx:
            return ctx.create_decimal((sign, digits, exponent))
    @classmethod
    def from_bid(cls, value):
        if not isinstance(value, bytes):
            raise TypeError("value must be an instance of bytes")
        if len(value) != 16:
            raise ValueError("value must be exactly 16 bytes")
        return cls((_UNPACK_64(value[8:])[0], _UNPACK_64(value[:8])[0]))
    @property
    def bid(self):
        return _PACK_64(self.__low) + _PACK_64(self.__high)
    def __str__(self):
        dec = self.to_decimal()
        if dec.is_nan():
            return "NaN"
        return str(dec)
    def __repr__(self):
        return "Decimal128('%s')" % (str(self),)
    def __setstate__(self, value):
        self.__high, self.__low = value
    def __getstate__(self):
        return self.__high, self.__low
    def __eq__(self, other):
        if isinstance(other, Decimal128):
            return self.bid == other.bid
        return NotImplemented
    def __ne__(self, other):
        return not self == other
