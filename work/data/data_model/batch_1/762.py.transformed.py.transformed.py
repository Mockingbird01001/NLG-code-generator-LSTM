
import binascii
import calendar
import datetime
import hashlib
import os
import random
import socket
import struct
import threading
import time
from bson.errors import InvalidId
from bson.py3compat import PY3, bytes_from_hex, string_type, text_type
from bson.tz_util import utc
def _machine_bytes():
    machine_hash = hashlib.md5()
    if PY3:
        machine_hash.update(socket.gethostname().encode())
    else:
        machine_hash.update(socket.gethostname())
    return machine_hash.digest()[0:3]
def _raise_invalid_id(oid):
    raise InvalidId(
        "%r is not a valid ObjectId, it must be a 12-byte input"
        " or a 24-character hex string" % oid)
class ObjectId(object):
    _inc = random.randint(0, 0xFFFFFF)
    _inc_lock = threading.Lock()
    _machine_bytes = _machine_bytes()
    __slots__ = ('__id')
    _type_marker = 7
    def __init__(self, oid=None):
        if oid is None:
            self.__generate()
        elif isinstance(oid, bytes) and len(oid) == 12:
            self.__id = oid
        else:
            self.__validate(oid)
    @classmethod
    def from_datetime(cls, generation_time):
        if generation_time.utcoffset() is not None:
            generation_time = generation_time - generation_time.utcoffset()
        timestamp = calendar.timegm(generation_time.timetuple())
        oid = struct.pack(
            ">i", int(timestamp)) + b"\x00\x00\x00\x00\x00\x00\x00\x00"
        return cls(oid)
    @classmethod
    def is_valid(cls, oid):
        if not oid:
            return False
        try:
            ObjectId(oid)
            return True
        except (InvalidId, TypeError):
            return False
    def __generate(self):
        oid = struct.pack(">i", int(time.time()))
        oid += ObjectId._machine_bytes
        oid += struct.pack(">H", os.getpid() % 0xFFFF)
        with ObjectId._inc_lock:
            oid += struct.pack(">i", ObjectId._inc)[1:4]
            ObjectId._inc = (ObjectId._inc + 1) % 0xFFFFFF
        self.__id = oid
    def __validate(self, oid):
        if isinstance(oid, ObjectId):
            self.__id = oid.binary
        elif isinstance(oid, string_type):
            if len(oid) == 24:
                try:
                    self.__id = bytes_from_hex(oid)
                except (TypeError, ValueError):
                    _raise_invalid_id(oid)
            else:
                _raise_invalid_id(oid)
        else:
            raise TypeError("id must be an instance of (bytes, %s, ObjectId), "
                            "not %s" % (text_type.__name__, type(oid)))
    @property
    def binary(self):
        return self.__id
    @property
    def generation_time(self):
        timestamp = struct.unpack(">i", self.__id[0:4])[0]
        return datetime.datetime.fromtimestamp(timestamp, utc)
    def __getstate__(self):
        return self.__id
    def __setstate__(self, value):
        if isinstance(value, dict):
            oid = value["_ObjectId__id"]
        else:
            oid = value
        if PY3 and isinstance(oid, text_type):
            self.__id = oid.encode('latin-1')
        else:
            self.__id = oid
    def __str__(self):
        if PY3:
            return binascii.hexlify(self.__id).decode()
        return binascii.hexlify(self.__id)
    def __repr__(self):
        return "ObjectId('%s')" % (str(self),)
    def __eq__(self, other):
        if isinstance(other, ObjectId):
            return self.__id == other.binary
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, ObjectId):
            return self.__id != other.binary
        return NotImplemented
    def __lt__(self, other):
        if isinstance(other, ObjectId):
            return self.__id < other.binary
        return NotImplemented
    def __le__(self, other):
        if isinstance(other, ObjectId):
            return self.__id <= other.binary
        return NotImplemented
    def __gt__(self, other):
        if isinstance(other, ObjectId):
            return self.__id > other.binary
        return NotImplemented
    def __ge__(self, other):
        if isinstance(other, ObjectId):
            return self.__id >= other.binary
        return NotImplemented
    def __hash__(self):
        return hash(self.__id)
