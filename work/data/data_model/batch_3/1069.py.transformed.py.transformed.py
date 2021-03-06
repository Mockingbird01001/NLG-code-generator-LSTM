
from __future__ import print_function
__version__ = "$Revision$"
version     = "0.9.0"
__author__  = "Lars Gust\u00e4bel (lars@gustaebel.de)"
__date__    = "$Date: 2011-02-25 17:42:01 +0200 (Fri, 25 Feb 2011) $"
__cvsid__   = "$Id: tarfile.py 88586 2011-02-25 15:42:01Z marc-andre.lemburg $"
__credits__ = "Gustavo Niemeyer, Niels Gust\u00e4bel, Richard Townsend."
import sys
import os
import stat
import errno
import time
import struct
import copy
import re
try:
    import grp, pwd
except ImportError:
    grp = pwd = None
symlink_exception = (AttributeError, NotImplementedError)
try:
    symlink_exception += (WindowsError,)
except NameError:
    pass
__all__ = ["TarFile", "TarInfo", "is_tarfile", "TarError"]
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
_open = builtins.open
NUL = b"\0"
BLOCKSIZE = 512
RECORDSIZE = BLOCKSIZE * 20
GNU_MAGIC = b"ustar  \0"
POSIX_MAGIC = b"ustar\x0000"
LENGTH_NAME = 100
LENGTH_LINK = 100
LENGTH_PREFIX = 155
REGTYPE = b"0"
AREGTYPE = b"\0"
LNKTYPE = b"1"
SYMTYPE = b"2"
CHRTYPE = b"3"
BLKTYPE = b"4"
DIRTYPE = b"5"
FIFOTYPE = b"6"
CONTTYPE = b"7"
GNUTYPE_LONGNAME = b"L"
GNUTYPE_LONGLINK = b"K"
GNUTYPE_SPARSE = b"S"
XHDTYPE = b"x"
XGLTYPE = b"g"
SOLARIS_XHDTYPE = b"X"
USTAR_FORMAT = 0
GNU_FORMAT = 1
PAX_FORMAT = 2
DEFAULT_FORMAT = GNU_FORMAT
SUPPORTED_TYPES = (REGTYPE, AREGTYPE, LNKTYPE,
                   SYMTYPE, DIRTYPE, FIFOTYPE,
                   CONTTYPE, CHRTYPE, BLKTYPE,
                   GNUTYPE_LONGNAME, GNUTYPE_LONGLINK,
                   GNUTYPE_SPARSE)
REGULAR_TYPES = (REGTYPE, AREGTYPE,
                 CONTTYPE, GNUTYPE_SPARSE)
GNU_TYPES = (GNUTYPE_LONGNAME, GNUTYPE_LONGLINK,
             GNUTYPE_SPARSE)
PAX_FIELDS = ("path", "linkpath", "size", "mtime",
              "uid", "gid", "uname", "gname")
PAX_NAME_FIELDS = set(("path", "linkpath", "uname", "gname"))
PAX_NUMBER_FIELDS = {
    "atime": float,
    "ctime": float,
    "mtime": float,
    "uid": int,
    "gid": int,
    "size": int
}
S_IFLNK = 0o120000
S_IFREG = 0o100000
S_IFBLK = 0o060000
S_IFDIR = 0o040000
S_IFCHR = 0o020000
S_IFIFO = 0o010000
TSUID   = 0o4000
TSGID   = 0o2000
TSVTX   = 0o1000
TUREAD  = 0o400
TUWRITE = 0o200
TUEXEC  = 0o100
TGREAD  = 0o040
TGWRITE = 0o020
TGEXEC  = 0o010
TOREAD  = 0o004
TOWRITE = 0o002
TOEXEC  = 0o001
if os.name in ("nt", "ce"):
    ENCODING = "utf-8"
else:
    ENCODING = sys.getfilesystemencoding()
def stn(s, length, encoding, errors):
    s = s.encode(encoding, errors)
    return s[:length] + (length - len(s)) * NUL
def nts(s, encoding, errors):
    p = s.find(b"\0")
    if p != -1:
        s = s[:p]
    return s.decode(encoding, errors)
def nti(s):
    if s[0] != chr(0o200):
        try:
            n = int(nts(s, "ascii", "strict") or "0", 8)
        except ValueError:
            raise InvalidHeaderError("invalid header")
    else:
        n = 0
        for i in range(len(s) - 1):
            n <<= 8
            n += ord(s[i + 1])
    return n
def itn(n, digits=8, format=DEFAULT_FORMAT):
    if 0 <= n < 8 ** (digits - 1):
        s = ("%0*o" % (digits - 1, n)).encode("ascii") + NUL
    else:
        if format != GNU_FORMAT or n >= 256 ** (digits - 1):
            raise ValueError("overflow in number field")
        if n < 0:
            n = struct.unpack("L", struct.pack("l", n))[0]
        s = bytearray()
        for i in range(digits - 1):
            s.insert(0, n & 0o377)
            n >>= 8
        s.insert(0, 0o200)
    return s
def calc_chksums(buf):
    unsigned_chksum = 256 + sum(struct.unpack("148B", buf[:148]) + struct.unpack("356B", buf[156:512]))
    signed_chksum = 256 + sum(struct.unpack("148b", buf[:148]) + struct.unpack("356b", buf[156:512]))
    return unsigned_chksum, signed_chksum
def copyfileobj(src, dst, length=None):
    if length == 0:
        return
    if length is None:
        while True:
            buf = src.read(16*1024)
            if not buf:
                break
            dst.write(buf)
        return
    BUFSIZE = 16 * 1024
    blocks, remainder = divmod(length, BUFSIZE)
    for b in range(blocks):
        buf = src.read(BUFSIZE)
        if len(buf) < BUFSIZE:
            raise IOError("end of file reached")
        dst.write(buf)
    if remainder != 0:
        buf = src.read(remainder)
        if len(buf) < remainder:
            raise IOError("end of file reached")
        dst.write(buf)
    return
filemode_table = (
    ((S_IFLNK,      "l"),
     (S_IFREG,      "-"),
     (S_IFBLK,      "b"),
     (S_IFDIR,      "d"),
     (S_IFCHR,      "c"),
     (S_IFIFO,      "p")),
    ((TUREAD,       "r"),),
    ((TUWRITE,      "w"),),
    ((TUEXEC|TSUID, "s"),
     (TSUID,        "S"),
     (TUEXEC,       "x")),
    ((TGREAD,       "r"),),
    ((TGWRITE,      "w"),),
    ((TGEXEC|TSGID, "s"),
     (TSGID,        "S"),
     (TGEXEC,       "x")),
    ((TOREAD,       "r"),),
    ((TOWRITE,      "w"),),
    ((TOEXEC|TSVTX, "t"),
     (TSVTX,        "T"),
     (TOEXEC,       "x"))
)
def filemode(mode):
    perm = []
    for table in filemode_table:
        for bit, char in table:
            if mode & bit == bit:
                perm.append(char)
                break
        else:
            perm.append("-")
    return "".join(perm)
class TarError(Exception):
    pass
class ExtractError(TarError):
    pass
class ReadError(TarError):
    pass
class CompressionError(TarError):
    pass
class StreamError(TarError):
    pass
class HeaderError(TarError):
    pass
class EmptyHeaderError(HeaderError):
    pass
class TruncatedHeaderError(HeaderError):
    pass
class EOFHeaderError(HeaderError):
    pass
class InvalidHeaderError(HeaderError):
    pass
class SubsequentHeaderError(HeaderError):
    pass
class _LowLevelFile(object):
    def __init__(self, name, mode):
        mode = {
            "r": os.O_RDONLY,
            "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        }[mode]
        if hasattr(os, "O_BINARY"):
            mode |= os.O_BINARY
        self.fd = os.open(name, mode, 0o666)
    def close(self):
        os.close(self.fd)
    def read(self, size):
        return os.read(self.fd, size)
    def write(self, s):
        os.write(self.fd, s)
class _Stream(object):
    def __init__(self, name, mode, comptype, fileobj, bufsize):
        self._extfileobj = True
        if fileobj is None:
            fileobj = _LowLevelFile(name, mode)
            self._extfileobj = False
        if comptype == '*':
            fileobj = _StreamProxy(fileobj)
            comptype = fileobj.getcomptype()
        self.name     = name or ""
        self.mode     = mode
        self.comptype = comptype
        self.fileobj  = fileobj
        self.bufsize  = bufsize
        self.buf      = b""
        self.pos      = 0
        self.closed   = False
        try:
            if comptype == "gz":
                try:
                    import zlib
                except ImportError:
                    raise CompressionError("zlib module is not available")
                self.zlib = zlib
                self.crc = zlib.crc32(b"")
                if mode == "r":
                    self._init_read_gz()
                else:
                    self._init_write_gz()
            if comptype == "bz2":
                try:
                    import bz2
                except ImportError:
                    raise CompressionError("bz2 module is not available")
                if mode == "r":
                    self.dbuf = b""
                    self.cmp = bz2.BZ2Decompressor()
                else:
                    self.cmp = bz2.BZ2Compressor()
        except:
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True
            raise
    def __del__(self):
        if hasattr(self, "closed") and not self.closed:
            self.close()
    def _init_write_gz(self):
        self.cmp = self.zlib.compressobj(9, self.zlib.DEFLATED,
                                            -self.zlib.MAX_WBITS,
                                            self.zlib.DEF_MEM_LEVEL,
                                            0)
        timestamp = struct.pack("<L", int(time.time()))
        self.__write(b"\037\213\010\010" + timestamp + b"\002\377")
        if self.name.endswith(".gz"):
            self.name = self.name[:-3]
        self.__write(self.name.encode("iso-8859-1", "replace") + NUL)
    def write(self, s):
        if self.comptype == "gz":
            self.crc = self.zlib.crc32(s, self.crc)
        self.pos += len(s)
        if self.comptype != "tar":
            s = self.cmp.compress(s)
        self.__write(s)
    def __write(self, s):
        self.buf += s
        while len(self.buf) > self.bufsize:
            self.fileobj.write(self.buf[:self.bufsize])
            self.buf = self.buf[self.bufsize:]
    def close(self):
        if self.closed:
            return
        if self.mode == "w" and self.comptype != "tar":
            self.buf += self.cmp.flush()
        if self.mode == "w" and self.buf:
            self.fileobj.write(self.buf)
            self.buf = b""
            if self.comptype == "gz":
                self.fileobj.write(struct.pack("<L", self.crc & 0xffffffff))
                self.fileobj.write(struct.pack("<L", self.pos & 0xffffFFFF))
        if not self._extfileobj:
            self.fileobj.close()
        self.closed = True
    def _init_read_gz(self):
        self.cmp = self.zlib.decompressobj(-self.zlib.MAX_WBITS)
        self.dbuf = b""
        if self.__read(2) != b"\037\213":
            raise ReadError("not a gzip file")
        if self.__read(1) != b"\010":
            raise CompressionError("unsupported compression method")
        flag = ord(self.__read(1))
        self.__read(6)
        if flag & 4:
            xlen = ord(self.__read(1)) + 256 * ord(self.__read(1))
            self.read(xlen)
        if flag & 8:
            while True:
                s = self.__read(1)
                if not s or s == NUL:
                    break
        if flag & 16:
            while True:
                s = self.__read(1)
                if not s or s == NUL:
                    break
        if flag & 2:
            self.__read(2)
    def tell(self):
        return self.pos
    def seek(self, pos=0):
        if pos - self.pos >= 0:
            blocks, remainder = divmod(pos - self.pos, self.bufsize)
            for i in range(blocks):
                self.read(self.bufsize)
            self.read(remainder)
        else:
            raise StreamError("seeking backwards is not allowed")
        return self.pos
    def read(self, size=None):
        if size is None:
            t = []
            while True:
                buf = self._read(self.bufsize)
                if not buf:
                    break
                t.append(buf)
            buf = "".join(t)
        else:
            buf = self._read(size)
        self.pos += len(buf)
        return buf
    def _read(self, size):
        if self.comptype == "tar":
            return self.__read(size)
        c = len(self.dbuf)
        while c < size:
            buf = self.__read(self.bufsize)
            if not buf:
                break
            try:
                buf = self.cmp.decompress(buf)
            except IOError:
                raise ReadError("invalid compressed data")
            self.dbuf += buf
            c += len(buf)
        buf = self.dbuf[:size]
        self.dbuf = self.dbuf[size:]
        return buf
    def __read(self, size):
        c = len(self.buf)
        while c < size:
            buf = self.fileobj.read(self.bufsize)
            if not buf:
                break
            self.buf += buf
            c += len(buf)
        buf = self.buf[:size]
        self.buf = self.buf[size:]
        return buf
class _StreamProxy(object):
    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.buf = self.fileobj.read(BLOCKSIZE)
    def read(self, size):
        self.read = self.fileobj.read
        return self.buf
    def getcomptype(self):
        if self.buf.startswith(b"\037\213\010"):
            return "gz"
        if self.buf.startswith(b"BZh91"):
            return "bz2"
        return "tar"
    def close(self):
        self.fileobj.close()
class _BZ2Proxy(object):
    blocksize = 16 * 1024
    def __init__(self, fileobj, mode):
        self.fileobj = fileobj
        self.mode = mode
        self.name = getattr(self.fileobj, "name", None)
        self.init()
    def init(self):
        import bz2
        self.pos = 0
        if self.mode == "r":
            self.bz2obj = bz2.BZ2Decompressor()
            self.fileobj.seek(0)
            self.buf = b""
        else:
            self.bz2obj = bz2.BZ2Compressor()
    def read(self, size):
        x = len(self.buf)
        while x < size:
            raw = self.fileobj.read(self.blocksize)
            if not raw:
                break
            data = self.bz2obj.decompress(raw)
            self.buf += data
            x += len(data)
        buf = self.buf[:size]
        self.buf = self.buf[size:]
        self.pos += len(buf)
        return buf
    def seek(self, pos):
        if pos < self.pos:
            self.init()
        self.read(pos - self.pos)
    def tell(self):
        return self.pos
    def write(self, data):
        self.pos += len(data)
        raw = self.bz2obj.compress(data)
        self.fileobj.write(raw)
    def close(self):
        if self.mode == "w":
            raw = self.bz2obj.flush()
            self.fileobj.write(raw)
class _FileInFile(object):
    def __init__(self, fileobj, offset, size, blockinfo=None):
        self.fileobj = fileobj
        self.offset = offset
        self.size = size
        self.position = 0
        if blockinfo is None:
            blockinfo = [(0, size)]
        self.map_index = 0
        self.map = []
        lastpos = 0
        realpos = self.offset
        for offset, size in blockinfo:
            if offset > lastpos:
                self.map.append((False, lastpos, offset, None))
            self.map.append((True, offset, offset + size, realpos))
            realpos += size
            lastpos = offset + size
        if lastpos < self.size:
            self.map.append((False, lastpos, self.size, None))
    def seekable(self):
        if not hasattr(self.fileobj, "seekable"):
            return True
        return self.fileobj.seekable()
    def tell(self):
        return self.position
    def seek(self, position):
        self.position = position
    def read(self, size=None):
        if size is None:
            size = self.size - self.position
        else:
            size = min(size, self.size - self.position)
        buf = b""
        while size > 0:
            while True:
                data, start, stop, offset = self.map[self.map_index]
                if start <= self.position < stop:
                    break
                else:
                    self.map_index += 1
                    if self.map_index == len(self.map):
                        self.map_index = 0
            length = min(size, stop - self.position)
            if data:
                self.fileobj.seek(offset + (self.position - start))
                buf += self.fileobj.read(length)
            else:
                buf += NUL * length
            size -= length
            self.position += length
        return buf
class ExFileObject(object):
    blocksize = 1024
    def __init__(self, tarfile, tarinfo):
        self.fileobj = _FileInFile(tarfile.fileobj,
                                   tarinfo.offset_data,
                                   tarinfo.size,
                                   tarinfo.sparse)
        self.name = tarinfo.name
        self.mode = "r"
        self.closed = False
        self.size = tarinfo.size
        self.position = 0
        self.buffer = b""
    def readable(self):
        return True
    def writable(self):
        return False
    def seekable(self):
        return self.fileobj.seekable()
    def read(self, size=None):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        buf = b""
        if self.buffer:
            if size is None:
                buf = self.buffer
                self.buffer = b""
            else:
                buf = self.buffer[:size]
                self.buffer = self.buffer[size:]
        if size is None:
            buf += self.fileobj.read()
        else:
            buf += self.fileobj.read(size - len(buf))
        self.position += len(buf)
        return buf
    read1 = read
    def readline(self, size=-1):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        pos = self.buffer.find(b"\n") + 1
        if pos == 0:
            while True:
                buf = self.fileobj.read(self.blocksize)
                self.buffer += buf
                if not buf or b"\n" in buf:
                    pos = self.buffer.find(b"\n") + 1
                    if pos == 0:
                        pos = len(self.buffer)
                    break
        if size != -1:
            pos = min(size, pos)
        buf = self.buffer[:pos]
        self.buffer = self.buffer[pos:]
        self.position += len(buf)
        return buf
    def readlines(self):
        result = []
        while True:
            line = self.readline()
            if not line: break
            result.append(line)
        return result
    def tell(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return self.position
    def seek(self, pos, whence=os.SEEK_SET):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if whence == os.SEEK_SET:
            self.position = min(max(pos, 0), self.size)
        elif whence == os.SEEK_CUR:
            if pos < 0:
                self.position = max(self.position + pos, 0)
            else:
                self.position = min(self.position + pos, self.size)
        elif whence == os.SEEK_END:
            self.position = max(min(self.size + pos, self.size), 0)
        else:
            raise ValueError("Invalid argument")
        self.buffer = b""
        self.fileobj.seek(self.position)
    def close(self):
        self.closed = True
    def __iter__(self):
        while True:
            line = self.readline()
            if not line:
                break
            yield line
class TarInfo(object):
    __slots__ = ("name", "mode", "uid", "gid", "size", "mtime",
                 "chksum", "type", "linkname", "uname", "gname",
                 "devmajor", "devminor",
                 "offset", "offset_data", "pax_headers", "sparse",
                 "tarfile", "_sparse_structs", "_link_target")
    def __init__(self, name=""):
        self.name = name
        self.mode = 0o644
        self.uid = 0
        self.gid = 0
        self.size = 0
        self.mtime = 0
        self.chksum = 0
        self.type = REGTYPE
        self.linkname = ""
        self.uname = ""
        self.gname = ""
        self.devmajor = 0
        self.devminor = 0
        self.offset = 0
        self.offset_data = 0
        self.sparse = None
        self.pax_headers = {}
    def _getpath(self):
        return self.name
    def _setpath(self, name):
        self.name = name
    path = property(_getpath, _setpath)
    def _getlinkpath(self):
        return self.linkname
    def _setlinkpath(self, linkname):
        self.linkname = linkname
    linkpath = property(_getlinkpath, _setlinkpath)
    def __repr__(self):
    def get_info(self):
        info = {
            "name":     self.name,
            "mode":     self.mode & 0o7777,
            "uid":      self.uid,
            "gid":      self.gid,
            "size":     self.size,
            "mtime":    self.mtime,
            "chksum":   self.chksum,
            "type":     self.type,
            "linkname": self.linkname,
            "uname":    self.uname,
            "gname":    self.gname,
            "devmajor": self.devmajor,
            "devminor": self.devminor
        }
        if info["type"] == DIRTYPE and not info["name"].endswith("/"):
            info["name"] += "/"
        return info
    def tobuf(self, format=DEFAULT_FORMAT, encoding=ENCODING, errors="surrogateescape"):
        info = self.get_info()
        if format == USTAR_FORMAT:
            return self.create_ustar_header(info, encoding, errors)
        elif format == GNU_FORMAT:
            return self.create_gnu_header(info, encoding, errors)
        elif format == PAX_FORMAT:
            return self.create_pax_header(info, encoding)
        else:
            raise ValueError("invalid format")
    def create_ustar_header(self, info, encoding, errors):
        info["magic"] = POSIX_MAGIC
        if len(info["linkname"]) > LENGTH_LINK:
            raise ValueError("linkname is too long")
        if len(info["name"]) > LENGTH_NAME:
            info["prefix"], info["name"] = self._posix_split_name(info["name"])
        return self._create_header(info, USTAR_FORMAT, encoding, errors)
    def create_gnu_header(self, info, encoding, errors):
        info["magic"] = GNU_MAGIC
        buf = b""
        if len(info["linkname"]) > LENGTH_LINK:
            buf += self._create_gnu_long_header(info["linkname"], GNUTYPE_LONGLINK, encoding, errors)
        if len(info["name"]) > LENGTH_NAME:
            buf += self._create_gnu_long_header(info["name"], GNUTYPE_LONGNAME, encoding, errors)
        return buf + self._create_header(info, GNU_FORMAT, encoding, errors)
    def create_pax_header(self, info, encoding):
        info["magic"] = POSIX_MAGIC
        pax_headers = self.pax_headers.copy()
        for name, hname, length in (
                ("name", "path", LENGTH_NAME), ("linkname", "linkpath", LENGTH_LINK),
                ("uname", "uname", 32), ("gname", "gname", 32)):
            if hname in pax_headers:
                continue
            try:
                info[name].encode("ascii", "strict")
            except UnicodeEncodeError:
                pax_headers[hname] = info[name]
                continue
            if len(info[name]) > length:
                pax_headers[hname] = info[name]
        for name, digits in (("uid", 8), ("gid", 8), ("size", 12), ("mtime", 12)):
            if name in pax_headers:
                info[name] = 0
                continue
            val = info[name]
            if not 0 <= val < 8 ** (digits - 1) or isinstance(val, float):
                pax_headers[name] = str(val)
                info[name] = 0
        if pax_headers:
            buf = self._create_pax_generic_header(pax_headers, XHDTYPE, encoding)
        else:
            buf = b""
        return buf + self._create_header(info, USTAR_FORMAT, "ascii", "replace")
    @classmethod
    def create_pax_global_header(cls, pax_headers):
        return cls._create_pax_generic_header(pax_headers, XGLTYPE, "utf8")
    def _posix_split_name(self, name):
        prefix = name[:LENGTH_PREFIX + 1]
        while prefix and prefix[-1] != "/":
            prefix = prefix[:-1]
        name = name[len(prefix):]
        prefix = prefix[:-1]
        if not prefix or len(name) > LENGTH_NAME:
            raise ValueError("name is too long")
        return prefix, name
    @staticmethod
    def _create_header(info, format, encoding, errors):
        parts = [
            stn(info.get("name", ""), 100, encoding, errors),
            itn(info.get("mode", 0) & 0o7777, 8, format),
            itn(info.get("uid", 0), 8, format),
            itn(info.get("gid", 0), 8, format),
            itn(info.get("size", 0), 12, format),
            itn(info.get("mtime", 0), 12, format),
            b"        ",
            info.get("type", REGTYPE),
            stn(info.get("linkname", ""), 100, encoding, errors),
            info.get("magic", POSIX_MAGIC),
            stn(info.get("uname", ""), 32, encoding, errors),
            stn(info.get("gname", ""), 32, encoding, errors),
            itn(info.get("devmajor", 0), 8, format),
            itn(info.get("devminor", 0), 8, format),
            stn(info.get("prefix", ""), 155, encoding, errors)
        ]
        buf = struct.pack("%ds" % BLOCKSIZE, b"".join(parts))
        chksum = calc_chksums(buf[-BLOCKSIZE:])[0]
        buf = buf[:-364] + ("%06o\0" % chksum).encode("ascii") + buf[-357:]
        return buf
    @staticmethod
    def _create_payload(payload):
        blocks, remainder = divmod(len(payload), BLOCKSIZE)
        if remainder > 0:
            payload += (BLOCKSIZE - remainder) * NUL
        return payload
    @classmethod
    def _create_gnu_long_header(cls, name, type, encoding, errors):
        name = name.encode(encoding, errors) + NUL
        info = {}
        info["name"] = "././@LongLink"
        info["type"] = type
        info["size"] = len(name)
        info["magic"] = GNU_MAGIC
        return cls._create_header(info, USTAR_FORMAT, encoding, errors) +                cls._create_payload(name)
    @classmethod
    def _create_pax_generic_header(cls, pax_headers, type, encoding):
        binary = False
        for keyword, value in pax_headers.items():
            try:
                value.encode("utf8", "strict")
            except UnicodeEncodeError:
                binary = True
                break
        records = b""
        if binary:
            records += b"21 hdrcharset=BINARY\n"
        for keyword, value in pax_headers.items():
            keyword = keyword.encode("utf8")
            if binary:
                value = value.encode(encoding, "surrogateescape")
            else:
                value = value.encode("utf8")
            l = len(keyword) + len(value) + 3
            n = p = 0
            while True:
                n = l + len(str(p))
                if n == p:
                    break
                p = n
            records += bytes(str(p), "ascii") + b" " + keyword + b"=" + value + b"\n"
        info = {}
        info["name"] = "././@PaxHeader"
        info["type"] = type
        info["size"] = len(records)
        info["magic"] = POSIX_MAGIC
        return cls._create_header(info, USTAR_FORMAT, "ascii", "replace") +                cls._create_payload(records)
    @classmethod
    def frombuf(cls, buf, encoding, errors):
        if len(buf) == 0:
            raise EmptyHeaderError("empty header")
        if len(buf) != BLOCKSIZE:
            raise TruncatedHeaderError("truncated header")
        if buf.count(NUL) == BLOCKSIZE:
            raise EOFHeaderError("end of file header")
        chksum = nti(buf[148:156])
        if chksum not in calc_chksums(buf):
            raise InvalidHeaderError("bad checksum")
        obj = cls()
        obj.name = nts(buf[0:100], encoding, errors)
        obj.mode = nti(buf[100:108])
        obj.uid = nti(buf[108:116])
        obj.gid = nti(buf[116:124])
        obj.size = nti(buf[124:136])
        obj.mtime = nti(buf[136:148])
        obj.chksum = chksum
        obj.type = buf[156:157]
        obj.linkname = nts(buf[157:257], encoding, errors)
        obj.uname = nts(buf[265:297], encoding, errors)
        obj.gname = nts(buf[297:329], encoding, errors)
        obj.devmajor = nti(buf[329:337])
        obj.devminor = nti(buf[337:345])
        prefix = nts(buf[345:500], encoding, errors)
        if obj.type == AREGTYPE and obj.name.endswith("/"):
            obj.type = DIRTYPE
        if obj.type == GNUTYPE_SPARSE:
            pos = 386
            structs = []
            for i in range(4):
                try:
                    offset = nti(buf[pos:pos + 12])
                    numbytes = nti(buf[pos + 12:pos + 24])
                except ValueError:
                    break
                structs.append((offset, numbytes))
                pos += 24
            isextended = bool(buf[482])
            origsize = nti(buf[483:495])
            obj._sparse_structs = (structs, isextended, origsize)
        if obj.isdir():
            obj.name = obj.name.rstrip("/")
        if prefix and obj.type not in GNU_TYPES:
            obj.name = prefix + "/" + obj.name
        return obj
    @classmethod
    def fromtarfile(cls, tarfile):
        buf = tarfile.fileobj.read(BLOCKSIZE)
        obj = cls.frombuf(buf, tarfile.encoding, tarfile.errors)
        obj.offset = tarfile.fileobj.tell() - BLOCKSIZE
        return obj._proc_member(tarfile)
    def _proc_member(self, tarfile):
        if self.type in (GNUTYPE_LONGNAME, GNUTYPE_LONGLINK):
            return self._proc_gnulong(tarfile)
        elif self.type == GNUTYPE_SPARSE:
            return self._proc_sparse(tarfile)
        elif self.type in (XHDTYPE, XGLTYPE, SOLARIS_XHDTYPE):
            return self._proc_pax(tarfile)
        else:
            return self._proc_builtin(tarfile)
    def _proc_builtin(self, tarfile):
        self.offset_data = tarfile.fileobj.tell()
        offset = self.offset_data
        if self.isreg() or self.type not in SUPPORTED_TYPES:
            offset += self._block(self.size)
        tarfile.offset = offset
        self._apply_pax_info(tarfile.pax_headers, tarfile.encoding, tarfile.errors)
        return self
    def _proc_gnulong(self, tarfile):
        buf = tarfile.fileobj.read(self._block(self.size))
        try:
            next = self.fromtarfile(tarfile)
        except HeaderError:
            raise SubsequentHeaderError("missing or bad subsequent header")
        next.offset = self.offset
        if self.type == GNUTYPE_LONGNAME:
            next.name = nts(buf, tarfile.encoding, tarfile.errors)
        elif self.type == GNUTYPE_LONGLINK:
            next.linkname = nts(buf, tarfile.encoding, tarfile.errors)
        return next
    def _proc_sparse(self, tarfile):
        structs, isextended, origsize = self._sparse_structs
        del self._sparse_structs
        while isextended:
            buf = tarfile.fileobj.read(BLOCKSIZE)
            pos = 0
            for i in range(21):
                try:
                    offset = nti(buf[pos:pos + 12])
                    numbytes = nti(buf[pos + 12:pos + 24])
                except ValueError:
                    break
                if offset and numbytes:
                    structs.append((offset, numbytes))
                pos += 24
            isextended = bool(buf[504])
        self.sparse = structs
        self.offset_data = tarfile.fileobj.tell()
        tarfile.offset = self.offset_data + self._block(self.size)
        self.size = origsize
        return self
    def _proc_pax(self, tarfile):
        buf = tarfile.fileobj.read(self._block(self.size))
        if self.type == XGLTYPE:
            pax_headers = tarfile.pax_headers
        else:
            pax_headers = tarfile.pax_headers.copy()
        match = re.search(br"\d+ hdrcharset=([^\n]+)\n", buf)
        if match is not None:
            pax_headers["hdrcharset"] = match.group(1).decode("utf8")
        hdrcharset = pax_headers.get("hdrcharset")
        if hdrcharset == "BINARY":
            encoding = tarfile.encoding
        else:
            encoding = "utf8"
        regex = re.compile(br"(\d+) ([^=]+)=")
        pos = 0
        while True:
            match = regex.match(buf, pos)
            if not match:
                break
            length, keyword = match.groups()
            length = int(length)
            value = buf[match.end(2) + 1:match.start(1) + length - 1]
            keyword = self._decode_pax_field(keyword, "utf8", "utf8",
                    tarfile.errors)
            if keyword in PAX_NAME_FIELDS:
                value = self._decode_pax_field(value, encoding, tarfile.encoding,
                        tarfile.errors)
            else:
                value = self._decode_pax_field(value, "utf8", "utf8",
                        tarfile.errors)
            pax_headers[keyword] = value
            pos += length
        try:
            next = self.fromtarfile(tarfile)
        except HeaderError:
            raise SubsequentHeaderError("missing or bad subsequent header")
        if "GNU.sparse.map" in pax_headers:
            self._proc_gnusparse_01(next, pax_headers)
        elif "GNU.sparse.size" in pax_headers:
            self._proc_gnusparse_00(next, pax_headers, buf)
        elif pax_headers.get("GNU.sparse.major") == "1" and pax_headers.get("GNU.sparse.minor") == "0":
            self._proc_gnusparse_10(next, pax_headers, tarfile)
        if self.type in (XHDTYPE, SOLARIS_XHDTYPE):
            next._apply_pax_info(pax_headers, tarfile.encoding, tarfile.errors)
            next.offset = self.offset
            if "size" in pax_headers:
                offset = next.offset_data
                if next.isreg() or next.type not in SUPPORTED_TYPES:
                    offset += next._block(next.size)
                tarfile.offset = offset
        return next
    def _proc_gnusparse_00(self, next, pax_headers, buf):
        offsets = []
        for match in re.finditer(br"\d+ GNU.sparse.offset=(\d+)\n", buf):
            offsets.append(int(match.group(1)))
        numbytes = []
        for match in re.finditer(br"\d+ GNU.sparse.numbytes=(\d+)\n", buf):
            numbytes.append(int(match.group(1)))
        next.sparse = list(zip(offsets, numbytes))
    def _proc_gnusparse_01(self, next, pax_headers):
        sparse = [int(x) for x in pax_headers["GNU.sparse.map"].split(",")]
        next.sparse = list(zip(sparse[::2], sparse[1::2]))
    def _proc_gnusparse_10(self, next, pax_headers, tarfile):
        fields = None
        sparse = []
        buf = tarfile.fileobj.read(BLOCKSIZE)
        fields, buf = buf.split(b"\n", 1)
        fields = int(fields)
        while len(sparse) < fields * 2:
            if b"\n" not in buf:
                buf += tarfile.fileobj.read(BLOCKSIZE)
            number, buf = buf.split(b"\n", 1)
            sparse.append(int(number))
        next.offset_data = tarfile.fileobj.tell()
        next.sparse = list(zip(sparse[::2], sparse[1::2]))
    def _apply_pax_info(self, pax_headers, encoding, errors):
        for keyword, value in pax_headers.items():
            if keyword == "GNU.sparse.name":
                setattr(self, "path", value)
            elif keyword == "GNU.sparse.size":
                setattr(self, "size", int(value))
            elif keyword == "GNU.sparse.realsize":
                setattr(self, "size", int(value))
            elif keyword in PAX_FIELDS:
                if keyword in PAX_NUMBER_FIELDS:
                    try:
                        value = PAX_NUMBER_FIELDS[keyword](value)
                    except ValueError:
                        value = 0
                if keyword == "path":
                    value = value.rstrip("/")
                setattr(self, keyword, value)
        self.pax_headers = pax_headers.copy()
    def _decode_pax_field(self, value, encoding, fallback_encoding, fallback_errors):
        try:
            return value.decode(encoding, "strict")
        except UnicodeDecodeError:
            return value.decode(fallback_encoding, fallback_errors)
    def _block(self, count):
        blocks, remainder = divmod(count, BLOCKSIZE)
        if remainder:
            blocks += 1
        return blocks * BLOCKSIZE
    def isreg(self):
        return self.type in REGULAR_TYPES
    def isfile(self):
        return self.isreg()
    def isdir(self):
        return self.type == DIRTYPE
    def issym(self):
        return self.type == SYMTYPE
    def islnk(self):
        return self.type == LNKTYPE
    def ischr(self):
        return self.type == CHRTYPE
    def isblk(self):
        return self.type == BLKTYPE
    def isfifo(self):
        return self.type == FIFOTYPE
    def issparse(self):
        return self.sparse is not None
    def isdev(self):
        return self.type in (CHRTYPE, BLKTYPE, FIFOTYPE)
class TarFile(object):
    debug = 0
    dereference = False
    ignore_zeros = False
    errorlevel = 1
    format = DEFAULT_FORMAT
    encoding = ENCODING
    errors = None
    tarinfo = TarInfo
    fileobject = ExFileObject
    def __init__(self, name=None, mode="r", fileobj=None, format=None,
            tarinfo=None, dereference=None, ignore_zeros=None, encoding=None,
            errors="surrogateescape", pax_headers=None, debug=None, errorlevel=None):
        if len(mode) > 1 or mode not in "raw":
            raise ValueError("mode must be 'r', 'a' or 'w'")
        self.mode = mode
        self._mode = {"r": "rb", "a": "r+b", "w": "wb"}[mode]
        if not fileobj:
            if self.mode == "a" and not os.path.exists(name):
                self.mode = "w"
                self._mode = "wb"
            fileobj = bltn_open(name, self._mode)
            self._extfileobj = False
        else:
            if name is None and hasattr(fileobj, "name"):
                name = fileobj.name
            if hasattr(fileobj, "mode"):
                self._mode = fileobj.mode
            self._extfileobj = True
        self.name = os.path.abspath(name) if name else None
        self.fileobj = fileobj
        if format is not None:
            self.format = format
        if tarinfo is not None:
            self.tarinfo = tarinfo
        if dereference is not None:
            self.dereference = dereference
        if ignore_zeros is not None:
            self.ignore_zeros = ignore_zeros
        if encoding is not None:
            self.encoding = encoding
        self.errors = errors
        if pax_headers is not None and self.format == PAX_FORMAT:
            self.pax_headers = pax_headers
        else:
            self.pax_headers = {}
        if debug is not None:
            self.debug = debug
        if errorlevel is not None:
            self.errorlevel = errorlevel
        self.closed = False
        self.members = []
        self._loaded = False
        self.offset = self.fileobj.tell()
        self.inodes = {}
        try:
            if self.mode == "r":
                self.firstmember = None
                self.firstmember = self.next()
            if self.mode == "a":
                while True:
                    self.fileobj.seek(self.offset)
                    try:
                        tarinfo = self.tarinfo.fromtarfile(self)
                        self.members.append(tarinfo)
                    except EOFHeaderError:
                        self.fileobj.seek(self.offset)
                        break
                    except HeaderError as e:
                        raise ReadError(str(e))
            if self.mode in "aw":
                self._loaded = True
                if self.pax_headers:
                    buf = self.tarinfo.create_pax_global_header(self.pax_headers.copy())
                    self.fileobj.write(buf)
                    self.offset += len(buf)
        except:
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True
            raise
    @classmethod
    def open(cls, name=None, mode="r", fileobj=None, bufsize=RECORDSIZE, **kwargs):
        if not name and not fileobj:
            raise ValueError("nothing to open")
        if mode in ("r", "r:*"):
            for comptype in cls.OPEN_METH:
                func = getattr(cls, cls.OPEN_METH[comptype])
                if fileobj is not None:
                    saved_pos = fileobj.tell()
                try:
                    return func(name, "r", fileobj, **kwargs)
                except (ReadError, CompressionError) as e:
                    if fileobj is not None:
                        fileobj.seek(saved_pos)
                    continue
            raise ReadError("file could not be opened successfully")
        elif ":" in mode:
            filemode, comptype = mode.split(":", 1)
            filemode = filemode or "r"
            comptype = comptype or "tar"
            if comptype in cls.OPEN_METH:
                func = getattr(cls, cls.OPEN_METH[comptype])
            else:
                raise CompressionError("unknown compression type %r" % comptype)
            return func(name, filemode, fileobj, **kwargs)
        elif "|" in mode:
            filemode, comptype = mode.split("|", 1)
            filemode = filemode or "r"
            comptype = comptype or "tar"
            if filemode not in "rw":
                raise ValueError("mode must be 'r' or 'w'")
            stream = _Stream(name, filemode, comptype, fileobj, bufsize)
            try:
                t = cls(name, filemode, stream, **kwargs)
            except:
                stream.close()
                raise
            t._extfileobj = False
            return t
        elif mode in "aw":
            return cls.taropen(name, mode, fileobj, **kwargs)
        raise ValueError("undiscernible mode")
    @classmethod
    def taropen(cls, name, mode="r", fileobj=None, **kwargs):
        if len(mode) > 1 or mode not in "raw":
            raise ValueError("mode must be 'r', 'a' or 'w'")
        return cls(name, mode, fileobj, **kwargs)
    @classmethod
    def gzopen(cls, name, mode="r", fileobj=None, compresslevel=9, **kwargs):
        if len(mode) > 1 or mode not in "rw":
            raise ValueError("mode must be 'r' or 'w'")
        try:
            import gzip
            gzip.GzipFile
        except (ImportError, AttributeError):
            raise CompressionError("gzip module is not available")
        extfileobj = fileobj is not None
        try:
            fileobj = gzip.GzipFile(name, mode + "b", compresslevel, fileobj)
            t = cls.taropen(name, mode, fileobj, **kwargs)
        except IOError:
            if not extfileobj and fileobj is not None:
                fileobj.close()
            if fileobj is None:
                raise
            raise ReadError("not a gzip file")
        except:
            if not extfileobj and fileobj is not None:
                fileobj.close()
            raise
        t._extfileobj = extfileobj
        return t
    @classmethod
    def bz2open(cls, name, mode="r", fileobj=None, compresslevel=9, **kwargs):
        if len(mode) > 1 or mode not in "rw":
            raise ValueError("mode must be 'r' or 'w'.")
        try:
            import bz2
        except ImportError:
            raise CompressionError("bz2 module is not available")
        if fileobj is not None:
            fileobj = _BZ2Proxy(fileobj, mode)
        else:
            fileobj = bz2.BZ2File(name, mode, compresslevel=compresslevel)
        try:
            t = cls.taropen(name, mode, fileobj, **kwargs)
        except (IOError, EOFError):
            fileobj.close()
            raise ReadError("not a bzip2 file")
        t._extfileobj = False
        return t
    OPEN_METH = {
        "tar": "taropen",
        "gz":  "gzopen",
        "bz2": "bz2open"
    }
    def close(self):
        if self.closed:
            return
        if self.mode in "aw":
            self.fileobj.write(NUL * (BLOCKSIZE * 2))
            self.offset += (BLOCKSIZE * 2)
            blocks, remainder = divmod(self.offset, RECORDSIZE)
            if remainder > 0:
                self.fileobj.write(NUL * (RECORDSIZE - remainder))
        if not self._extfileobj:
            self.fileobj.close()
        self.closed = True
    def getmember(self, name):
        tarinfo = self._getmember(name)
        if tarinfo is None:
            raise KeyError("filename %r not found" % name)
        return tarinfo
    def getmembers(self):
        self._check()
        if not self._loaded:
            self._load()
        return self.members
    def getnames(self):
        return [tarinfo.name for tarinfo in self.getmembers()]
    def gettarinfo(self, name=None, arcname=None, fileobj=None):
        self._check("aw")
        if fileobj is not None:
            name = fileobj.name
        if arcname is None:
            arcname = name
        drv, arcname = os.path.splitdrive(arcname)
        arcname = arcname.replace(os.sep, "/")
        arcname = arcname.lstrip("/")
        tarinfo = self.tarinfo()
        tarinfo.tarfile = self
        if fileobj is None:
            if hasattr(os, "lstat") and not self.dereference:
                statres = os.lstat(name)
            else:
                statres = os.stat(name)
        else:
            statres = os.fstat(fileobj.fileno())
        linkname = ""
        stmd = statres.st_mode
        if stat.S_ISREG(stmd):
            inode = (statres.st_ino, statres.st_dev)
            if not self.dereference and statres.st_nlink > 1 and                    inode in self.inodes and arcname != self.inodes[inode]:
                type = LNKTYPE
                linkname = self.inodes[inode]
            else:
                type = REGTYPE
                if inode[0]:
                    self.inodes[inode] = arcname
        elif stat.S_ISDIR(stmd):
            type = DIRTYPE
        elif stat.S_ISFIFO(stmd):
            type = FIFOTYPE
        elif stat.S_ISLNK(stmd):
            type = SYMTYPE
            linkname = os.readlink(name)
        elif stat.S_ISCHR(stmd):
            type = CHRTYPE
        elif stat.S_ISBLK(stmd):
            type = BLKTYPE
        else:
            return None
        tarinfo.name = arcname
        tarinfo.mode = stmd
        tarinfo.uid = statres.st_uid
        tarinfo.gid = statres.st_gid
        if type == REGTYPE:
            tarinfo.size = statres.st_size
        else:
            tarinfo.size = 0
        tarinfo.mtime = statres.st_mtime
        tarinfo.type = type
        tarinfo.linkname = linkname
        if pwd:
            try:
                tarinfo.uname = pwd.getpwuid(tarinfo.uid)[0]
            except KeyError:
                pass
        if grp:
            try:
                tarinfo.gname = grp.getgrgid(tarinfo.gid)[0]
            except KeyError:
                pass
        if type in (CHRTYPE, BLKTYPE):
            if hasattr(os, "major") and hasattr(os, "minor"):
                tarinfo.devmajor = os.major(statres.st_rdev)
                tarinfo.devminor = os.minor(statres.st_rdev)
        return tarinfo
    def list(self, verbose=True):
        self._check()
        for tarinfo in self:
            if verbose:
                print(filemode(tarinfo.mode), end=' ')
                print("%s/%s" % (tarinfo.uname or tarinfo.uid,
                                 tarinfo.gname or tarinfo.gid), end=' ')
                if tarinfo.ischr() or tarinfo.isblk():
                    print("%10s" % ("%d,%d"                                    % (tarinfo.devmajor, tarinfo.devminor)), end=' ')
                else:
                    print("%10d" % tarinfo.size, end=' ')
                print("%d-%02d-%02d %02d:%02d:%02d"                      % time.localtime(tarinfo.mtime)[:6], end=' ')
            print(tarinfo.name + ("/" if tarinfo.isdir() else ""), end=' ')
            if verbose:
                if tarinfo.issym():
                    print("->", tarinfo.linkname, end=' ')
                if tarinfo.islnk():
                    print("link to", tarinfo.linkname, end=' ')
            print()
    def add(self, name, arcname=None, recursive=True, exclude=None, filter=None):
        self._check("aw")
        if arcname is None:
            arcname = name
        if exclude is not None:
            import warnings
            warnings.warn("use the filter argument instead",
                    DeprecationWarning, 2)
            if exclude(name):
                self._dbg(2, "tarfile: Excluded %r" % name)
                return
        if self.name is not None and os.path.abspath(name) == self.name:
            self._dbg(2, "tarfile: Skipped %r" % name)
            return
        self._dbg(1, name)
        tarinfo = self.gettarinfo(name, arcname)
        if tarinfo is None:
            self._dbg(1, "tarfile: Unsupported type %r" % name)
            return
        if filter is not None:
            tarinfo = filter(tarinfo)
            if tarinfo is None:
                self._dbg(2, "tarfile: Excluded %r" % name)
                return
        if tarinfo.isreg():
            f = bltn_open(name, "rb")
            self.addfile(tarinfo, f)
            f.close()
        elif tarinfo.isdir():
            self.addfile(tarinfo)
            if recursive:
                for f in os.listdir(name):
                    self.add(os.path.join(name, f), os.path.join(arcname, f),
                            recursive, exclude, filter=filter)
        else:
            self.addfile(tarinfo)
    def addfile(self, tarinfo, fileobj=None):
        self._check("aw")
        tarinfo = copy.copy(tarinfo)
        buf = tarinfo.tobuf(self.format, self.encoding, self.errors)
        self.fileobj.write(buf)
        self.offset += len(buf)
        if fileobj is not None:
            copyfileobj(fileobj, self.fileobj, tarinfo.size)
            blocks, remainder = divmod(tarinfo.size, BLOCKSIZE)
            if remainder > 0:
                self.fileobj.write(NUL * (BLOCKSIZE - remainder))
                blocks += 1
            self.offset += blocks * BLOCKSIZE
        self.members.append(tarinfo)
    def extractall(self, path=".", members=None):
        directories = []
        if members is None:
            members = self
        for tarinfo in members:
            if tarinfo.isdir():
                directories.append(tarinfo)
                tarinfo = copy.copy(tarinfo)
                tarinfo.mode = 0o700
            self.extract(tarinfo, path, set_attrs=not tarinfo.isdir())
        directories.sort(key=lambda a: a.name)
        directories.reverse()
        for tarinfo in directories:
            dirpath = os.path.join(path, tarinfo.name)
            try:
                self.chown(tarinfo, dirpath)
                self.utime(tarinfo, dirpath)
                self.chmod(tarinfo, dirpath)
            except ExtractError as e:
                if self.errorlevel > 1:
                    raise
                else:
                    self._dbg(1, "tarfile: %s" % e)
    def extract(self, member, path="", set_attrs=True):
        self._check("r")
        if isinstance(member, str):
            tarinfo = self.getmember(member)
        else:
            tarinfo = member
        if tarinfo.islnk():
            tarinfo._link_target = os.path.join(path, tarinfo.linkname)
        try:
            self._extract_member(tarinfo, os.path.join(path, tarinfo.name),
                                 set_attrs=set_attrs)
        except EnvironmentError as e:
            if self.errorlevel > 0:
                raise
            else:
                if e.filename is None:
                    self._dbg(1, "tarfile: %s" % e.strerror)
                else:
                    self._dbg(1, "tarfile: %s %r" % (e.strerror, e.filename))
        except ExtractError as e:
            if self.errorlevel > 1:
                raise
            else:
                self._dbg(1, "tarfile: %s" % e)
    def extractfile(self, member):
        self._check("r")
        if isinstance(member, str):
            tarinfo = self.getmember(member)
        else:
            tarinfo = member
        if tarinfo.isreg():
            return self.fileobject(self, tarinfo)
        elif tarinfo.type not in SUPPORTED_TYPES:
            return self.fileobject(self, tarinfo)
        elif tarinfo.islnk() or tarinfo.issym():
            if isinstance(self.fileobj, _Stream):
                raise StreamError("cannot extract (sym)link as file object")
            else:
                return self.extractfile(self._find_link_target(tarinfo))
        else:
            return None
    def _extract_member(self, tarinfo, targetpath, set_attrs=True):
        targetpath = targetpath.rstrip("/")
        targetpath = targetpath.replace("/", os.sep)
        upperdirs = os.path.dirname(targetpath)
        if upperdirs and not os.path.exists(upperdirs):
            os.makedirs(upperdirs)
        if tarinfo.islnk() or tarinfo.issym():
            self._dbg(1, "%s -> %s" % (tarinfo.name, tarinfo.linkname))
        else:
            self._dbg(1, tarinfo.name)
        if tarinfo.isreg():
            self.makefile(tarinfo, targetpath)
        elif tarinfo.isdir():
            self.makedir(tarinfo, targetpath)
        elif tarinfo.isfifo():
            self.makefifo(tarinfo, targetpath)
        elif tarinfo.ischr() or tarinfo.isblk():
            self.makedev(tarinfo, targetpath)
        elif tarinfo.islnk() or tarinfo.issym():
            self.makelink(tarinfo, targetpath)
        elif tarinfo.type not in SUPPORTED_TYPES:
            self.makeunknown(tarinfo, targetpath)
        else:
            self.makefile(tarinfo, targetpath)
        if set_attrs:
            self.chown(tarinfo, targetpath)
            if not tarinfo.issym():
                self.chmod(tarinfo, targetpath)
                self.utime(tarinfo, targetpath)
    def makedir(self, tarinfo, targetpath):
        try:
            os.mkdir(targetpath, 0o700)
        except EnvironmentError as e:
            if e.errno != errno.EEXIST:
                raise
    def makefile(self, tarinfo, targetpath):
        source = self.fileobj
        source.seek(tarinfo.offset_data)
        target = bltn_open(targetpath, "wb")
        if tarinfo.sparse is not None:
            for offset, size in tarinfo.sparse:
                target.seek(offset)
                copyfileobj(source, target, size)
        else:
            copyfileobj(source, target, tarinfo.size)
        target.seek(tarinfo.size)
        target.truncate()
        target.close()
    def makeunknown(self, tarinfo, targetpath):
        self.makefile(tarinfo, targetpath)
        self._dbg(1, "tarfile: Unknown file type %r, "                     "extracted as regular file." % tarinfo.type)
    def makefifo(self, tarinfo, targetpath):
        if hasattr(os, "mkfifo"):
            os.mkfifo(targetpath)
        else:
            raise ExtractError("fifo not supported by system")
    def makedev(self, tarinfo, targetpath):
        if not hasattr(os, "mknod") or not hasattr(os, "makedev"):
            raise ExtractError("special devices not supported by system")
        mode = tarinfo.mode
        if tarinfo.isblk():
            mode |= stat.S_IFBLK
        else:
            mode |= stat.S_IFCHR
        os.mknod(targetpath, mode,
                 os.makedev(tarinfo.devmajor, tarinfo.devminor))
    def makelink(self, tarinfo, targetpath):
        try:
            if tarinfo.issym():
                os.symlink(tarinfo.linkname, targetpath)
            else:
                if os.path.exists(tarinfo._link_target):
                    os.link(tarinfo._link_target, targetpath)
                else:
                    self._extract_member(self._find_link_target(tarinfo),
                                         targetpath)
        except symlink_exception:
            if tarinfo.issym():
                linkpath = os.path.join(os.path.dirname(tarinfo.name),
                                        tarinfo.linkname)
            else:
                linkpath = tarinfo.linkname
        else:
            try:
                self._extract_member(self._find_link_target(tarinfo),
                                     targetpath)
            except KeyError:
                raise ExtractError("unable to resolve link inside archive")
    def chown(self, tarinfo, targetpath):
        if pwd and hasattr(os, "geteuid") and os.geteuid() == 0:
            try:
                g = grp.getgrnam(tarinfo.gname)[2]
            except KeyError:
                g = tarinfo.gid
            try:
                u = pwd.getpwnam(tarinfo.uname)[2]
            except KeyError:
                u = tarinfo.uid
            try:
                if tarinfo.issym() and hasattr(os, "lchown"):
                    os.lchown(targetpath, u, g)
                else:
                    if sys.platform != "os2emx":
                        os.chown(targetpath, u, g)
            except EnvironmentError as e:
                raise ExtractError("could not change owner")
    def chmod(self, tarinfo, targetpath):
        if hasattr(os, 'chmod'):
            try:
                os.chmod(targetpath, tarinfo.mode)
            except EnvironmentError as e:
                raise ExtractError("could not change mode")
    def utime(self, tarinfo, targetpath):
        if not hasattr(os, 'utime'):
            return
        try:
            os.utime(targetpath, (tarinfo.mtime, tarinfo.mtime))
        except EnvironmentError as e:
            raise ExtractError("could not change modification time")
    def next(self):
        self._check("ra")
        if self.firstmember is not None:
            m = self.firstmember
            self.firstmember = None
            return m
        self.fileobj.seek(self.offset)
        tarinfo = None
        while True:
            try:
                tarinfo = self.tarinfo.fromtarfile(self)
            except EOFHeaderError as e:
                if self.ignore_zeros:
                    self._dbg(2, "0x%X: %s" % (self.offset, e))
                    self.offset += BLOCKSIZE
                    continue
            except InvalidHeaderError as e:
                if self.ignore_zeros:
                    self._dbg(2, "0x%X: %s" % (self.offset, e))
                    self.offset += BLOCKSIZE
                    continue
                elif self.offset == 0:
                    raise ReadError(str(e))
            except EmptyHeaderError:
                if self.offset == 0:
                    raise ReadError("empty file")
            except TruncatedHeaderError as e:
                if self.offset == 0:
                    raise ReadError(str(e))
            except SubsequentHeaderError as e:
                raise ReadError(str(e))
            break
        if tarinfo is not None:
            self.members.append(tarinfo)
        else:
            self._loaded = True
        return tarinfo
    def _getmember(self, name, tarinfo=None, normalize=False):
        members = self.getmembers()
        if tarinfo is not None:
            members = members[:members.index(tarinfo)]
        if normalize:
            name = os.path.normpath(name)
        for member in reversed(members):
            if normalize:
                member_name = os.path.normpath(member.name)
            else:
                member_name = member.name
            if name == member_name:
                return member
    def _load(self):
        while True:
            tarinfo = self.next()
            if tarinfo is None:
                break
        self._loaded = True
    def _check(self, mode=None):
        if self.closed:
            raise IOError("%s is closed" % self.__class__.__name__)
        if mode is not None and self.mode not in mode:
            raise IOError("bad operation for mode %r" % self.mode)
    def _find_link_target(self, tarinfo):
        if tarinfo.issym():
            linkname = os.path.dirname(tarinfo.name) + "/" + tarinfo.linkname
            limit = None
        else:
            linkname = tarinfo.linkname
            limit = tarinfo
        member = self._getmember(linkname, tarinfo=limit, normalize=True)
        if member is None:
            raise KeyError("linkname %r not found" % linkname)
        return member
    def __iter__(self):
        if self._loaded:
            return iter(self.members)
        else:
            return TarIter(self)
    def _dbg(self, level, msg):
        if level <= self.debug:
            print(msg, file=sys.stderr)
    def __enter__(self):
        self._check()
        return self
    def __exit__(self, type, value, traceback):
        if type is None:
            self.close()
        else:
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True
class TarIter(object):
    def __init__(self, tarfile):
        self.tarfile = tarfile
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if not self.tarfile._loaded:
            tarinfo = self.tarfile.next()
            if not tarinfo:
                self.tarfile._loaded = True
                raise StopIteration
        else:
            try:
                tarinfo = self.tarfile.members[self.index]
            except IndexError:
                raise StopIteration
        self.index += 1
        return tarinfo
    next = __next__
def is_tarfile(name):
    try:
        t = open(name)
        t.close()
        return True
    except TarError:
        return False
bltn_open = open
open = TarFile.open
