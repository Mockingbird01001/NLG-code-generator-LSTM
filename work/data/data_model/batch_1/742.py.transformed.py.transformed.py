
from numbers import Number
from future.utils import PY3, istext, with_metaclass, isnewbytes
from future.types import no, issubset
from future.types.newobject import newobject
if PY3:
    unicode = str
    from collections.abc import Iterable
else:
    from collections import Iterable
class BaseNewStr(type):
    def __instancecheck__(cls, instance):
        if cls == newstr:
            return isinstance(instance, unicode)
        else:
            return issubclass(instance.__class__, cls)
class newstr(with_metaclass(BaseNewStr, unicode)):
    no_convert_msg = "Can't convert '{0}' object to str implicitly"
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            return super(newstr, cls).__new__(cls)
        elif type(args[0]) == newstr and cls == newstr:
            return args[0]
        elif isinstance(args[0], unicode):
            value = args[0]
        elif isinstance(args[0], bytes):
            if 'encoding' in kwargs or len(args) > 1:
                value = args[0].decode(*args[1:], **kwargs)
            else:
                value = args[0].__str__()
        else:
            value = args[0]
        return super(newstr, cls).__new__(cls, value)
    def __repr__(self):
        value = super(newstr, self).__repr__()
        return value[1:]
    def __getitem__(self, y):
        return newstr(super(newstr, self).__getitem__(y))
    def __contains__(self, key):
        errmsg = "'in <string>' requires string as left operand, not {0}"
        if type(key) == newstr:
            newkey = key
        elif isinstance(key, unicode) or isinstance(key, bytes) and not isnewbytes(key):
            newkey = newstr(key)
        else:
            raise TypeError(errmsg.format(type(key)))
        return issubset(list(newkey), list(self))
    @no('newbytes')
    def __add__(self, other):
        return newstr(super(newstr, self).__add__(other))
    @no('newbytes')
    def __radd__(self, left):
        try:
            return newstr(left) + self
        except:
            return NotImplemented
    def __mul__(self, other):
        return newstr(super(newstr, self).__mul__(other))
    def __rmul__(self, other):
        return newstr(super(newstr, self).__rmul__(other))
    def join(self, iterable):
        errmsg = 'sequence item {0}: expected unicode string, found bytes'
        for i, item in enumerate(iterable):
            if isnewbytes(item):
                raise TypeError(errmsg.format(i))
        if type(self) == newstr:
            return newstr(super(newstr, self).join(iterable))
        else:
            return newstr(super(newstr, newstr(self)).join(iterable))
    @no('newbytes')
    def find(self, sub, *args):
        return super(newstr, self).find(sub, *args)
    @no('newbytes')
    def rfind(self, sub, *args):
        return super(newstr, self).rfind(sub, *args)
    @no('newbytes', (1, 2))
    def replace(self, old, new, *args):
        return newstr(super(newstr, self).replace(old, new, *args))
    def decode(self, *args):
        raise AttributeError("decode method has been disabled in newstr")
    def encode(self, encoding='utf-8', errors='strict'):
        from future.types.newbytes import newbytes
        if errors == 'surrogateescape':
            if encoding == 'utf-16':
                raise NotImplementedError('FIXME: surrogateescape handling is '
                                          'not yet implemented properly')
            mybytes = []
            for c in self:
                code = ord(c)
                if 0xD800 <= code <= 0xDCFF:
                    mybytes.append(newbytes([code - 0xDC00]))
                else:
                    mybytes.append(c.encode(encoding=encoding))
            return newbytes(b'').join(mybytes)
        return newbytes(super(newstr, self).encode(encoding, errors))
    @no('newbytes', 1)
    def startswith(self, prefix, *args):
        if isinstance(prefix, Iterable):
            for thing in prefix:
                if isnewbytes(thing):
                    raise TypeError(self.no_convert_msg.format(type(thing)))
        return super(newstr, self).startswith(prefix, *args)
    @no('newbytes', 1)
    def endswith(self, prefix, *args):
        if isinstance(prefix, Iterable):
            for thing in prefix:
                if isnewbytes(thing):
                    raise TypeError(self.no_convert_msg.format(type(thing)))
        return super(newstr, self).endswith(prefix, *args)
    @no('newbytes', 1)
    def split(self, sep=None, maxsplit=-1):
        parts = super(newstr, self).split(sep, maxsplit)
        return [newstr(part) for part in parts]
    @no('newbytes', 1)
    def rsplit(self, sep=None, maxsplit=-1):
        parts = super(newstr, self).rsplit(sep, maxsplit)
        return [newstr(part) for part in parts]
    @no('newbytes', 1)
    def partition(self, sep):
        parts = super(newstr, self).partition(sep)
        return tuple(newstr(part) for part in parts)
    @no('newbytes', 1)
    def rpartition(self, sep):
        parts = super(newstr, self).rpartition(sep)
        return tuple(newstr(part) for part in parts)
    @no('newbytes', 1)
    def index(self, sub, *args):
        pos = self.find(sub, *args)
        if pos == -1:
            raise ValueError('substring not found')
        return pos
    def splitlines(self, keepends=False):
        parts = super(newstr, self).splitlines(keepends)
        return [newstr(part) for part in parts]
    def __eq__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__eq__(other)
        else:
            return NotImplemented
    def __hash__(self):
        if (isinstance(self, unicode) or
            isinstance(self, bytes) and not isnewbytes(self)):
            return super(newstr, self).__hash__()
        else:
            raise NotImplementedError()
    def __ne__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__ne__(other)
        else:
            return True
    unorderable_err = 'unorderable types: str() and {0}'
    def __lt__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__lt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))
    def __le__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__le__(other)
        raise TypeError(self.unorderable_err.format(type(other)))
    def __gt__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__gt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))
    def __ge__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__ge__(other)
        raise TypeError(self.unorderable_err.format(type(other)))
    def __getattribute__(self, name):
        if name in ['decode', u'decode']:
            raise AttributeError("decode method has been disabled in newstr")
        return super(newstr, self).__getattribute__(name)
    def __native__(self):
        return unicode(self)
    @staticmethod
    def maketrans(x, y=None, z=None):
        if y is None:
            assert z is None
            if not isinstance(x, dict):
                raise TypeError('if you give only one argument to maketrans it must be a dict')
            result = {}
            for (key, value) in x.items():
                if len(key) > 1:
                    raise ValueError('keys in translate table must be strings or integers')
                result[ord(key)] = value
        else:
            if not isinstance(x, unicode) and isinstance(y, unicode):
                raise TypeError('x and y must be unicode strings')
            if not len(x) == len(y):
                raise ValueError('the first two maketrans arguments must have equal length')
            result = {}
            for (xi, yi) in zip(x, y):
                if len(xi) > 1:
                    raise ValueError('keys in translate table must be strings or integers')
                result[ord(xi)] = ord(yi)
        if z is not None:
            for char in z:
                result[ord(char)] = None
        return result
    def translate(self, table):
        l = []
        for c in self:
            if ord(c) in table:
                val = table[ord(c)]
                if val is None:
                    continue
                elif isinstance(val, unicode):
                    l.append(val)
                else:
                    l.append(chr(val))
            else:
                l.append(c)
        return ''.join(l)
    def isprintable(self):
        raise NotImplementedError('fixme')
    def isidentifier(self):
        raise NotImplementedError('fixme')
    def format_map(self):
        raise NotImplementedError('fixme')
__all__ = ['newstr']
