
import os
import warnings
from collections import Counter, OrderedDict
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import (
    os_fspath, contextlib_nullcontext
)
from numpy.core.overrides import set_module
from .arrayprint import get_printoptions
__all__ = ['record', 'recarray', 'format_parser']
ndarray = sb.ndarray
_byteorderconv = {'b':'>',
                  'l':'<',
                  'n':'=',
                  'B':'>',
                  'L':'<',
                  'N':'=',
                  'S':'s',
                  's':'s',
                  '>':'>',
                  '<':'<',
                  '=':'=',
                  '|':'|',
                  'I':'|',
                  'i':'|'}
numfmt = nt.typeDict
class _OrderedCounter(Counter, OrderedDict):
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))
    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
def find_duplicate(list):
    return [
        item
        for item, counts in _OrderedCounter(list).items()
        if counts > 1
    ]
@set_module('numpy')
class format_parser:
    def __init__(self, formats, names, titles, aligned=False, byteorder=None):
        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdtype(byteorder)
    def _parseFormats(self, formats, aligned=False):
        if formats is None:
            raise ValueError("Need formats argument")
        if isinstance(formats, list):
            dtype = sb.dtype(
                [('f{}'.format(i), format_) for i, format_ in enumerate(formats)],
                aligned,
            )
        else:
            dtype = sb.dtype(formats, aligned)
        fields = dtype.fields
        if fields is None:
            dtype = sb.dtype([('f1', dtype)], aligned)
            fields = dtype.fields
        keys = dtype.names
        self._f_formats = [fields[key][0] for key in keys]
        self._offsets = [fields[key][1] for key in keys]
        self._nfields = len(keys)
    def _setfieldnames(self, names, titles):
        if names:
            if type(names) in [list, tuple]:
                pass
            elif isinstance(names, str):
                names = names.split(',')
            else:
                raise NameError("illegal input names %s" % repr(names))
            self._names = [n.strip() for n in names[:self._nfields]]
        else:
            self._names = []
        self._names += ['f%d' % i for i in range(len(self._names),
                                                 self._nfields)]
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError("Duplicate field names: %s" % _dup)
        if titles:
            self._titles = [n.strip() for n in titles[:self._nfields]]
        else:
            self._titles = []
            titles = []
        if self._nfields > len(titles):
            self._titles += [None] * (self._nfields - len(titles))
    def _createdtype(self, byteorder):
        dtype = sb.dtype({
            'names': self._names,
            'formats': self._f_formats,
            'offsets': self._offsets,
            'titles': self._titles,
        })
        if byteorder is not None:
            byteorder = _byteorderconv[byteorder[0]]
            dtype = dtype.newbyteorder(byteorder)
        self.dtype = dtype
class record(nt.void):
    __name__ = 'record'
    __module__ = 'numpy'
    def __repr__(self):
        if get_printoptions()['legacy'] == '1.13':
            return self.__str__()
        return super(record, self).__repr__()
    def __str__(self):
        if get_printoptions()['legacy'] == '1.13':
            return str(self.item())
        return super(record, self).__str__()
    def __getattribute__(self, attr):
        if attr in ('setfield', 'getfield', 'dtype'):
            return nt.void.__getattribute__(self, attr)
        try:
            return nt.void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            obj = self.getfield(*res[:2])
            try:
                dt = obj.dtype
            except AttributeError:
                return obj
            if dt.names is not None:
                return obj.view((self.__class__, obj.dtype))
            return obj
        else:
            raise AttributeError("'record' object has no "
                    "attribute '%s'" % attr)
    def __setattr__(self, attr, val):
        if attr in ('setfield', 'getfield', 'dtype'):
            raise AttributeError("Cannot set '%s' attribute" % attr)
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            return self.setfield(val, *res[:2])
        else:
            if getattr(self, attr, None):
                return nt.void.__setattr__(self, attr, val)
            else:
                raise AttributeError("'record' object has no "
                        "attribute '%s'" % attr)
    def __getitem__(self, indx):
        obj = nt.void.__getitem__(self, indx)
        if isinstance(obj, nt.void) and obj.dtype.names is not None:
            return obj.view((self.__class__, obj.dtype))
        else:
            return obj
    def pprint(self):
        names = self.dtype.names
        maxlen = max(len(name) for name in names)
        fmt = '%% %ds: %%s' % maxlen
        rows = [fmt % (name, getattr(self, name)) for name in names]
        return "\n".join(rows)
class recarray(ndarray):
    __name__ = 'recarray'
    __module__ = 'numpy'
    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False, order='C'):
        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(formats, names, titles, aligned, byteorder).dtype
        if buf is None:
            self = ndarray.__new__(subtype, shape, (record, descr), order=order)
        else:
            self = ndarray.__new__(subtype, shape, (record, descr),
                                      buffer=buf, offset=offset,
                                      strides=strides, order=order)
        return self
    def __array_finalize__(self, obj):
        if self.dtype.type is not record and self.dtype.names is not None:
            self.dtype = self.dtype
    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError("recarray has no attribute %s" % attr) from e
        obj = self.getfield(*res)
        if obj.dtype.names is not None:
            if issubclass(obj.dtype.type, nt.void):
                return obj.view(dtype=(self.dtype.type, obj.dtype))
            return obj
        else:
            return obj.view(ndarray)
    def __setattr__(self, attr, val):
        if attr == 'dtype' and issubclass(val.type, nt.void) and val.names is not None:
            val = sb.dtype((record, val))
        newattr = attr not in self.__dict__
        try:
            ret = object.__setattr__(self, attr, val)
        except Exception:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                raise
        else:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                return ret
            if newattr:
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(
                "record array has no attribute %s" % attr
            ) from e
        return self.setfield(val, *res)
    def __getitem__(self, indx):
        obj = super(recarray, self).__getitem__(indx)
        if isinstance(obj, ndarray):
            if obj.dtype.names is not None:
                obj = obj.view(type(self))
                if issubclass(obj.dtype.type, nt.void):
                    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj
            else:
                return obj.view(type=ndarray)
        else:
            return obj
    def __repr__(self):
        repr_dtype = self.dtype
        if self.dtype.type is record or not issubclass(self.dtype.type, nt.void):
            if repr_dtype.type is record:
                repr_dtype = sb.dtype((nt.void, repr_dtype))
            prefix = "rec.array("
            fmt = 'rec.array(%s,%sdtype=%s)'
        else:
            prefix = "array("
            fmt = 'array(%s,%sdtype=%s).view(numpy.recarray)'
        if self.size > 0 or self.shape == (0,):
            lst = sb.array2string(
                self, separator=', ', prefix=prefix, suffix=',')
        else:
            lst = "[], shape=%s" % (repr(self.shape),)
        lf = '\n'+' '*len(prefix)
        if get_printoptions()['legacy'] == '1.13':
            lf = ' ' + lf
        return fmt % (lst, lf, repr_dtype)
    def field(self, attr, val=None):
        if isinstance(attr, int):
            names = ndarray.__getattribute__(self, 'dtype').names
            attr = names[attr]
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        res = fielddict[attr][:2]
        if val is None:
            obj = self.getfield(*res)
            if obj.dtype.names is not None:
                return obj
            return obj.view(ndarray)
        else:
            return self.setfield(val, *res)
def _deprecate_shape_0_as_None(shape):
    if shape == 0:
        warnings.warn(
            "Passing `shape=0` to have the shape be inferred is deprecated, "
            "and in future will be equivalent to `shape=(0,)`. To infer "
            "the shape and suppress this warning, pass `shape=None` instead.",
            FutureWarning, stacklevel=3)
        return None
    else:
        return shape
def fromarrays(arrayList, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    arrayList = [sb.asarray(x) for x in arrayList]
    shape = _deprecate_shape_0_as_None(shape)
    if shape is None:
        shape = arrayList[0].shape
    elif isinstance(shape, int):
        shape = (shape,)
    if formats is None and dtype is None:
        formats = [obj.dtype for obj in arrayList]
    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    _names = descr.names
    if len(descr) != len(arrayList):
        raise ValueError("mismatch between the number of fields "
                "and the number of arrays")
    d0 = descr[0].shape
    nn = len(d0)
    if nn > 0:
        shape = shape[:-nn]
    for k, obj in enumerate(arrayList):
        nn = descr[k].ndim
        testshape = obj.shape[:obj.ndim - nn]
        if testshape != shape:
            raise ValueError("array-shape mismatch in array %d" % k)
    _array = recarray(shape, descr)
    for i in range(len(arrayList)):
        _array[_names[i]] = arrayList[i]
    return _array
def fromrecords(recList, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None):
    if formats is None and dtype is None:
        obj = sb.array(recList, dtype=object)
        arrlist = [sb.array(obj[..., i].tolist()) for i in range(obj.shape[-1])]
        return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                          titles=titles, aligned=aligned, byteorder=byteorder)
    if dtype is not None:
        descr = sb.dtype((record, dtype))
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    try:
        retval = sb.array(recList, dtype=descr)
    except (TypeError, ValueError):
        shape = _deprecate_shape_0_as_None(shape)
        if shape is None:
            shape = len(recList)
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) > 1:
            raise ValueError("Can only deal with 1-d array.")
        _array = recarray(shape, descr)
        for k in range(_array.size):
            _array[k] = tuple(recList[k])
        warnings.warn(
            "fromrecords expected a list of tuples, may have received a list "
            "of lists instead. In the future that will raise an error",
            FutureWarning, stacklevel=2)
        return _array
    else:
        if shape is not None and retval.shape != shape:
            retval.shape = shape
    res = retval.view(recarray)
    return res
def fromstring(datastring, dtype=None, shape=None, offset=0, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    if dtype is None and formats is None:
        raise TypeError("fromstring() needs a 'dtype' or 'formats' argument")
    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    itemsize = descr.itemsize
    shape = _deprecate_shape_0_as_None(shape)
    if shape in (None, -1):
        shape = (len(datastring) - offset) // itemsize
    _array = recarray(shape, descr, buf=datastring, offset=offset)
    return _array
def get_remaining_size(fd):
    pos = fd.tell()
    try:
        fd.seek(0, 2)
        return fd.tell() - pos
    finally:
        fd.seek(pos, 0)
def fromfile(fd, dtype=None, shape=None, offset=0, formats=None,
             names=None, titles=None, aligned=False, byteorder=None):
    if dtype is None and formats is None:
        raise TypeError("fromfile() needs a 'dtype' or 'formats' argument")
    shape = _deprecate_shape_0_as_None(shape)
    if shape is None:
        shape = (-1,)
    elif isinstance(shape, int):
        shape = (shape,)
    if hasattr(fd, 'readinto'):
        ctx = contextlib_nullcontext(fd)
    else:
        ctx = open(os_fspath(fd), 'rb')
    with ctx as fd:
        if offset > 0:
            fd.seek(offset, 1)
        size = get_remaining_size(fd)
        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(formats, names, titles, aligned, byteorder).dtype
        itemsize = descr.itemsize
        shapeprod = sb.array(shape).prod(dtype=nt.intp)
        shapesize = shapeprod * itemsize
        if shapesize < 0:
            shape = list(shape)
            shape[shape.index(-1)] = size // -shapesize
            shape = tuple(shape)
            shapeprod = sb.array(shape).prod(dtype=nt.intp)
        nbytes = shapeprod * itemsize
        if nbytes > size:
            raise ValueError(
                    "Not enough bytes left in file for specified shape and type")
        _array = recarray(shape, descr)
        nbytesread = fd.readinto(_array.data)
        if nbytesread != nbytes:
            raise IOError("Didn't read as many bytes as expected")
    return _array
def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
          names=None, titles=None, aligned=False, byteorder=None, copy=True):
    if ((isinstance(obj, (type(None), str)) or hasattr(obj, 'readinto')) and
           formats is None and dtype is None):
        raise ValueError("Must define formats (or dtype) if object is "
                         "None, string, or an open file")
    kwds = {}
    if dtype is not None:
        dtype = sb.dtype(dtype)
    elif formats is not None:
        dtype = format_parser(formats, names, titles,
                              aligned, byteorder).dtype
    else:
        kwds = {'formats': formats,
                'names': names,
                'titles': titles,
                'aligned': aligned,
                'byteorder': byteorder
                }
    if obj is None:
        if shape is None:
            raise ValueError("Must define a shape if obj is None")
        return recarray(shape, dtype, buf=obj, offset=offset, strides=strides)
    elif isinstance(obj, bytes):
        return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)
    elif isinstance(obj, (list, tuple)):
        if isinstance(obj[0], (tuple, list)):
            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
        else:
            return fromarrays(obj, dtype=dtype, shape=shape, **kwds)
    elif isinstance(obj, recarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new
    elif hasattr(obj, 'readinto'):
        return fromfile(obj, dtype=dtype, shape=shape, offset=offset)
    elif isinstance(obj, ndarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new.view(recarray)
    else:
        interface = getattr(obj, "__array_interface__", None)
        if interface is None or not isinstance(interface, dict):
            raise ValueError("Unknown input type")
        obj = sb.array(obj)
        if dtype is not None and (obj.dtype != dtype):
            obj = obj.view(dtype)
        return obj.view(recarray)
