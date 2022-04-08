
__docformat__ = "restructuredtext en"
import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _decode_line(line, encoding=None):
    if type(line) is bytes:
        if encoding is None:
            encoding = "latin1"
        line = line.decode(encoding)
    return line
def _is_string_like(obj):
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True
def _is_bytes_like(obj):
    try:
        obj + b''
    except (TypeError, ValueError):
        return False
    return True
def has_nested_fields(ndtype):
    for name in ndtype.names or ():
        if ndtype[name].names is not None:
            return True
    return False
def flatten_dtype(ndtype, flatten_base=False):
    names = ndtype.names
    if names is None:
        if flatten_base:
            return [ndtype.base] * int(np.prod(ndtype.shape))
        return [ndtype.base]
    else:
        types = []
        for field in names:
            info = ndtype.fields[field]
            flat_dt = flatten_dtype(info[0], flatten_base)
            types.extend(flat_dt)
        return types
class LineSplitter:
    def autostrip(self, method):
        return lambda input: [_.strip() for _ in method(input)]
                 encoding=None):
        delimiter = _decode_line(delimiter)
        comments = _decode_line(comments)
        self.comments = comments
        if (delimiter is None) or isinstance(delimiter, str):
            delimiter = delimiter or None
            _handyman = self._delimited_splitter
        elif hasattr(delimiter, '__iter__'):
            _handyman = self._variablewidth_splitter
            idx = np.cumsum([0] + list(delimiter))
            delimiter = [slice(i, j) for (i, j) in zip(idx[:-1], idx[1:])]
        elif int(delimiter):
            (_handyman, delimiter) = (
                    self._fixedwidth_splitter, int(delimiter))
        else:
            (_handyman, delimiter) = (self._delimited_splitter, None)
        self.delimiter = delimiter
        if autostrip:
            self._handyman = self.autostrip(_handyman)
        else:
            self._handyman = _handyman
        self.encoding = encoding
    def _delimited_splitter(self, line):
        if self.comments is not None:
            line = line.split(self.comments)[0]
        line = line.strip(" \r\n")
        if not line:
            return []
        return line.split(self.delimiter)
    def _fixedwidth_splitter(self, line):
        if self.comments is not None:
            line = line.split(self.comments)[0]
        line = line.strip("\r\n")
        if not line:
            return []
        fixed = self.delimiter
        slices = [slice(i, i + fixed) for i in range(0, len(line), fixed)]
        return [line[s] for s in slices]
    def _variablewidth_splitter(self, line):
        if self.comments is not None:
            line = line.split(self.comments)[0]
        if not line:
            return []
        slices = self.delimiter
        return [line[s] for s in slices]
    def __call__(self, line):
        return self._handyman(_decode_line(line, self.encoding))
class NameValidator:
    defaultexcludelist = ['return', 'file', 'print']
    def __init__(self, excludelist=None, deletechars=None,
                 case_sensitive=None, replace_space='_'):
        if excludelist is None:
            excludelist = []
        excludelist.extend(self.defaultexcludelist)
        self.excludelist = excludelist
        if deletechars is None:
            delete = self.defaultdeletechars
        else:
            delete = set(deletechars)
        delete.add('"')
        self.deletechars = delete
        if (case_sensitive is None) or (case_sensitive is True):
            self.case_converter = lambda x: x
        elif (case_sensitive is False) or case_sensitive.startswith('u'):
            self.case_converter = lambda x: x.upper()
        elif case_sensitive.startswith('l'):
            self.case_converter = lambda x: x.lower()
        else:
            msg = 'unrecognized case_sensitive value %s.' % case_sensitive
            raise ValueError(msg)
        self.replace_space = replace_space
    def validate(self, names, defaultfmt="f%i", nbfields=None):
        if (names is None):
            if (nbfields is None):
                return None
            names = []
        if isinstance(names, str):
            names = [names, ]
        if nbfields is not None:
            nbnames = len(names)
            if (nbnames < nbfields):
                names = list(names) + [''] * (nbfields - nbnames)
            elif (nbnames > nbfields):
                names = names[:nbfields]
        deletechars = self.deletechars
        excludelist = self.excludelist
        case_converter = self.case_converter
        replace_space = self.replace_space
        validatednames = []
        seen = dict()
        nbempty = 0
        for item in names:
            item = case_converter(item).strip()
            if replace_space:
                item = item.replace(' ', replace_space)
            item = ''.join([c for c in item if c not in deletechars])
            if item == '':
                item = defaultfmt % nbempty
                while item in names:
                    nbempty += 1
                    item = defaultfmt % nbempty
                nbempty += 1
            elif item in excludelist:
                item += '_'
            cnt = seen.get(item, 0)
            if cnt > 0:
                validatednames.append(item + '_%d' % cnt)
            else:
                validatednames.append(item)
            seen[item] = cnt + 1
        return tuple(validatednames)
    def __call__(self, names, defaultfmt="f%i", nbfields=None):
        return self.validate(names, defaultfmt=defaultfmt, nbfields=nbfields)
def str2bool(value):
    value = value.upper()
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    else:
        raise ValueError("Invalid boolean")
class ConverterError(Exception):
    pass
class ConverterLockError(ConverterError):
    pass
class ConversionWarning(UserWarning):
    pass
class StringConverter:
    _mapper = [(nx.bool_, str2bool, False),
               (nx.int_, int, -1),]
    if nx.dtype(nx.int_).itemsize < nx.dtype(nx.int64).itemsize:
        _mapper.append((nx.int64, int, -1))
    _mapper.extend([(nx.float64, float, nx.nan),
                    (nx.complex128, complex, nx.nan + 0j),
                    (nx.longdouble, nx.longdouble, nx.nan),
                    (nx.integer, int, -1),
                    (nx.floating, float, nx.nan),
                    (nx.complexfloating, complex, nx.nan + 0j),
                    (nx.unicode_, asunicode, '???'),
                    (nx.string_, asbytes, '???'),
                    ])
    @classmethod
    def _getdtype(cls, val):
        return np.array(val).dtype
    @classmethod
    def _getsubdtype(cls, val):
        return np.array(val).dtype.type
    @classmethod
    def _dtypeortype(cls, dtype):
        if dtype.type == np.datetime64:
            return dtype
        return dtype.type
    @classmethod
    def upgrade_mapper(cls, func, default=None):
        if hasattr(func, '__call__'):
            cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
            return
        elif hasattr(func, '__iter__'):
            if isinstance(func[0], (tuple, list)):
                for _ in func:
                    cls._mapper.insert(-1, _)
                return
            if default is None:
                default = [None] * len(func)
            else:
                default = list(default)
                default.append([None] * (len(func) - len(default)))
            for fct, dft in zip(func, default):
                cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))
    @classmethod
    def _find_map_entry(cls, dtype):
        for i, (deftype, func, default_def) in enumerate(cls._mapper):
            if dtype.type == deftype:
                return i, (deftype, func, default_def)
        for i, (deftype, func, default_def) in enumerate(cls._mapper):
            if np.issubdtype(dtype.type, deftype):
                return i, (deftype, func, default_def)
        raise LookupError
    def __init__(self, dtype_or_func=None, default=None, missing_values=None,
                 locked=False):
        self._locked = bool(locked)
        if dtype_or_func is None:
            self.func = str2bool
            self._status = 0
            self.default = default or False
            dtype = np.dtype('bool')
        else:
            try:
                self.func = None
                dtype = np.dtype(dtype_or_func)
            except TypeError:
                if not hasattr(dtype_or_func, '__call__'):
                    errmsg = ("The input argument `dtype` is neither a"
                              " function nor a dtype (got '%s' instead)")
                    raise TypeError(errmsg % type(dtype_or_func))
                self.func = dtype_or_func
                if default is None:
                    try:
                        default = self.func('0')
                    except ValueError:
                        default = None
                dtype = self._getdtype(default)
            try:
                self._status, (_, func, default_def) = self._find_map_entry(dtype)
            except LookupError:
                self.default = default
                _, func, _ = self._mapper[-1]
                self._status = 0
            else:
                if default is None:
                    self.default = default_def
                else:
                    self.default = default
            if self.func is None:
                self.func = func
            if self.func == self._mapper[1][1]:
                if issubclass(dtype.type, np.uint64):
                    self.func = np.uint64
                elif issubclass(dtype.type, np.int64):
                    self.func = np.int64
                else:
                    self.func = lambda x: int(float(x))
        if missing_values is None:
            self.missing_values = {''}
        else:
            if isinstance(missing_values, str):
                missing_values = missing_values.split(",")
            self.missing_values = set(list(missing_values) + [''])
        self._callingfunction = self._strict_call
        self.type = self._dtypeortype(dtype)
        self._checked = False
        self._initial_default = default
    def _loose_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            return self.default
    def _strict_call(self, value):
        try:
            new_value = self.func(value)
            if self.func is int:
                try:
                    np.array(value, dtype=self.type)
                except OverflowError:
                    raise ValueError
            return new_value
        except ValueError:
            if value.strip() in self.missing_values:
                if not self._status:
                    self._checked = False
                return self.default
            raise ValueError("Cannot convert string '%s'" % value)
    def __call__(self, value):
        return self._callingfunction(value)
    def _do_upgrade(self):
        if self._locked:
            errmsg = "Converter is locked and cannot be upgraded"
            raise ConverterLockError(errmsg)
        _statusmax = len(self._mapper)
        _status = self._status
        if _status == _statusmax:
            errmsg = "Could not find a valid conversion function"
            raise ConverterError(errmsg)
        elif _status < _statusmax - 1:
            _status += 1
        self.type, self.func, default = self._mapper[_status]
        self._status = _status
        if self._initial_default is not None:
            self.default = self._initial_default
        else:
            self.default = default
    def upgrade(self, value):
        self._checked = True
        try:
            return self._strict_call(value)
        except ValueError:
            self._do_upgrade()
            return self.upgrade(value)
    def iterupgrade(self, value):
        self._checked = True
        if not hasattr(value, '__iter__'):
            value = (value,)
        _strict_call = self._strict_call
        try:
            for _m in value:
                _strict_call(_m)
        except ValueError:
            self._do_upgrade()
            self.iterupgrade(value)
    def update(self, func, default=None, testing_value=None,
               missing_values='', locked=False):
        self.func = func
        self._locked = locked
        if default is not None:
            self.default = default
            self.type = self._dtypeortype(self._getdtype(default))
        else:
            try:
                tester = func(testing_value or '1')
            except (TypeError, ValueError):
                tester = None
            self.type = self._dtypeortype(self._getdtype(tester))
        if missing_values is None:
            self.missing_values = set()
        else:
            if not np.iterable(missing_values):
                missing_values = [missing_values]
            if not all(isinstance(v, str) for v in missing_values):
                raise TypeError("missing_values must be strings or unicode")
            self.missing_values.update(missing_values)
def easy_dtype(ndtype, names=None, defaultfmt="f%i", **validationargs):
    try:
        ndtype = np.dtype(ndtype)
    except TypeError:
        validate = NameValidator(**validationargs)
        nbfields = len(ndtype)
        if names is None:
            names = [''] * len(ndtype)
        elif isinstance(names, str):
            names = names.split(",")
        names = validate(names, nbfields=nbfields, defaultfmt=defaultfmt)
        ndtype = np.dtype(dict(formats=ndtype, names=names))
    else:
        if names is not None:
            validate = NameValidator(**validationargs)
            if isinstance(names, str):
                names = names.split(",")
            if ndtype.names is None:
                formats = tuple([ndtype.type] * len(names))
                names = validate(names, defaultfmt=defaultfmt)
                ndtype = np.dtype(list(zip(names, formats)))
            else:
                ndtype.names = validate(names, nbfields=len(ndtype.names),
                                        defaultfmt=defaultfmt)
        elif ndtype.names is not None:
            validate = NameValidator(**validationargs)
            numbered_names = tuple("f%i" % i for i in range(len(ndtype.names)))
            if ((ndtype.names == numbered_names) and (defaultfmt != "f%i")):
                ndtype.names = validate([''] * len(ndtype.names),
                                        defaultfmt=defaultfmt)
            else:
                ndtype.names = validate(ndtype.names, defaultfmt=defaultfmt)
    return ndtype
