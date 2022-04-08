
import ast
import re
import sys
import platform
from .multiarray import dtype, array, ndarray
try:
    import ctypes
except ImportError:
    ctypes = None
IS_PYPY = platform.python_implementation() == 'PyPy'
if sys.byteorder == 'little':
    _nbo = '<'
else:
    _nbo = '>'
def _makenames_list(adict, align):
    allfields = []
    for fname, obj in adict.items():
        n = len(obj)
        if not isinstance(obj, tuple) or n not in (2, 3):
            raise ValueError("entry not a 2- or 3- tuple")
        if n > 2 and obj[2] == fname:
            continue
        num = int(obj[1])
        if num < 0:
            raise ValueError("invalid offset.")
        format = dtype(obj[0], align=align)
        if n > 2:
            title = obj[2]
        else:
            title = None
        allfields.append((fname, format, num, title))
    allfields.sort(key=lambda x: x[2])
    names = [x[0] for x in allfields]
    formats = [x[1] for x in allfields]
    offsets = [x[2] for x in allfields]
    titles = [x[3] for x in allfields]
    return names, formats, offsets, titles
def _usefields(adict, align):
    try:
        names = adict[-1]
    except KeyError:
        names = None
    if names is None:
        names, formats, offsets, titles = _makenames_list(adict, align)
    else:
        formats = []
        offsets = []
        titles = []
        for name in names:
            res = adict[name]
            formats.append(res[0])
            offsets.append(res[1])
            if len(res) > 2:
                titles.append(res[2])
            else:
                titles.append(None)
    return dtype({"names": names,
                  "formats": formats,
                  "offsets": offsets,
                  "titles": titles}, align)
def _array_descr(descriptor):
    fields = descriptor.fields
    if fields is None:
        subdtype = descriptor.subdtype
        if subdtype is None:
            if descriptor.metadata is None:
                return descriptor.str
            else:
                new = descriptor.metadata.copy()
                if new:
                    return (descriptor.str, new)
                else:
                    return descriptor.str
        else:
            return (_array_descr(subdtype[0]), subdtype[1])
    names = descriptor.names
    ordered_fields = [fields[x] + (x,) for x in names]
    result = []
    offset = 0
    for field in ordered_fields:
        if field[1] > offset:
            num = field[1] - offset
            result.append(('', f'|V{num}'))
            offset += num
        elif field[1] < offset:
            raise ValueError(
                "dtype.descr is not defined for types with overlapping or "
                "out-of-order fields")
        if len(field) > 3:
            name = (field[2], field[3])
        else:
            name = field[2]
        if field[0].subdtype:
            tup = (name, _array_descr(field[0].subdtype[0]),
                   field[0].subdtype[1])
        else:
            tup = (name, _array_descr(field[0]))
        offset += field[0].itemsize
        result.append(tup)
    if descriptor.itemsize > offset:
        num = descriptor.itemsize - offset
        result.append(('', f'|V{num}'))
    return result
def _reconstruct(subtype, shape, dtype):
    return ndarray.__new__(subtype, shape, dtype)
format_re = re.compile(r'(?P<order1>[<>|=]?)'
                       r'(?P<repeats> *[(]?[ ,0-9]*[)]? *)'
                       r'(?P<order2>[<>|=]?)'
                       r'(?P<dtype>[A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)')
sep_re = re.compile(r'\s*,\s*')
space_re = re.compile(r'\s+$')
_convorder = {'=': _nbo}
def _commastring(astr):
    startindex = 0
    result = []
    while startindex < len(astr):
        mo = format_re.match(astr, pos=startindex)
        try:
            (order1, repeats, order2, dtype) = mo.groups()
        except (TypeError, AttributeError):
            raise ValueError(
                f'format number {len(result)+1} of "{astr}" is not recognized'
                ) from None
        startindex = mo.end()
        if startindex < len(astr):
            if space_re.match(astr, pos=startindex):
                startindex = len(astr)
            else:
                mo = sep_re.match(astr, pos=startindex)
                if not mo:
                    raise ValueError(
                        'format number %d of "%s" is not recognized' %
                        (len(result)+1, astr))
                startindex = mo.end()
        if order2 == '':
            order = order1
        elif order1 == '':
            order = order2
        else:
            order1 = _convorder.get(order1, order1)
            order2 = _convorder.get(order2, order2)
            if (order1 != order2):
                raise ValueError(
                    'inconsistent byte-order specification %s and %s' %
                    (order1, order2))
            order = order1
        if order in ('|', '=', _nbo):
            order = ''
        dtype = order + dtype
        if (repeats == ''):
            newitem = dtype
        else:
            newitem = (dtype, ast.literal_eval(repeats))
        result.append(newitem)
    return result
class dummy_ctype:
    def __init__(self, cls):
        self._cls = cls
    def __mul__(self, other):
        return self
    def __call__(self, *other):
        return self._cls(other)
    def __eq__(self, other):
        return self._cls == other._cls
    def __ne__(self, other):
        return self._cls != other._cls
def _getintp_ctype():
    val = _getintp_ctype.cache
    if val is not None:
        return val
    if ctypes is None:
        import numpy as np
        val = dummy_ctype(np.intp)
    else:
        char = dtype('p').char
        if char == 'i':
            val = ctypes.c_int
        elif char == 'l':
            val = ctypes.c_long
        elif char == 'q':
            val = ctypes.c_longlong
        else:
            val = ctypes.c_long
    _getintp_ctype.cache = val
    return val
_getintp_ctype.cache = None
class _missing_ctypes:
    def cast(self, num, obj):
        return num.value
    class c_void_p:
        def __init__(self, ptr):
            self.value = ptr
class _ctypes:
    def __init__(self, array, ptr=None):
        self._arr = array
        if ctypes:
            self._ctypes = ctypes
            self._data = self._ctypes.c_void_p(ptr)
        else:
            self._ctypes = _missing_ctypes()
            self._data = self._ctypes.c_void_p(ptr)
            self._data._objects = array
        if self._arr.ndim == 0:
            self._zerod = True
        else:
            self._zerod = False
    def data_as(self, obj):
        ptr = self._ctypes.cast(self._data, obj)
        ptr._arr = self._arr
        return ptr
    def shape_as(self, obj):
        if self._zerod:
            return None
        return (obj*self._arr.ndim)(*self._arr.shape)
    def strides_as(self, obj):
        if self._zerod:
            return None
        return (obj*self._arr.ndim)(*self._arr.strides)
    @property
    def data(self):
        return self._data.value
    @property
    def shape(self):
        return self.shape_as(_getintp_ctype())
    @property
    def strides(self):
        return self.strides_as(_getintp_ctype())
    @property
    def _as_parameter_(self):
        return self.data_as(ctypes.c_void_p)
    get_data = data.fget
    get_shape = shape.fget
    get_strides = strides.fget
    get_as_parameter = _as_parameter_.fget
def _newnames(datatype, order):
    oldnames = datatype.names
    nameslist = list(oldnames)
    if isinstance(order, str):
        order = [order]
    seen = set()
    if isinstance(order, (list, tuple)):
        for name in order:
            try:
                nameslist.remove(name)
            except ValueError:
                if name in seen:
                    raise ValueError(f"duplicate field name: {name}") from None
                else:
                    raise ValueError(f"unknown field name: {name}") from None
            seen.add(name)
        return tuple(list(order) + nameslist)
    raise ValueError(f"unsupported order value: {order}")
def _copy_fields(ary):
    dt = ary.dtype
    copy_dtype = {'names': dt.names,
                  'formats': [dt.fields[name][0] for name in dt.names]}
    return array(ary, dtype=copy_dtype, copy=True)
def _getfield_is_safe(oldtype, newtype, offset):
    if newtype.hasobject or oldtype.hasobject:
        if offset == 0 and newtype == oldtype:
            return
        if oldtype.names is not None:
            for name in oldtype.names:
                if (oldtype.fields[name][1] == offset and
                        oldtype.fields[name][0] == newtype):
                    return
        raise TypeError("Cannot get/set field of an object array")
    return
def _view_is_safe(oldtype, newtype):
    if oldtype == newtype:
        return
    if newtype.hasobject or oldtype.hasobject:
        raise TypeError("Cannot change data-type for object array.")
    return
_pep3118_native_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'h',
    'H': 'H',
    'i': 'i',
    'I': 'I',
    'l': 'l',
    'L': 'L',
    'q': 'q',
    'Q': 'Q',
    'e': 'e',
    'f': 'f',
    'd': 'd',
    'g': 'g',
    'Zf': 'F',
    'Zd': 'D',
    'Zg': 'G',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',
}
_pep3118_native_typechars = ''.join(_pep3118_native_map.keys())
_pep3118_standard_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'i2',
    'H': 'u2',
    'i': 'i4',
    'I': 'u4',
    'l': 'i4',
    'L': 'u4',
    'q': 'i8',
    'Q': 'u8',
    'e': 'f2',
    'f': 'f',
    'd': 'd',
    'Zf': 'F',
    'Zd': 'D',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',
}
_pep3118_standard_typechars = ''.join(_pep3118_standard_map.keys())
_pep3118_unsupported_map = {
    'u': 'UCS-2 strings',
    '&': 'pointers',
    't': 'bitfields',
    'X': 'function pointers',
}
class _Stream:
    def __init__(self, s):
        self.s = s
        self.byteorder = '@'
    def advance(self, n):
        res = self.s[:n]
        self.s = self.s[n:]
        return res
    def consume(self, c):
        if self.s[:len(c)] == c:
            self.advance(len(c))
            return True
        return False
    def consume_until(self, c):
        if callable(c):
            i = 0
            while i < len(self.s) and not c(self.s[i]):
                i = i + 1
            return self.advance(i)
        else:
            i = self.s.index(c)
            res = self.advance(i)
            self.advance(len(c))
            return res
    @property
    def next(self):
        return self.s[0]
    def __bool__(self):
        return bool(self.s)
def _dtype_from_pep3118(spec):
    stream = _Stream(spec)
    dtype, align = __dtype_from_pep3118(stream, is_subdtype=False)
    return dtype
def __dtype_from_pep3118(stream, is_subdtype):
    field_spec = dict(
        names=[],
        formats=[],
        offsets=[],
        itemsize=0
    )
    offset = 0
    common_alignment = 1
    is_padding = False
    while stream:
        value = None
        if stream.consume('}'):
            break
        shape = None
        if stream.consume('('):
            shape = stream.consume_until(')')
            shape = tuple(map(int, shape.split(',')))
        if stream.next in ('@', '=', '<', '>', '^', '!'):
            byteorder = stream.advance(1)
            if byteorder == '!':
                byteorder = '>'
            stream.byteorder = byteorder
        if stream.byteorder in ('@', '^'):
            type_map = _pep3118_native_map
            type_map_chars = _pep3118_native_typechars
        else:
            type_map = _pep3118_standard_map
            type_map_chars = _pep3118_standard_typechars
        itemsize_str = stream.consume_until(lambda c: not c.isdigit())
        if itemsize_str:
            itemsize = int(itemsize_str)
        else:
            itemsize = 1
        is_padding = False
        if stream.consume('T{'):
            value, align = __dtype_from_pep3118(
                stream, is_subdtype=True)
        elif stream.next in type_map_chars:
            if stream.next == 'Z':
                typechar = stream.advance(2)
            else:
                typechar = stream.advance(1)
            is_padding = (typechar == 'x')
            dtypechar = type_map[typechar]
            if dtypechar in 'USV':
                dtypechar += '%d' % itemsize
                itemsize = 1
            numpy_byteorder = {'@': '=', '^': '='}.get(
                stream.byteorder, stream.byteorder)
            value = dtype(numpy_byteorder + dtypechar)
            align = value.alignment
        elif stream.next in _pep3118_unsupported_map:
            desc = _pep3118_unsupported_map[stream.next]
            raise NotImplementedError(
                "Unrepresentable PEP 3118 data type {!r} ({})"
                .format(stream.next, desc))
        else:
            raise ValueError("Unknown PEP 3118 data type specifier %r" % stream.s)
        extra_offset = 0
        if stream.byteorder == '@':
            start_padding = (-offset) % align
            intra_padding = (-value.itemsize) % align
            offset += start_padding
            if intra_padding != 0:
                if itemsize > 1 or (shape is not None and _prod(shape) > 1):
                    value = _add_trailing_padding(value, intra_padding)
                else:
                    extra_offset += intra_padding
            common_alignment = _lcm(align, common_alignment)
        if itemsize != 1:
            value = dtype((value, (itemsize,)))
        if shape is not None:
            value = dtype((value, shape))
        if stream.consume(':'):
            name = stream.consume_until(':')
        else:
            name = None
        if not (is_padding and name is None):
            if name is not None and name in field_spec['names']:
                raise RuntimeError(f"Duplicate field name '{name}' in PEP3118 format")
            field_spec['names'].append(name)
            field_spec['formats'].append(value)
            field_spec['offsets'].append(offset)
        offset += value.itemsize
        offset += extra_offset
        field_spec['itemsize'] = offset
    if stream.byteorder == '@':
        field_spec['itemsize'] += (-offset) % common_alignment
    if (field_spec['names'] == [None]
            and field_spec['offsets'][0] == 0
            and field_spec['itemsize'] == field_spec['formats'][0].itemsize
            and not is_subdtype):
        ret = field_spec['formats'][0]
    else:
        _fix_names(field_spec)
        ret = dtype(field_spec)
    return ret, common_alignment
def _fix_names(field_spec):
    names = field_spec['names']
    for i, name in enumerate(names):
        if name is not None:
            continue
        j = 0
        while True:
            name = f'f{j}'
            if name not in names:
                break
            j = j + 1
        names[i] = name
def _add_trailing_padding(value, padding):
    if value.fields is None:
        field_spec = dict(
            names=['f0'],
            formats=[value],
            offsets=[0],
            itemsize=value.itemsize
        )
    else:
        fields = value.fields
        names = value.names
        field_spec = dict(
            names=names,
            formats=[fields[name][0] for name in names],
            offsets=[fields[name][1] for name in names],
            itemsize=value.itemsize
        )
    field_spec['itemsize'] += padding
    return dtype(field_spec)
def _prod(a):
    p = 1
    for x in a:
        p *= x
    return p
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a
def _lcm(a, b):
    return a // _gcd(a, b) * b
def array_ufunc_errmsg_formatter(dummy, ufunc, method, *inputs, **kwargs):
    args_string = ', '.join(['{!r}'.format(arg) for arg in inputs] +
                            ['{}={!r}'.format(k, v)
                             for k, v in kwargs.items()])
    args = inputs + kwargs.get('out', ())
    types_string = ', '.join(repr(type(arg).__name__) for arg in args)
    return ('operand type(s) all returned NotImplemented from '
            '__array_ufunc__({!r}, {!r}, {}): {}'
            .format(ufunc, method, args_string, types_string))
def array_function_errmsg_formatter(public_api, types):
    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    return ("no implementation found for '{}' on types that implement "
            '__array_function__: {}'.format(func_name, list(types)))
def _ufunc_doc_signature_formatter(ufunc):
    if ufunc.nin == 1:
        in_args = 'x'
    else:
        in_args = ', '.join(f'x{i+1}' for i in range(ufunc.nin))
    if ufunc.nout == 0:
        out_args = ', /, out=()'
    elif ufunc.nout == 1:
        out_args = ', /, out=None'
    else:
        out_args = '[, {positional}], / [, out={default}]'.format(
            positional=', '.join(
                'out{}'.format(i+1) for i in range(ufunc.nout)),
            default=repr((None,)*ufunc.nout)
        )
    kwargs = (
        ", casting='same_kind'"
        ", order='K'"
        ", dtype=None"
        ", subok=True"
        "[, signature"
        ", extobj]"
    )
    if ufunc.signature is None:
        kwargs = ", where=True" + kwargs
    return '{name}({in_args}{out_args}, *{kwargs})'.format(
        name=ufunc.__name__,
        in_args=in_args,
        out_args=out_args,
        kwargs=kwargs
    )
def npy_ctypes_check(cls):
    try:
        if IS_PYPY:
            ctype_base = cls.__mro__[-3]
        else:
            ctype_base = cls.__mro__[-2]
        return '_ctypes' in ctype_base.__module__
    except Exception:
        return False
class recursive:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.func(self, *args, **kwargs)
