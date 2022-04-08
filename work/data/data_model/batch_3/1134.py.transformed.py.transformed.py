
from numpy.compat import unicode
from numpy.core._string_helpers import english_lower
from numpy.core.multiarray import typeinfo, dtype
from numpy.core._dtype import _kind_name
sctypeDict = {}
allTypes = {}
_abstract_types = {}
_concrete_typeinfo = {}
for k, v in typeinfo.items():
    k = english_lower(k)
    if isinstance(v, type):
        _abstract_types[k] = v
    else:
        _concrete_typeinfo[k] = v
_concrete_types = {v.type for k, v in _concrete_typeinfo.items()}
def _bits_of(obj):
    try:
        info = next(v for v in _concrete_typeinfo.values() if v.type is obj)
    except StopIteration:
        if obj in _abstract_types.values():
            raise ValueError("Cannot count the bits of an abstract type")
        return dtype(obj).itemsize * 8
    else:
        return info.bits
def bitname(obj):
    bits = _bits_of(obj)
    dt = dtype(obj)
    char = dt.kind
    base = _kind_name(dt)
    if base == 'object':
        bits = 0
    if bits != 0:
        char = "%s%d" % (char, bits // 8)
    return base, bits, char
def _add_types():
    for name, info in _concrete_typeinfo.items():
        allTypes[name] = info.type
        sctypeDict[name] = info.type
        sctypeDict[info.char] = info.type
        sctypeDict[info.num] = info.type
    for name, cls in _abstract_types.items():
        allTypes[name] = cls
_add_types()
_int_ctypes = ['long', 'longlong', 'int', 'short', 'byte']
_uint_ctypes = list('u' + t for t in _int_ctypes)
def _add_aliases():
    for name, info in _concrete_typeinfo.items():
        if name in _int_ctypes or name in _uint_ctypes:
            continue
        base, bit, char = bitname(info.type)
        myname = "%s%d" % (base, bit)
        if name in ('longdouble', 'clongdouble') and myname in allTypes:
            continue
        allTypes[myname] = info.type
        sctypeDict[myname] = info.type
        sctypeDict[char] = info.type
    for name in ["Bytes0", "Datetime64", "Str0", "Uint32", "Uint64"]:
        if english_lower(name) not in allTypes:
            continue
        allTypes[name] = allTypes[english_lower(name)]
        sctypeDict[name] = sctypeDict[english_lower(name)]
_add_aliases()
def _add_integer_aliases():
    seen_bits = set()
    for i_ctype, u_ctype in zip(_int_ctypes, _uint_ctypes):
        i_info = _concrete_typeinfo[i_ctype]
        u_info = _concrete_typeinfo[u_ctype]
        bits = i_info.bits
        for info, charname, intname in [
                (i_info,'i%d' % (bits//8,), 'int%d' % bits),
                (u_info,'u%d' % (bits//8,), 'uint%d' % bits)]:
            if bits not in seen_bits:
                allTypes[intname] = info.type
                sctypeDict[intname] = info.type
                sctypeDict[charname] = info.type
        seen_bits.add(bits)
_add_integer_aliases()
void = allTypes['void']
def _set_up_aliases():
    type_pairs = [('complex_', 'cdouble'),
                  ('int0', 'intp'),
                  ('uint0', 'uintp'),
                  ('single', 'float'),
                  ('csingle', 'cfloat'),
                  ('singlecomplex', 'cfloat'),
                  ('float_', 'double'),
                  ('intc', 'int'),
                  ('uintc', 'uint'),
                  ('int_', 'long'),
                  ('uint', 'ulong'),
                  ('cfloat', 'cdouble'),
                  ('longfloat', 'longdouble'),
                  ('clongfloat', 'clongdouble'),
                  ('longcomplex', 'clongdouble'),
                  ('bool_', 'bool'),
                  ('bytes_', 'string'),
                  ('string_', 'string'),
                  ('str_', 'unicode'),
                  ('unicode_', 'unicode'),
                  ('object_', 'object')]
    for alias, t in type_pairs:
        allTypes[alias] = allTypes[t]
        sctypeDict[alias] = sctypeDict[t]
    to_remove = ['ulong', 'object', 'int', 'float',
                 'complex', 'bool', 'string', 'datetime', 'timedelta',
                 'bytes', 'str']
    for t in to_remove:
        try:
            del allTypes[t]
            del sctypeDict[t]
        except KeyError:
            pass
_set_up_aliases()
sctypes = {'int': [],
           'uint':[],
           'float':[],
           'complex':[],
           'others':[bool, object, bytes, unicode, void]}
def _add_array_type(typename, bits):
    try:
        t = allTypes['%s%d' % (typename, bits)]
    except KeyError:
        pass
    else:
        sctypes[typename].append(t)
def _set_array_types():
    ibytes = [1, 2, 4, 8, 16, 32, 64]
    fbytes = [2, 4, 8, 10, 12, 16, 32, 64]
    for bytes in ibytes:
        bits = 8*bytes
        _add_array_type('int', bits)
        _add_array_type('uint', bits)
    for bytes in fbytes:
        bits = 8*bytes
        _add_array_type('float', bits)
        _add_array_type('complex', 2*bits)
    _gi = dtype('p')
    if _gi.type not in sctypes['int']:
        indx = 0
        sz = _gi.itemsize
        _lst = sctypes['int']
        while (indx < len(_lst) and sz >= _lst[indx](0).itemsize):
            indx += 1
        sctypes['int'].insert(indx, _gi.type)
        sctypes['uint'].insert(indx, dtype('P').type)
_set_array_types()
_toadd = ['int', 'float', 'complex', 'bool', 'object',
          'str', 'bytes', ('a', 'bytes_')]
for name in _toadd:
    if isinstance(name, tuple):
        sctypeDict[name[0]] = allTypes[name[1]]
    else:
        sctypeDict[name] = allTypes['%s_' % name]
del _toadd, name
