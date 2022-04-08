
import numpy as np
_kind_to_stem = {
    'u': 'uint',
    'i': 'int',
    'c': 'complex',
    'f': 'float',
    'b': 'bool',
    'V': 'void',
    'O': 'object',
    'M': 'datetime',
    'm': 'timedelta',
    'S': 'bytes',
    'U': 'str',
}
def _kind_name(dtype):
    try:
        return _kind_to_stem[dtype.kind]
    except KeyError as e:
        raise RuntimeError(
            "internal dtype error, unknown kind {!r}"
            .format(dtype.kind)
        ) from None
def __str__(dtype):
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=True)
    elif dtype.subdtype:
        return _subarray_str(dtype)
    elif issubclass(dtype.type, np.flexible) or not dtype.isnative:
        return dtype.str
    else:
        return dtype.name
def __repr__(dtype):
    arg_str = _construction_repr(dtype, include_align=False)
    if dtype.isalignedstruct:
        arg_str = arg_str + ", align=True"
    return "dtype({})".format(arg_str)
def _unpack_field(dtype, offset, title=None):
    return dtype, offset, title
def _isunsized(dtype):
    return dtype.itemsize == 0
def _construction_repr(dtype, include_align=False, short=False):
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=include_align)
    elif dtype.subdtype:
        return _subarray_str(dtype)
    else:
        return _scalar_str(dtype, short=short)
def _scalar_str(dtype, short):
    byteorder = _byte_order_str(dtype)
    if dtype.type == np.bool_:
        if short:
            return "'?'"
        else:
            return "'bool'"
    elif dtype.type == np.object_:
        return "'O'"
    elif dtype.type == np.string_:
        if _isunsized(dtype):
            return "'S'"
        else:
            return "'S%d'" % dtype.itemsize
    elif dtype.type == np.unicode_:
        if _isunsized(dtype):
            return "'%sU'" % byteorder
        else:
            return "'%sU%d'" % (byteorder, dtype.itemsize / 4)
    elif issubclass(dtype.type, np.void):
        if _isunsized(dtype):
            return "'V'"
        else:
            return "'V%d'" % dtype.itemsize
    elif dtype.type == np.datetime64:
        return "'%sM8%s'" % (byteorder, _datetime_metadata_str(dtype))
    elif dtype.type == np.timedelta64:
        return "'%sm8%s'" % (byteorder, _datetime_metadata_str(dtype))
    elif np.issubdtype(dtype, np.number):
        if short or dtype.byteorder not in ('=', '|'):
            return "'%s%c%d'" % (byteorder, dtype.kind, dtype.itemsize)
        else:
            return "'%s%d'" % (_kind_name(dtype), 8*dtype.itemsize)
    elif dtype.isbuiltin == 2:
        return dtype.type.__name__
    else:
        raise RuntimeError(
            "Internal error: NumPy dtype unrecognized type number")
def _byte_order_str(dtype):
    swapped = np.dtype(int).newbyteorder('S')
    native = swapped.newbyteorder('S')
    byteorder = dtype.byteorder
    if byteorder == '=':
        return native.byteorder
    if byteorder == 'S':
        return swapped.byteorder
    elif byteorder == '|':
        return ''
    else:
        return byteorder
def _datetime_metadata_str(dtype):
    unit, count = np.datetime_data(dtype)
    if unit == 'generic':
        return ''
    elif count == 1:
        return '[{}]'.format(unit)
    else:
        return '[{}{}]'.format(count, unit)
def _struct_dict_str(dtype, includealignedflag):
    names = dtype.names
    fld_dtypes = []
    offsets = []
    titles = []
    for name in names:
        fld_dtype, offset, title = _unpack_field(*dtype.fields[name])
        fld_dtypes.append(fld_dtype)
        offsets.append(offset)
        titles.append(title)
    ret = "{'names':["
    ret += ",".join(repr(name) for name in names)
    ret += "], 'formats':["
    ret += ",".join(
        _construction_repr(fld_dtype, short=True) for fld_dtype in fld_dtypes)
    ret += "], 'offsets':["
    ret += ",".join("%d" % offset for offset in offsets)
    if any(title is not None for title in titles):
        ret += "], 'titles':["
        ret += ",".join(repr(title) for title in titles)
    ret += "], 'itemsize':%d" % dtype.itemsize
    if (includealignedflag and dtype.isalignedstruct):
        ret += ", 'aligned':True}"
    else:
        ret += "}"
    return ret
def _is_packed(dtype):
    total_offset = 0
    for name in dtype.names:
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])
        if fld_offset != total_offset:
            return False
        total_offset += fld_dtype.itemsize
    if total_offset != dtype.itemsize:
        return False
    return True
def _struct_list_str(dtype):
    items = []
    for name in dtype.names:
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])
        item = "("
        if title is not None:
            item += "({!r}, {!r}), ".format(title, name)
        else:
            item += "{!r}, ".format(name)
        if fld_dtype.subdtype is not None:
            base, shape = fld_dtype.subdtype
            item += "{}, {}".format(
                _construction_repr(base, short=True),
                shape
            )
        else:
            item += _construction_repr(fld_dtype, short=True)
        item += ")"
        items.append(item)
    return "[" + ", ".join(items) + "]"
def _struct_str(dtype, include_align):
    if not (include_align and dtype.isalignedstruct) and _is_packed(dtype):
        sub = _struct_list_str(dtype)
    else:
        sub = _struct_dict_str(dtype, include_align)
    if dtype.type != np.void:
        return "({t.__module__}.{t.__name__}, {f})".format(t=dtype.type, f=sub)
    else:
        return sub
def _subarray_str(dtype):
    base, shape = dtype.subdtype
    return "({}, {})".format(
        _construction_repr(base, short=True),
        shape
    )
def _name_includes_bit_suffix(dtype):
    if dtype.type == np.object_:
        return False
    elif dtype.type == np.bool_:
        return False
    elif np.issubdtype(dtype, np.flexible) and _isunsized(dtype):
        return False
    else:
        return True
def _name_get(dtype):
    if dtype.isbuiltin == 2:
        return dtype.type.__name__
    if issubclass(dtype.type, np.void):
        name = dtype.type.__name__
    else:
        name = _kind_name(dtype)
    if _name_includes_bit_suffix(dtype):
        name += "{}".format(dtype.itemsize * 8)
    if dtype.type in (np.datetime64, np.timedelta64):
        name += _datetime_metadata_str(dtype)
    return name
