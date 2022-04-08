
import numpy as np
def _from_ctypes_array(t):
    return np.dtype((dtype_from_ctypes_type(t._type_), (t._length_,)))
def _from_ctypes_structure(t):
    for item in t._fields_:
        if len(item) > 2:
            raise TypeError(
                "ctypes bitfields have no dtype equivalent")
    if hasattr(t, "_pack_"):
        import ctypes
        formats = []
        offsets = []
        names = []
        current_offset = 0
        for fname, ftyp in t._fields_:
            names.append(fname)
            formats.append(dtype_from_ctypes_type(ftyp))
            effective_pack = min(t._pack_, ctypes.alignment(ftyp))
            current_offset = ((current_offset + effective_pack - 1) // effective_pack) * effective_pack
            offsets.append(current_offset)
            current_offset += ctypes.sizeof(ftyp)
        return np.dtype(dict(
            formats=formats,
            offsets=offsets,
            names=names,
            itemsize=ctypes.sizeof(t)))
    else:
        fields = []
        for fname, ftyp in t._fields_:
            fields.append((fname, dtype_from_ctypes_type(ftyp)))
        return np.dtype(fields, align=True)
def _from_ctypes_scalar(t):
    if getattr(t, '__ctype_be__', None) is t:
        return np.dtype('>' + t._type_)
    elif getattr(t, '__ctype_le__', None) is t:
        return np.dtype('<' + t._type_)
    else:
        return np.dtype(t._type_)
def _from_ctypes_union(t):
    import ctypes
    formats = []
    offsets = []
    names = []
    for fname, ftyp in t._fields_:
        names.append(fname)
        formats.append(dtype_from_ctypes_type(ftyp))
        offsets.append(0)
    return np.dtype(dict(
        formats=formats,
        offsets=offsets,
        names=names,
        itemsize=ctypes.sizeof(t)))
def dtype_from_ctypes_type(t):
    import _ctypes
    if issubclass(t, _ctypes.Array):
        return _from_ctypes_array(t)
    elif issubclass(t, _ctypes._Pointer):
        raise TypeError("ctypes pointers have no dtype equivalent")
    elif issubclass(t, _ctypes.Structure):
        return _from_ctypes_structure(t)
    elif issubclass(t, _ctypes.Union):
        return _from_ctypes_union(t)
    elif isinstance(getattr(t, '_type_', None), str):
        return _from_ctypes_scalar(t)
    else:
        raise NotImplementedError(
            "Unknown ctypes type {}".format(t.__name__))
