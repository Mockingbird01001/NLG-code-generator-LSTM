
__all__ = ['load_library', 'ndpointer', 'c_intp', 'as_ctypes', 'as_array']
import os
from numpy import (
    integer, ndarray, dtype as _dtype, array, frombuffer
)
from numpy.core.multiarray import _flagdict, flagsobj
try:
    import ctypes
except ImportError:
    ctypes = None
if ctypes is None:
    def _dummy(*args, **kwds):
        raise ImportError("ctypes is not available.")
    load_library = _dummy
    as_ctypes = _dummy
    as_array = _dummy
    from numpy import intp as c_intp
    _ndptr_base = object
else:
    import numpy.core._internal as nic
    c_intp = nic._getintp_ctype()
    del nic
    _ndptr_base = ctypes.c_void_p
    def load_library(libname, loader_path):
        if ctypes.__version__ < '1.0.1':
            import warnings
            warnings.warn("All features of ctypes interface may not work "
                          "with ctypes < 1.0.1", stacklevel=2)
        ext = os.path.splitext(libname)[1]
        if not ext:
            from numpy.distutils.misc_util import get_shared_lib_extension
            so_ext = get_shared_lib_extension()
            libname_ext = [libname + so_ext]
            so_ext2 = get_shared_lib_extension(is_python_ext=True)
            if not so_ext2 == so_ext:
                libname_ext.insert(0, libname + so_ext2)
        else:
            libname_ext = [libname]
        loader_path = os.path.abspath(loader_path)
        if not os.path.isdir(loader_path):
            libdir = os.path.dirname(loader_path)
        else:
            libdir = loader_path
        for ln in libname_ext:
            libpath = os.path.join(libdir, ln)
            if os.path.exists(libpath):
                try:
                    return ctypes.cdll[libpath]
                except OSError:
                    raise
        raise OSError("no file with expected extension")
def _num_fromflags(flaglist):
    num = 0
    for val in flaglist:
        num += _flagdict[val]
    return num
_flagnames = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE',
              'OWNDATA', 'UPDATEIFCOPY', 'WRITEBACKIFCOPY']
def _flags_fromnum(num):
    res = []
    for key in _flagnames:
        value = _flagdict[key]
        if (num & value):
            res.append(key)
    return res
class _ndptr(_ndptr_base):
    @classmethod
    def from_param(cls, obj):
        if not isinstance(obj, ndarray):
            raise TypeError("argument must be an ndarray")
        if cls._dtype_ is not None               and obj.dtype != cls._dtype_:
            raise TypeError("array must have data type %s" % cls._dtype_)
        if cls._ndim_ is not None               and obj.ndim != cls._ndim_:
            raise TypeError("array must have %d dimension(s)" % cls._ndim_)
        if cls._shape_ is not None               and obj.shape != cls._shape_:
            raise TypeError("array must have shape %s" % str(cls._shape_))
        if cls._flags_ is not None               and ((obj.flags.num & cls._flags_) != cls._flags_):
            raise TypeError("array must have flags %s" %
                    _flags_fromnum(cls._flags_))
        return obj.ctypes
class _concrete_ndptr(_ndptr):
    def _check_retval_(self):
        return self.contents
    @property
    def contents(self):
        full_dtype = _dtype((self._dtype_, self._shape_))
        full_ctype = ctypes.c_char * full_dtype.itemsize
        buffer = ctypes.cast(self, ctypes.POINTER(full_ctype)).contents
        return frombuffer(buffer, dtype=full_dtype).squeeze(axis=0)
_pointer_type_cache = {}
def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    if dtype is not None:
        dtype = _dtype(dtype)
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, integer)):
            num = flags
            flags = _flags_fromnum(num)
        elif isinstance(flags, flagsobj):
            num = flags.num
            flags = _flags_fromnum(num)
        if num is None:
            try:
                flags = [x.strip().upper() for x in flags]
            except Exception as e:
                raise TypeError("invalid flags specification") from e
            num = _num_fromflags(flags)
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape,)
    cache_key = (dtype, ndim, shape, num)
    try:
        return _pointer_type_cache[cache_key]
    except KeyError:
        pass
    if dtype is None:
        name = 'any'
    elif dtype.names is not None:
        name = str(id(dtype))
    else:
        name = dtype.str
    if ndim is not None:
        name += "_%dd" % ndim
    if shape is not None:
        name += "_"+"x".join(str(x) for x in shape)
    if flags is not None:
        name += "_"+"_".join(flags)
    if dtype is not None and shape is not None:
        base = _concrete_ndptr
    else:
        base = _ndptr
    klass = type("ndpointer_%s"%name, (base,),
                 {"_dtype_": dtype,
                  "_shape_" : shape,
                  "_ndim_" : ndim,
                  "_flags_" : num})
    _pointer_type_cache[cache_key] = klass
    return klass
if ctypes is not None:
    def _ctype_ndarray(element_type, shape):
        for dim in shape[::-1]:
            element_type = dim * element_type
            element_type.__module__ = None
        return element_type
    def _get_scalar_type_map():
        ct = ctypes
        simple_types = [
            ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
            ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
            ct.c_float, ct.c_double,
            ct.c_bool,
        ]
        return {_dtype(ctype): ctype for ctype in simple_types}
    _scalar_type_map = _get_scalar_type_map()
    def _ctype_from_dtype_scalar(dtype):
        dtype_with_endian = dtype.newbyteorder('S').newbyteorder('S')
        dtype_native = dtype.newbyteorder('=')
        try:
            ctype = _scalar_type_map[dtype_native]
        except KeyError as e:
            raise NotImplementedError(
                "Converting {!r} to a ctypes type".format(dtype)
            ) from None
        if dtype_with_endian.byteorder == '>':
            ctype = ctype.__ctype_be__
        elif dtype_with_endian.byteorder == '<':
            ctype = ctype.__ctype_le__
        return ctype
    def _ctype_from_dtype_subarray(dtype):
        element_dtype, shape = dtype.subdtype
        ctype = _ctype_from_dtype(element_dtype)
        return _ctype_ndarray(ctype, shape)
    def _ctype_from_dtype_structured(dtype):
        field_data = []
        for name in dtype.names:
            field_dtype, offset = dtype.fields[name][:2]
            field_data.append((offset, name, _ctype_from_dtype(field_dtype)))
        field_data = sorted(field_data, key=lambda f: f[0])
        if len(field_data) > 1 and all(offset == 0 for offset, name, ctype in field_data):
            size = 0
            _fields_ = []
            for offset, name, ctype in field_data:
                _fields_.append((name, ctype))
                size = max(size, ctypes.sizeof(ctype))
            if dtype.itemsize != size:
                _fields_.append(('', ctypes.c_char * dtype.itemsize))
            return type('union', (ctypes.Union,), dict(
                _fields_=_fields_,
                _pack_=1,
                __module__=None,
            ))
        else:
            last_offset = 0
            _fields_ = []
            for offset, name, ctype in field_data:
                padding = offset - last_offset
                if padding < 0:
                    raise NotImplementedError("Overlapping fields")
                if padding > 0:
                    _fields_.append(('', ctypes.c_char * padding))
                _fields_.append((name, ctype))
                last_offset = offset + ctypes.sizeof(ctype)
            padding = dtype.itemsize - last_offset
            if padding > 0:
                _fields_.append(('', ctypes.c_char * padding))
            return type('struct', (ctypes.Structure,), dict(
                _fields_=_fields_,
                _pack_=1,
                __module__=None,
            ))
    def _ctype_from_dtype(dtype):
        if dtype.fields is not None:
            return _ctype_from_dtype_structured(dtype)
        elif dtype.subdtype is not None:
            return _ctype_from_dtype_subarray(dtype)
        else:
            return _ctype_from_dtype_scalar(dtype)
    def as_ctypes_type(dtype):
        return _ctype_from_dtype(_dtype(dtype))
    def as_array(obj, shape=None):
        if isinstance(obj, ctypes._Pointer):
            if shape is None:
                raise TypeError(
                    'as_array() requires a shape argument when called on a '
                    'pointer')
            p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
            obj = ctypes.cast(obj, p_arr_type).contents
        return array(obj, copy=False)
    def as_ctypes(obj):
        ai = obj.__array_interface__
        if ai["strides"]:
            raise TypeError("strided arrays not supported")
        if ai["version"] != 3:
            raise TypeError("only __array_interface__ version 3 supported")
        addr, readonly = ai["data"]
        if readonly:
            raise TypeError("readonly arrays unsupported")
        ctype_scalar = as_ctypes_type(ai["typestr"])
        result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
        result = result_type.from_address(addr)
        result.__keep = obj
        return result
