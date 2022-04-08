
import sys
try:
    import numpy
except ImportError:
    numpy = None
HAVE_NUMPY = numpy is not None
is_64bits = sys.maxsize > 2**32
def _make_variant_dtype():
    ptr_typecode = '<u8' if is_64bits else '<u4'
    _tagBRECORD_format = [
        ('pvRecord', ptr_typecode),
        ('pRecInfo', ptr_typecode),
    ]
    U_VARIANT_format = dict(
        names=[
            'VT_BOOL', 'VT_I1', 'VT_I2', 'VT_I4', 'VT_I8', 'VT_INT', 'VT_UI1',
            'VT_UI2', 'VT_UI4', 'VT_UI8', 'VT_UINT', 'VT_R4', 'VT_R8', 'VT_CY',
            'c_wchar_p', 'c_void_p', 'pparray', 'bstrVal', '_tagBRECORD',
        ],
        formats=[
            '<i2', '<i1', '<i2', '<i4', '<i8', '<i4', '<u1', '<u2', '<u4',
            '<u8', '<u4', '<f4', '<f8', '<i8', ptr_typecode, ptr_typecode,
            ptr_typecode, ptr_typecode, _tagBRECORD_format,
        ],
        offsets=[0] * 19
    )
    tagVARIANT_format = [
        ("vt", '<u2'),
        ("wReserved1", '<u2'),
        ("wReserved2", '<u2'),
        ("wReserved3", '<u2'),
        ("_", U_VARIANT_format),
    ]
    return numpy.dtype(tagVARIANT_format)
def isndarray(value):
    if not HAVE_NUMPY:
        return False
    return isinstance(value, numpy.ndarray)
def isdatetime64(value):
    if not HAVE_NUMPY:
        return False
    return isinstance(value, datetime64)
def _check_ctypeslib_typecodes():
    import numpy as np
    from numpy import ctypeslib
    try:
        from numpy.ctypeslib import _typecodes
    except ImportError:
        from numpy.ctypeslib import as_ctypes_type
        ctypes_to_dtypes = {}
        for tp in set(np.sctypeDict.values()):
            try:
                ctype_for = as_ctypes_type(tp)
                ctypes_to_dtypes[ctype_for] = tp
            except NotImplementedError:
                continue
        ctypeslib._typecodes = ctypes_to_dtypes
    return ctypeslib._typecodes
com_null_date64 = None
datetime64 = None
VARIANT_dtype = None
typecodes = {}
if HAVE_NUMPY:
    typecodes = _check_ctypeslib_typecodes()
    try:
        VARIANT_dtype = _make_variant_dtype()
    except ValueError:
        pass
    try:
        from numpy import datetime64
    except ImportError:
        pass
    else:
        try:
            com_null_date64 = datetime64("1899-12-30T00:00:00", "ns")
        except TypeError:
            pass
