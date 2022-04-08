import threading
import array
from ctypes import (POINTER, Structure, byref, cast, c_long, memmove, pointer,
                    sizeof)
from comtypes import _safearray, IUnknown, com_interface_registry, npsupport
from comtypes.patcher import Patch
numpy = npsupport.numpy
_safearray_type_cache = {}
class _SafeArrayAsNdArrayContextManager(object):
    thread_local = threading.local()
    def __enter__(self):
        try:
            self.thread_local.count += 1
        except AttributeError:
            self.thread_local.count = 1
    def __exit__(self, exc_type, exc_value, traceback):
        self.thread_local.count -= 1
    def __bool__(self):
        return bool(getattr(self.thread_local, 'count', 0))
safearray_as_ndarray = _SafeArrayAsNdArrayContextManager()
def _midlSAFEARRAY(itemtype):
    try:
        return POINTER(_safearray_type_cache[itemtype])
    except KeyError:
        sa_type = _make_safearray_type(itemtype)
        _safearray_type_cache[itemtype] = sa_type
        return POINTER(sa_type)
def _make_safearray_type(itemtype):
    from comtypes.automation import _ctype_to_vartype, VT_RECORD,         VT_UNKNOWN, IDispatch, VT_DISPATCH
    meta = type(_safearray.tagSAFEARRAY)
    sa_type = meta.__new__(meta,
                           "SAFEARRAY_%s" % itemtype.__name__,
                           (_safearray.tagSAFEARRAY,), {})
    try:
        vartype = _ctype_to_vartype[itemtype]
        extra = None
    except KeyError:
        if issubclass(itemtype, Structure):
            try:
                guids = itemtype._recordinfo_
            except AttributeError:
                extra = None
            else:
                from comtypes.typeinfo import GetRecordInfoFromGuids
                extra = GetRecordInfoFromGuids(*guids)
            vartype = VT_RECORD
        elif issubclass(itemtype, POINTER(IDispatch)):
            vartype = VT_DISPATCH
            extra = pointer(itemtype._iid_)
        elif issubclass(itemtype, POINTER(IUnknown)):
            vartype = VT_UNKNOWN
            extra = pointer(itemtype._iid_)
        else:
            raise TypeError(itemtype)
    @Patch(POINTER(sa_type))
    class _(object):
        _itemtype_ = itemtype
        _vartype_ = vartype
        _needsfree = False
        @classmethod
        def create(cls, value, extra=None):
            if npsupport.isndarray(value):
                return cls.create_from_ndarray(value, extra)
            pa = _safearray.SafeArrayCreateVectorEx(cls._vartype_,
                                                    0,
                                                    len(value),
                                                    extra)
            if not pa:
                if cls._vartype_ == VT_RECORD and extra is None:
                    raise TypeError("Cannot create SAFEARRAY type VT_RECORD without IRecordInfo.")
                raise MemoryError()
            pa = cast(pa, cls)
            ptr = POINTER(cls._itemtype_)()
            _safearray.SafeArrayAccessData(pa, byref(ptr))
            try:
                if isinstance(value, array.array):
                    addr, n = value.buffer_info()
                    nbytes = len(value) * sizeof(cls._itemtype_)
                    memmove(ptr, addr, nbytes)
                else:
                    for index, item in enumerate(value):
                        ptr[index] = item
            finally:
                _safearray.SafeArrayUnaccessData(pa)
            return pa
        @classmethod
        def create_from_ndarray(cls, value, extra, lBound=0):
            from comtypes.automation import VARIANT
            if cls._itemtype_ is VARIANT:
                if value.dtype != npsupport.VARIANT_dtype:
                    value = _ndarray_to_variant_array(value)
            else:
                ai = value.__array_interface__
                if ai["version"] != 3:
                    raise TypeError("only __array_interface__ version 3 supported")
                if cls._itemtype_ != npsupport.typecodes[ai["typestr"]]:
                    raise TypeError("Wrong array item type")
            if not value.flags.f_contiguous:
                value = numpy.array(value, order="F")
            rgsa = (_safearray.SAFEARRAYBOUND * value.ndim)()
            nitems = 1
            for i, d in enumerate(value.shape):
                nitems *= d
                rgsa[i].cElements = d
                rgsa[i].lBound = lBound
            pa = _safearray.SafeArrayCreateEx(cls._vartype_,
                                              value.ndim,
                                              rgsa,
                                              extra)
            if not pa:
                if cls._vartype_ == VT_RECORD and extra is None:
                    raise TypeError("Cannot create SAFEARRAY type VT_RECORD without IRecordInfo.")
                raise MemoryError()
            pa = cast(pa, cls)
            ptr = POINTER(cls._itemtype_)()
            _safearray.SafeArrayAccessData(pa, byref(ptr))
            try:
                nbytes = nitems * sizeof(cls._itemtype_)
                memmove(ptr, value.ctypes.data, nbytes)
            finally:
                _safearray.SafeArrayUnaccessData(pa)
            return pa
        @classmethod
        def from_param(cls, value):
            if not isinstance(value, cls):
                value = cls.create(value, extra)
                value._needsfree = True
            return value
        def __getitem__(self, index):
            if index != 0:
                raise IndexError("Only index 0 allowed")
            return self.unpack()
        def __setitem__(self, index, value):
            raise TypeError("Setting items not allowed")
        def __ctypes_from_outparam__(self):
            self._needsfree = True
            return self[0]
        def __del__(self, _SafeArrayDestroy=_safearray.SafeArrayDestroy):
            if self._needsfree:
                _SafeArrayDestroy(self)
        def _get_size(self, dim):
            ub = _safearray.SafeArrayGetUBound(self, dim) + 1
            lb = _safearray.SafeArrayGetLBound(self, dim)
            return ub - lb
        def unpack(self):
            dim = _safearray.SafeArrayGetDim(self)
            if dim == 1:
                num_elements = self._get_size(1)
                result = self._get_elements_raw(num_elements)
                if safearray_as_ndarray:
                    import numpy
                    return numpy.asarray(result)
                return tuple(result)
            elif dim == 2:
                rows, cols = self._get_size(1), self._get_size(2)
                result = self._get_elements_raw(rows * cols)
                if safearray_as_ndarray:
                    import numpy
                    return numpy.asarray(result).reshape((cols, rows)).T
                result = [tuple(result[r::rows]) for r in range(rows)]
                return tuple(result)
            else:
                lowerbounds = [_safearray.SafeArrayGetLBound(self, d)
                               for d in range(1, dim+1)]
                indexes = (c_long * dim)(*lowerbounds)
                upperbounds = [_safearray.SafeArrayGetUBound(self, d)
                               for d in range(1, dim+1)]
                row = self._get_row(0, indexes, lowerbounds, upperbounds)
                if safearray_as_ndarray:
                    import numpy
                    return numpy.asarray(row)
                return row
        def _get_elements_raw(self, num_elements):
            from comtypes.automation import VARIANT
            ptr = POINTER(self._itemtype_)()
            _safearray.SafeArrayAccessData(self, byref(ptr))
            try:
                if self._itemtype_ == VARIANT:
                    return [i.value for i in ptr[:num_elements]]
                elif issubclass(self._itemtype_, POINTER(IUnknown)):
                    iid = _safearray.SafeArrayGetIID(self)
                    itf = com_interface_registry[str(iid)]
                    elems = ptr[:num_elements]
                    result = []
                    for p in elems:
                        if bool(p):
                            p.AddRef()
                            result.append(p.QueryInterface(itf))
                        else:
                            result.append(POINTER(itf)())
                    return result
                else:
                    if not issubclass(self._itemtype_, Structure):
                        if (safearray_as_ndarray and self._itemtype_ in
                                list(npsupport.typecodes.values())):
                            arr = numpy.ctypeslib.as_array(ptr,
                                                           (num_elements,))
                            return arr.copy()
                        return ptr[:num_elements]
                    def keep_safearray(v):
                        v.__keepref = self
                        return v
                    return [keep_safearray(x) for x in ptr[:num_elements]]
            finally:
                _safearray.SafeArrayUnaccessData(self)
        def _get_row(self, dim, indices, lowerbounds, upperbounds):
            restore = indices[dim]
            result = []
            obj = self._itemtype_()
            pobj = byref(obj)
            if dim+1 == len(indices):
                for i in range(indices[dim], upperbounds[dim]+1):
                    indices[dim] = i
                    _safearray.SafeArrayGetElement(self, indices, pobj)
                    result.append(obj.value)
            else:
                for i in range(indices[dim], upperbounds[dim]+1):
                    indices[dim] = i
                    result.append(self._get_row(dim+1, indices, lowerbounds, upperbounds))
            indices[dim] = restore
            return tuple(result)
    @Patch(POINTER(POINTER(sa_type)))
    class __(object):
        @classmethod
        def from_param(cls, value):
            if isinstance(value, cls._type_):
                return byref(value)
            return byref(cls._type_.create(value, extra))
        def __setitem__(self, index, value):
            pa = self._type_.create(value, extra)
            super(POINTER(POINTER(sa_type)), self).__setitem__(index, pa)
    return sa_type
def _ndarray_to_variant_array(value):
    if npsupport.VARIANT_dtype is None:
        msg = "VARIANT ndarrays require NumPy 1.7 or newer."
        raise RuntimeError(msg)
    if numpy.issubdtype(value.dtype, npsupport.datetime64):
        return _datetime64_ndarray_to_variant_array(value)
    from comtypes.automation import VARIANT
    varr = numpy.zeros(value.shape, npsupport.VARIANT_dtype, order='F')
    varr.flat = [VARIANT(v) for v in value.flat]
    return varr
def _datetime64_ndarray_to_variant_array(value):
    from comtypes.automation import VT_DATE
    value = numpy.array(value, "datetime64[ns]")
    value = value - npsupport.com_null_date64
    value = value / numpy.timedelta64(1, 'D')
    varr = numpy.zeros(value.shape, npsupport.VARIANT_dtype, order='F')
    varr['vt'] = VT_DATE
    varr['_']['VT_R8'].flat = value.flat
    return varr
