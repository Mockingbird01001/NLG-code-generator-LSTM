
import warnings
import numpy as np
from numpy import (
        bool_, dtype, ndarray, recarray, array as narray
        )
from numpy.core.records import (
        fromarrays as recfromarrays, fromrecords as recfromrecords
        )
_byteorderconv = np.core.records._byteorderconv
import numpy.ma as ma
from numpy.ma import (
        MAError, MaskedArray, masked, nomask, masked_array, getdata,
        getmaskarray, filled
        )
_check_fill_value = ma.core._check_fill_value
__all__ = [
    'MaskedRecords', 'mrecarray', 'fromarrays', 'fromrecords',
    'fromtextfile', 'addfield',
    ]
reserved_fields = ['_data', '_mask', '_fieldmask', 'dtype']
def _checknames(descr, names=None):
    ndescr = len(descr)
    default_names = ['f%i' % i for i in range(ndescr)]
    if names is None:
        new_names = default_names
    else:
        if isinstance(names, (tuple, list)):
            new_names = names
        elif isinstance(names, str):
            new_names = names.split(',')
        else:
            raise NameError(f'illegal input names {names!r}')
        nnames = len(new_names)
        if nnames < ndescr:
            new_names += default_names[nnames:]
    ndescr = []
    for (n, d, t) in zip(new_names, default_names, descr.descr):
        if n in reserved_fields:
            if t[0] in reserved_fields:
                ndescr.append((d, t[1]))
            else:
                ndescr.append(t)
        else:
            ndescr.append((n, t[1]))
    return np.dtype(ndescr)
def _get_fieldmask(self):
    mdescr = [(n, '|b1') for n in self.dtype.names]
    fdmask = np.empty(self.shape, dtype=mdescr)
    fdmask.flat = tuple([False] * len(mdescr))
    return fdmask
class MaskedRecords(MaskedArray):
    def __new__(cls, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False,
                mask=nomask, hard_mask=False, fill_value=None, keep_mask=True,
                copy=False,
                **options):
        self = recarray.__new__(cls, shape, dtype=dtype, buf=buf, offset=offset,
                                strides=strides, formats=formats, names=names,
                                titles=titles, byteorder=byteorder,
                                aligned=aligned,)
        mdtype = ma.make_mask_descr(self.dtype)
        if mask is nomask or not np.size(mask):
            if not keep_mask:
                self._mask = tuple([False] * len(mdtype))
        else:
            mask = np.array(mask, copy=copy)
            if mask.shape != self.shape:
                (nd, nm) = (self.size, mask.size)
                if nm == 1:
                    mask = np.resize(mask, self.shape)
                elif nm == nd:
                    mask = np.reshape(mask, self.shape)
                else:
                    msg = "Mask and data not compatible: data size is %i, " +                          "mask size is %i."
                    raise MAError(msg % (nd, nm))
                copy = True
            if not keep_mask:
                self.__setmask__(mask)
                self._sharedmask = True
            else:
                if mask.dtype == mdtype:
                    _mask = mask
                else:
                    _mask = np.array([tuple([m] * len(mdtype)) for m in mask],
                                     dtype=mdtype)
                self._mask = _mask
        return self
    def __array_finalize__(self, obj):
        _mask = getattr(obj, '_mask', None)
        if _mask is None:
            objmask = getattr(obj, '_mask', nomask)
            _dtype = ndarray.__getattribute__(self, 'dtype')
            if objmask is nomask:
                _mask = ma.make_mask_none(self.shape, dtype=_dtype)
            else:
                mdescr = ma.make_mask_descr(_dtype)
                _mask = narray([tuple([m] * len(mdescr)) for m in objmask],
                               dtype=mdescr).view(recarray)
        _dict = self.__dict__
        _dict.update(_mask=_mask)
        self._update_from(obj)
        if _dict['_baseclass'] == ndarray:
            _dict['_baseclass'] = recarray
        return
    @property
    def _data(self):
        return ndarray.view(self, recarray)
    @property
    def _fieldmask(self):
        return self._mask
    def __len__(self):
        if self.ndim:
            return len(self._data)
        return len(self.dtype)
    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(f'record array has no attribute {attr}') from e
        _localdict = ndarray.__getattribute__(self, '__dict__')
        _data = ndarray.view(self, _localdict['_baseclass'])
        obj = _data.getfield(*res)
        if obj.dtype.names is not None:
            raise NotImplementedError("MaskedRecords is currently limited to"
                                      "simple records.")
        hasmasked = False
        _mask = _localdict.get('_mask', None)
        if _mask is not None:
            try:
                _mask = _mask[attr]
            except IndexError:
                pass
            tp_len = len(_mask.dtype)
            hasmasked = _mask.view((bool, ((tp_len,) if tp_len else ()))).any()
        if (obj.shape or hasmasked):
            obj = obj.view(MaskedArray)
            obj._baseclass = ndarray
            obj._isfield = True
            obj._mask = _mask
            _fill_value = _localdict.get('_fill_value', None)
            if _fill_value is not None:
                try:
                    obj._fill_value = _fill_value[attr]
                except ValueError:
                    obj._fill_value = None
        else:
            obj = obj.item()
        return obj
    def __setattr__(self, attr, val):
        if attr in ['mask', 'fieldmask']:
            self.__setmask__(val)
            return
        _localdict = object.__getattribute__(self, '__dict__')
        newattr = attr not in _localdict
        try:
            ret = object.__setattr__(self, attr, val)
        except Exception:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            optinfo = ndarray.__getattribute__(self, '_optinfo') or {}
            if not (attr in fielddict or attr in optinfo):
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
        except (TypeError, KeyError):
            raise AttributeError(f'record array has no attribute {attr}')
        if val is masked:
            _fill_value = _localdict['_fill_value']
            if _fill_value is not None:
                dval = _localdict['_fill_value'][attr]
            else:
                dval = val
            mval = True
        else:
            dval = filled(val)
            mval = getmaskarray(val)
        obj = ndarray.__getattribute__(self, '_data').setfield(dval, *res)
        _localdict['_mask'].__setitem__(attr, mval)
        return obj
    def __getitem__(self, indx):
        _localdict = self.__dict__
        _mask = ndarray.__getattribute__(self, '_mask')
        _data = ndarray.view(self, _localdict['_baseclass'])
        if isinstance(indx, str):
            obj = _data[indx].view(MaskedArray)
            obj._mask = _mask[indx]
            obj._sharedmask = True
            fval = _localdict['_fill_value']
            if fval is not None:
                obj._fill_value = fval[indx]
            if not obj.ndim and obj._mask:
                return masked
            return obj
        obj = np.array(_data[indx], copy=False).view(mrecarray)
        obj._mask = np.array(_mask[indx], copy=False).view(recarray)
        return obj
    def __setitem__(self, indx, value):
        MaskedArray.__setitem__(self, indx, value)
        if isinstance(indx, str):
            self._mask[indx] = ma.getmaskarray(value)
    def __str__(self):
        if self.size > 1:
            mstr = [f"({','.join([str(i) for i in s])})"
                    for s in zip(*[getattr(self, f) for f in self.dtype.names])]
            return f"[{', '.join(mstr)}]"
        else:
            mstr = [f"{','.join([str(i) for i in s])}"
                    for s in zip([getattr(self, f) for f in self.dtype.names])]
            return f"({', '.join(mstr)})"
    def __repr__(self):
        _names = self.dtype.names
        fmt = "%%%is : %%s" % (max([len(n) for n in _names]) + 4,)
        reprstr = [fmt % (f, getattr(self, f)) for f in self.dtype.names]
        reprstr.insert(0, 'masked_records(')
        reprstr.extend([fmt % ('    fill_value', self.fill_value),
                         '              )'])
        return str("\n".join(reprstr))
    def view(self, dtype=None, type=None):
        if dtype is None:
            if type is None:
                output = ndarray.view(self)
            else:
                output = ndarray.view(self, type)
        elif type is None:
            try:
                if issubclass(dtype, ndarray):
                    output = ndarray.view(self, dtype)
                    dtype = None
                else:
                    output = ndarray.view(self, dtype)
            except TypeError:
                dtype = np.dtype(dtype)
                if dtype.fields is None:
                    basetype = self.__class__.__bases__[0]
                    output = self.__array__().view(dtype, basetype)
                    output._update_from(self)
                else:
                    output = ndarray.view(self, dtype)
                output._fill_value = None
        else:
            output = ndarray.view(self, dtype, type)
        if (getattr(output, '_mask', nomask) is not nomask):
            mdtype = ma.make_mask_descr(output.dtype)
            output._mask = self._mask.view(mdtype, ndarray)
            output._mask.shape = output.shape
        return output
    def harden_mask(self):
        self._hardmask = True
    def soften_mask(self):
        self._hardmask = False
    def copy(self):
        copied = self._data.copy().view(type(self))
        copied._mask = self._mask.copy()
        return copied
    def tolist(self, fill_value=None):
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        result = narray(self.filled().tolist(), dtype=object)
        mask = narray(self._mask.tolist())
        result[mask] = None
        return result.tolist()
    def __getstate__(self):
        state = (1,
                 self.shape,
                 self.dtype,
                 self.flags.fnc,
                 self._data.tobytes(),
                 self._mask.tobytes(),
                 self._fill_value,
                 )
        return state
    def __setstate__(self, state):
        (ver, shp, typ, isf, raw, msk, flv) = state
        ndarray.__setstate__(self, (shp, typ, isf, raw))
        mdtype = dtype([(k, bool_) for (k, _) in self.dtype.descr])
        self.__dict__['_mask'].__setstate__((shp, mdtype, isf, msk))
        self.fill_value = flv
    def __reduce__(self):
        return (_mrreconstruct,
                (self.__class__, self._baseclass, (0,), 'b',),
                self.__getstate__())
def _mrreconstruct(subtype, baseclass, baseshape, basetype,):
    _data = ndarray.__new__(baseclass, baseshape, basetype).view(subtype)
    _mask = ndarray.__new__(ndarray, baseshape, 'b1')
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)
mrecarray = MaskedRecords
def fromarrays(arraylist, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None,
               fill_value=None):
    datalist = [getdata(x) for x in arraylist]
    masklist = [np.atleast_1d(getmaskarray(x)) for x in arraylist]
    _array = recfromarrays(datalist,
                           dtype=dtype, shape=shape, formats=formats,
                           names=names, titles=titles, aligned=aligned,
                           byteorder=byteorder).view(mrecarray)
    _array._mask.flat = list(zip(*masklist))
    if fill_value is not None:
        _array.fill_value = fill_value
    return _array
def fromrecords(reclist, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None,
                fill_value=None, mask=nomask):
    _mask = getattr(reclist, '_mask', None)
    if isinstance(reclist, ndarray):
        if isinstance(reclist, MaskedArray):
            reclist = reclist.filled().view(ndarray)
        if dtype is None:
            dtype = reclist.dtype
        reclist = reclist.tolist()
    mrec = recfromrecords(reclist, dtype=dtype, shape=shape, formats=formats,
                          names=names, titles=titles,
                          aligned=aligned, byteorder=byteorder).view(mrecarray)
    if fill_value is not None:
        mrec.fill_value = fill_value
    if mask is not nomask:
        mask = np.array(mask, copy=False)
        maskrecordlength = len(mask.dtype)
        if maskrecordlength:
            mrec._mask.flat = mask
        elif mask.ndim == 2:
            mrec._mask.flat = [tuple(m) for m in mask]
        else:
            mrec.__setmask__(mask)
    if _mask is not None:
        mrec._mask[:] = _mask
    return mrec
def _guessvartypes(arr):
    vartypes = []
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[0]
    elif arr.ndim > 2:
        raise ValueError("The array should be 2D at most!")
    for f in arr:
        try:
            int(f)
        except (ValueError, TypeError):
            try:
                float(f)
            except (ValueError, TypeError):
                try:
                    complex(f)
                except (ValueError, TypeError):
                    vartypes.append(arr.dtype)
                else:
                    vartypes.append(np.dtype(complex))
            else:
                vartypes.append(np.dtype(float))
        else:
            vartypes.append(np.dtype(int))
    return vartypes
def openfile(fname):
    if hasattr(fname, 'readline'):
        return fname
    try:
        f = open(fname)
    except IOError:
        raise IOError(f"No such file: '{fname}'")
    if f.readline()[:2] != "\\x":
        f.seek(0, 0)
        return f
    f.close()
    raise NotImplementedError("Wow, binary file")
                 varnames=None, vartypes=None):
    ftext = openfile(fname)
    while True:
        line = ftext.readline()
        firstline = line[:line.find(commentchar)].strip()
        _varnames = firstline.split(delimitor)
        if len(_varnames) > 1:
            break
    if varnames is None:
        varnames = _varnames
    _variables = masked_array([line.strip().split(delimitor) for line in ftext
                               if line[0] != commentchar and len(line) > 1])
    (_, nfields) = _variables.shape
    ftext.close()
    if vartypes is None:
        vartypes = _guessvartypes(_variables[0])
    else:
        vartypes = [np.dtype(v) for v in vartypes]
        if len(vartypes) != nfields:
            msg = "Attempting to %i dtypes for %i fields!"
            msg += " Reverting to default."
            warnings.warn(msg % (len(vartypes), nfields), stacklevel=2)
            vartypes = _guessvartypes(_variables[0])
    mdescr = [(n, f) for (n, f) in zip(varnames, vartypes)]
    mfillv = [ma.default_fill_value(f) for f in vartypes]
    _mask = (_variables.T == missingchar)
    _datalist = [masked_array(a, mask=m, dtype=t, fill_value=f)
                 for (a, m, t, f) in zip(_variables.T, _mask, vartypes, mfillv)]
    return fromarrays(_datalist, dtype=mdescr)
def addfield(mrecord, newfield, newfieldname=None):
    _data = mrecord._data
    _mask = mrecord._mask
    if newfieldname is None or newfieldname in reserved_fields:
        newfieldname = 'f%i' % len(_data.dtype)
    newfield = ma.array(newfield)
    newdtype = np.dtype(_data.dtype.descr + [(newfieldname, newfield.dtype)])
    newdata = recarray(_data.shape, newdtype)
    [newdata.setfield(_data.getfield(*f), *f)
         for f in _data.dtype.fields.values()]
    newdata.setfield(newfield._data, *newdata.dtype.fields[newfieldname])
    newdata = newdata.view(MaskedRecords)
    newmdtype = np.dtype([(n, bool_) for n in newdtype.names])
    newmask = recarray(_data.shape, newmdtype)
    [newmask.setfield(_mask.getfield(*f), *f)
         for f in _mask.dtype.fields.values()]
    newmask.setfield(getmaskarray(newfield),
                     *newmask.dtype.fields[newfieldname])
    newdata._mask = newmask
    return newdata
