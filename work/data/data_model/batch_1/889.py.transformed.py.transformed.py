
import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
from numpy.testing import suppress_warnings
_check_fill_value = np.ma.core._check_fill_value
__all__ = [
    'append_fields', 'apply_along_fields', 'assign_fields_by_name',
    'drop_fields', 'find_duplicates', 'flatten_descr',
    'get_fieldstructure', 'get_names', 'get_names_flat',
    'join_by', 'merge_arrays', 'rec_append_fields',
    'rec_drop_fields', 'rec_join', 'recursive_fill_fields',
    'rename_fields', 'repack_fields', 'require_fields',
    'stack_arrays', 'structured_to_unstructured', 'unstructured_to_structured',
    ]
def _recursive_fill_fields_dispatcher(input, output):
    return (input, output)
@array_function_dispatch(_recursive_fill_fields_dispatcher)
def recursive_fill_fields(input, output):
    newdtype = output.dtype
    for field in newdtype.names:
        try:
            current = input[field]
        except ValueError:
            continue
        if current.dtype.names is not None:
            recursive_fill_fields(current, output[field])
        else:
            output[field][:len(current)] = current
    return output
def _get_fieldspec(dtype):
    if dtype.names is None:
        return [('', dtype)]
    else:
        fields = ((name, dtype.fields[name]) for name in dtype.names)
        return [
            (name if len(f) == 2 else (f[2], name), f[0])
            for name, f in fields
        ]
def get_names(adtype):
    listnames = []
    names = adtype.names
    for name in names:
        current = adtype[name]
        if current.names is not None:
            listnames.append((name, tuple(get_names(current))))
        else:
            listnames.append(name)
    return tuple(listnames)
def get_names_flat(adtype):
    listnames = []
    names = adtype.names
    for name in names:
        listnames.append(name)
        current = adtype[name]
        if current.names is not None:
            listnames.extend(get_names_flat(current))
    return tuple(listnames)
def flatten_descr(ndtype):
    names = ndtype.names
    if names is None:
        return (('', ndtype),)
    else:
        descr = []
        for field in names:
            (typ, _) = ndtype.fields[field]
            if typ.names is not None:
                descr.extend(flatten_descr(typ))
            else:
                descr.append((field, typ))
        return tuple(descr)
def _zip_dtype(seqarrays, flatten=False):
    newdtype = []
    if flatten:
        for a in seqarrays:
            newdtype.extend(flatten_descr(a.dtype))
    else:
        for a in seqarrays:
            current = a.dtype
            if current.names is not None and len(current.names) == 1:
                newdtype.extend(_get_fieldspec(current))
            else:
                newdtype.append(('', current))
    return np.dtype(newdtype)
def _zip_descr(seqarrays, flatten=False):
    return _zip_dtype(seqarrays, flatten=flatten).descr
def get_fieldstructure(adtype, lastname=None, parents=None,):
    if parents is None:
        parents = {}
    names = adtype.names
    for name in names:
        current = adtype[name]
        if current.names is not None:
            if lastname:
                parents[name] = [lastname, ]
            else:
                parents[name] = []
            parents.update(get_fieldstructure(current, name, parents))
        else:
            lastparent = [_ for _ in (parents.get(lastname, []) or [])]
            if lastparent:
                lastparent.append(lastname)
            elif lastname:
                lastparent = [lastname, ]
            parents[name] = lastparent or []
    return parents
def _izip_fields_flat(iterable):
    for element in iterable:
        if isinstance(element, np.void):
            yield from _izip_fields_flat(tuple(element))
        else:
            yield element
def _izip_fields(iterable):
    for element in iterable:
        if (hasattr(element, '__iter__') and
                not isinstance(element, str)):
            yield from _izip_fields(element)
        elif isinstance(element, np.void) and len(tuple(element)) == 1:
            yield from _izip_fields(element)
        else:
            yield element
def _izip_records(seqarrays, fill_value=None, flatten=True):
    if flatten:
        zipfunc = _izip_fields_flat
    else:
        zipfunc = _izip_fields
    for tup in itertools.zip_longest(*seqarrays, fillvalue=fill_value):
        yield tuple(zipfunc(tup))
def _fix_output(output, usemask=True, asrecarray=False):
    if not isinstance(output, MaskedArray):
        usemask = False
    if usemask:
        if asrecarray:
            output = output.view(MaskedRecords)
    else:
        output = ma.filled(output)
        if asrecarray:
            output = output.view(recarray)
    return output
def _fix_defaults(output, defaults=None):
    names = output.dtype.names
    (data, mask, fill_value) = (output.data, output.mask, output.fill_value)
    for (k, v) in (defaults or {}).items():
        if k in names:
            fill_value[k] = v
            data[k][mask[k]] = v
    return output
def _merge_arrays_dispatcher(seqarrays, fill_value=None, flatten=None,
                             usemask=None, asrecarray=None):
    return seqarrays
@array_function_dispatch(_merge_arrays_dispatcher)
def merge_arrays(seqarrays, fill_value=-1, flatten=False,
                 usemask=False, asrecarray=False):
    if (len(seqarrays) == 1):
        seqarrays = np.asanyarray(seqarrays[0])
    if isinstance(seqarrays, (ndarray, np.void)):
        seqdtype = seqarrays.dtype
        if seqdtype.names is None:
            seqdtype = np.dtype([('', seqdtype)])
        if not flatten or _zip_dtype((seqarrays,), flatten=True) == seqdtype:
            seqarrays = seqarrays.ravel()
            if usemask:
                if asrecarray:
                    seqtype = MaskedRecords
                else:
                    seqtype = MaskedArray
            elif asrecarray:
                seqtype = recarray
            else:
                seqtype = ndarray
            return seqarrays.view(dtype=seqdtype, type=seqtype)
        else:
            seqarrays = (seqarrays,)
    else:
        seqarrays = [np.asanyarray(_m) for _m in seqarrays]
    sizes = tuple(a.size for a in seqarrays)
    maxlength = max(sizes)
    newdtype = _zip_dtype(seqarrays, flatten=flatten)
    seqdata = []
    seqmask = []
    if usemask:
        for (a, n) in zip(seqarrays, sizes):
            nbmissing = (maxlength - n)
            data = a.ravel().__array__()
            mask = ma.getmaskarray(a).ravel()
            if nbmissing:
                fval = _check_fill_value(fill_value, a.dtype)
                if isinstance(fval, (ndarray, np.void)):
                    if len(fval.dtype) == 1:
                        fval = fval.item()[0]
                        fmsk = True
                    else:
                        fval = np.array(fval, dtype=a.dtype, ndmin=1)
                        fmsk = np.ones((1,), dtype=mask.dtype)
            else:
                fval = None
                fmsk = True
            seqdata.append(itertools.chain(data, [fval] * nbmissing))
            seqmask.append(itertools.chain(mask, [fmsk] * nbmissing))
        data = tuple(_izip_records(seqdata, flatten=flatten))
        output = ma.array(np.fromiter(data, dtype=newdtype, count=maxlength),
                          mask=list(_izip_records(seqmask, flatten=flatten)))
        if asrecarray:
            output = output.view(MaskedRecords)
    else:
        for (a, n) in zip(seqarrays, sizes):
            nbmissing = (maxlength - n)
            data = a.ravel().__array__()
            if nbmissing:
                fval = _check_fill_value(fill_value, a.dtype)
                if isinstance(fval, (ndarray, np.void)):
                    if len(fval.dtype) == 1:
                        fval = fval.item()[0]
                    else:
                        fval = np.array(fval, dtype=a.dtype, ndmin=1)
            else:
                fval = None
            seqdata.append(itertools.chain(data, [fval] * nbmissing))
        output = np.fromiter(tuple(_izip_records(seqdata, flatten=flatten)),
                             dtype=newdtype, count=maxlength)
        if asrecarray:
            output = output.view(recarray)
    return output
def _drop_fields_dispatcher(base, drop_names, usemask=None, asrecarray=None):
    return (base,)
@array_function_dispatch(_drop_fields_dispatcher)
def drop_fields(base, drop_names, usemask=True, asrecarray=False):
    if _is_string_like(drop_names):
        drop_names = [drop_names]
    else:
        drop_names = set(drop_names)
    def _drop_descr(ndtype, drop_names):
        names = ndtype.names
        newdtype = []
        for name in names:
            current = ndtype[name]
            if name in drop_names:
                continue
            if current.names is not None:
                descr = _drop_descr(current, drop_names)
                if descr:
                    newdtype.append((name, descr))
            else:
                newdtype.append((name, current))
        return newdtype
    newdtype = _drop_descr(base.dtype, drop_names)
    output = np.empty(base.shape, dtype=newdtype)
    output = recursive_fill_fields(base, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)
def _keep_fields(base, keep_names, usemask=True, asrecarray=False):
    newdtype = [(n, base.dtype[n]) for n in keep_names]
    output = np.empty(base.shape, dtype=newdtype)
    output = recursive_fill_fields(base, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)
def _rec_drop_fields_dispatcher(base, drop_names):
    return (base,)
@array_function_dispatch(_rec_drop_fields_dispatcher)
def rec_drop_fields(base, drop_names):
    return drop_fields(base, drop_names, usemask=False, asrecarray=True)
def _rename_fields_dispatcher(base, namemapper):
    return (base,)
@array_function_dispatch(_rename_fields_dispatcher)
def rename_fields(base, namemapper):
    def _recursive_rename_fields(ndtype, namemapper):
        newdtype = []
        for name in ndtype.names:
            newname = namemapper.get(name, name)
            current = ndtype[name]
            if current.names is not None:
                newdtype.append(
                    (newname, _recursive_rename_fields(current, namemapper))
                    )
            else:
                newdtype.append((newname, current))
        return newdtype
    newdtype = _recursive_rename_fields(base.dtype, namemapper)
    return base.view(newdtype)
def _append_fields_dispatcher(base, names, data, dtypes=None,
                              fill_value=None, usemask=None, asrecarray=None):
    yield base
    yield from data
@array_function_dispatch(_append_fields_dispatcher)
def append_fields(base, names, data, dtypes=None,
                  fill_value=-1, usemask=True, asrecarray=False):
    if isinstance(names, (tuple, list)):
        if len(names) != len(data):
            msg = "The number of arrays does not match the number of names"
            raise ValueError(msg)
    elif isinstance(names, str):
        names = [names, ]
        data = [data, ]
    if dtypes is None:
        data = [np.array(a, copy=False, subok=True) for a in data]
        data = [a.view([(name, a.dtype)]) for (name, a) in zip(names, data)]
    else:
        if not isinstance(dtypes, (tuple, list)):
            dtypes = [dtypes, ]
        if len(data) != len(dtypes):
            if len(dtypes) == 1:
                dtypes = dtypes * len(data)
            else:
                msg = "The dtypes argument must be None, a dtype, or a list."
                raise ValueError(msg)
        data = [np.array(a, copy=False, subok=True, dtype=d).view([(n, d)])
                for (a, n, d) in zip(data, names, dtypes)]
    base = merge_arrays(base, usemask=usemask, fill_value=fill_value)
    if len(data) > 1:
        data = merge_arrays(data, flatten=True, usemask=usemask,
                            fill_value=fill_value)
    else:
        data = data.pop()
    output = ma.masked_all(
        max(len(base), len(data)),
        dtype=_get_fieldspec(base.dtype) + _get_fieldspec(data.dtype))
    output = recursive_fill_fields(base, output)
    output = recursive_fill_fields(data, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)
def _rec_append_fields_dispatcher(base, names, data, dtypes=None):
    yield base
    yield from data
@array_function_dispatch(_rec_append_fields_dispatcher)
def rec_append_fields(base, names, data, dtypes=None):
    return append_fields(base, names, data=data, dtypes=dtypes,
                         asrecarray=True, usemask=False)
def _repack_fields_dispatcher(a, align=None, recurse=None):
    return (a,)
@array_function_dispatch(_repack_fields_dispatcher)
def repack_fields(a, align=False, recurse=False):
    if not isinstance(a, np.dtype):
        dt = repack_fields(a.dtype, align=align, recurse=recurse)
        return a.astype(dt, copy=False)
    if a.names is None:
        return a
    fieldinfo = []
    for name in a.names:
        tup = a.fields[name]
        if recurse:
            fmt = repack_fields(tup[0], align=align, recurse=True)
        else:
            fmt = tup[0]
        if len(tup) == 3:
            name = (tup[2], name)
        fieldinfo.append((name, fmt))
    dt = np.dtype(fieldinfo, align=align)
    return np.dtype((a.type, dt))
def _get_fields_and_offsets(dt, offset=0):
    def count_elem(dt):
        count = 1
        while dt.shape != ():
            for size in dt.shape:
                count *= size
            dt = dt.base
        return dt, count
    fields = []
    for name in dt.names:
        field = dt.fields[name]
        f_dt, f_offset = field[0], field[1]
        f_dt, n = count_elem(f_dt)
        if f_dt.names is None:
            fields.append((np.dtype((f_dt, (n,))), n, f_offset + offset))
        else:
            subfields = _get_fields_and_offsets(f_dt, f_offset + offset)
            size = f_dt.itemsize
            for i in range(n):
                if i == 0:
                    fields.extend(subfields)
                else:
                    fields.extend([(d, c, o + i*size) for d, c, o in subfields])
    return fields
def _structured_to_unstructured_dispatcher(arr, dtype=None, copy=None,
                                           casting=None):
    return (arr,)
@array_function_dispatch(_structured_to_unstructured_dispatcher)
def structured_to_unstructured(arr, dtype=None, copy=False, casting='unsafe'):
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')
    fields = _get_fields_and_offsets(arr.dtype)
    n_fields = len(fields)
    if n_fields == 0 and dtype is None:
        raise ValueError("arr has no fields. Unable to guess dtype")
    elif n_fields == 0:
        raise NotImplementedError("arr with no fields is not supported")
    dts, counts, offsets = zip(*fields)
    names = ['f{}'.format(n) for n in range(n_fields)]
    if dtype is None:
        out_dtype = np.result_type(*[dt.base for dt in dts])
    else:
        out_dtype = dtype
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': arr.dtype.itemsize})
    with suppress_warnings() as sup:
        sup.filter(FutureWarning, "Numpy has detected")
        arr = arr.view(flattened_fields)
    packed_fields = np.dtype({'names': names,
                              'formats': [(out_dtype, dt.shape) for dt in dts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)
    return arr.view((out_dtype, (sum(counts),)))
def _unstructured_to_structured_dispatcher(arr, dtype=None, names=None,
                                           align=None, copy=None, casting=None):
    return (arr,)
@array_function_dispatch(_unstructured_to_structured_dispatcher)
def unstructured_to_structured(arr, dtype=None, names=None, align=False,
                               copy=False, casting='unsafe'):
    if arr.shape == ():
        raise ValueError('arr must have at least one dimension')
    n_elem = arr.shape[-1]
    if n_elem == 0:
        raise NotImplementedError("last axis with size 0 is not supported")
    if dtype is None:
        if names is None:
            names = ['f{}'.format(n) for n in range(n_elem)]
        out_dtype = np.dtype([(n, arr.dtype) for n in names], align=align)
        fields = _get_fields_and_offsets(out_dtype)
        dts, counts, offsets = zip(*fields)
    else:
        if names is not None:
            raise ValueError("don't supply both dtype and names")
        fields = _get_fields_and_offsets(dtype)
        if len(fields) == 0:
            dts, counts, offsets = [], [], []
        else:
            dts, counts, offsets = zip(*fields)
        if n_elem != sum(counts):
            raise ValueError('The length of the last dimension of arr must '
                             'be equal to the number of fields in dtype')
        out_dtype = dtype
        if align and not out_dtype.isalignedstruct:
            raise ValueError("align was True but dtype is not aligned")
    names = ['f{}'.format(n) for n in range(len(fields))]
    packed_fields = np.dtype({'names': names,
                              'formats': [(arr.dtype, dt.shape) for dt in dts]})
    arr = np.ascontiguousarray(arr).view(packed_fields)
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': out_dtype.itemsize})
    arr = arr.astype(flattened_fields, copy=copy, casting=casting)
    return arr.view(out_dtype)[..., 0]
def _apply_along_fields_dispatcher(func, arr):
    return (arr,)
@array_function_dispatch(_apply_along_fields_dispatcher)
def apply_along_fields(func, arr):
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')
    uarr = structured_to_unstructured(arr)
    return func(uarr, axis=-1)
def _assign_fields_by_name_dispatcher(dst, src, zero_unassigned=None):
    return dst, src
@array_function_dispatch(_assign_fields_by_name_dispatcher)
def assign_fields_by_name(dst, src, zero_unassigned=True):
    if dst.dtype.names is None:
        dst[...] = src
        return
    for name in dst.dtype.names:
        if name not in src.dtype.names:
            if zero_unassigned:
                dst[name] = 0
        else:
            assign_fields_by_name(dst[name], src[name],
                                  zero_unassigned)
def _require_fields_dispatcher(array, required_dtype):
    return (array,)
@array_function_dispatch(_require_fields_dispatcher)
def require_fields(array, required_dtype):
    out = np.empty(array.shape, dtype=required_dtype)
    assign_fields_by_name(out, array)
    return out
def _stack_arrays_dispatcher(arrays, defaults=None, usemask=None,
                             asrecarray=None, autoconvert=None):
    return arrays
@array_function_dispatch(_stack_arrays_dispatcher)
def stack_arrays(arrays, defaults=None, usemask=True, asrecarray=False,
                 autoconvert=False):
    if isinstance(arrays, ndarray):
        return arrays
    elif len(arrays) == 1:
        return arrays[0]
    seqarrays = [np.asanyarray(a).ravel() for a in arrays]
    nrecords = [len(a) for a in seqarrays]
    ndtype = [a.dtype for a in seqarrays]
    fldnames = [d.names for d in ndtype]
    dtype_l = ndtype[0]
    newdescr = _get_fieldspec(dtype_l)
    names = [n for n, d in newdescr]
    for dtype_n in ndtype[1:]:
        for fname, fdtype in _get_fieldspec(dtype_n):
            if fname not in names:
                newdescr.append((fname, fdtype))
                names.append(fname)
            else:
                nameidx = names.index(fname)
                _, cdtype = newdescr[nameidx]
                if autoconvert:
                    newdescr[nameidx] = (fname, max(fdtype, cdtype))
                elif fdtype != cdtype:
                    raise TypeError("Incompatible type '%s' <> '%s'" %
                                    (cdtype, fdtype))
    if len(newdescr) == 1:
        output = ma.concatenate(seqarrays)
    else:
        output = ma.masked_all((np.sum(nrecords),), newdescr)
        offset = np.cumsum(np.r_[0, nrecords])
        seen = []
        for (a, n, i, j) in zip(seqarrays, fldnames, offset[:-1], offset[1:]):
            names = a.dtype.names
            if names is None:
                output['f%i' % len(seen)][i:j] = a
            else:
                for name in n:
                    output[name][i:j] = a[name]
                    if name not in seen:
                        seen.append(name)
    return _fix_output(_fix_defaults(output, defaults),
                       usemask=usemask, asrecarray=asrecarray)
def _find_duplicates_dispatcher(
        a, key=None, ignoremask=None, return_index=None):
    return (a,)
@array_function_dispatch(_find_duplicates_dispatcher)
def find_duplicates(a, key=None, ignoremask=True, return_index=False):
    a = np.asanyarray(a).ravel()
    fields = get_fieldstructure(a.dtype)
    base = a
    if key:
        for f in fields[key]:
            base = base[f]
        base = base[key]
    sortidx = base.argsort()
    sortedbase = base[sortidx]
    sorteddata = sortedbase.filled()
    flag = (sorteddata[:-1] == sorteddata[1:])
    if ignoremask:
        sortedmask = sortedbase.recordmask
        flag[sortedmask[1:]] = False
    flag = np.concatenate(([False], flag))
    flag[:-1] = flag[:-1] + flag[1:]
    duplicates = a[sortidx][flag]
    if return_index:
        return (duplicates, sortidx[flag])
    else:
        return duplicates
def _join_by_dispatcher(
        key, r1, r2, jointype=None, r1postfix=None, r2postfix=None,
        defaults=None, usemask=None, asrecarray=None):
    return (r1, r2)
@array_function_dispatch(_join_by_dispatcher)
def join_by(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2',
            defaults=None, usemask=True, asrecarray=False):
    if jointype not in ('inner', 'outer', 'leftouter'):
        raise ValueError(
                "The 'jointype' argument should be in 'inner', "
                "'outer' or 'leftouter' (got '%s' instead)" % jointype
                )
    if isinstance(key, str):
        key = (key,)
    if len(set(key)) != len(key):
        dup = next(x for n,x in enumerate(key) if x in key[n+1:])
        raise ValueError("duplicate join key %r" % dup)
    for name in key:
        if name not in r1.dtype.names:
            raise ValueError('r1 does not have key field %r' % name)
        if name not in r2.dtype.names:
            raise ValueError('r2 does not have key field %r' % name)
    r1 = r1.ravel()
    r2 = r2.ravel()
    nb1 = len(r1)
    (r1names, r2names) = (r1.dtype.names, r2.dtype.names)
    collisions = (set(r1names) & set(r2names)) - set(key)
    if collisions and not (r1postfix or r2postfix):
        msg = "r1 and r2 contain common names, r1postfix and r2postfix "
        msg += "can't both be empty"
        raise ValueError(msg)
    key1 = [ n for n in r1names if n in key ]
    r1k = _keep_fields(r1, key1)
    r2k = _keep_fields(r2, key1)
    aux = ma.concatenate((r1k, r2k))
    idx_sort = aux.argsort(order=key)
    aux = aux[idx_sort]
    flag_in = ma.concatenate(([False], aux[1:] == aux[:-1]))
    flag_in[:-1] = flag_in[1:] + flag_in[:-1]
    idx_in = idx_sort[flag_in]
    idx_1 = idx_in[(idx_in < nb1)]
    idx_2 = idx_in[(idx_in >= nb1)] - nb1
    (r1cmn, r2cmn) = (len(idx_1), len(idx_2))
    if jointype == 'inner':
        (r1spc, r2spc) = (0, 0)
    elif jointype == 'outer':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[(idx_out < nb1)]))
        idx_2 = np.concatenate((idx_2, idx_out[(idx_out >= nb1)] - nb1))
        (r1spc, r2spc) = (len(idx_1) - r1cmn, len(idx_2) - r2cmn)
    elif jointype == 'leftouter':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[(idx_out < nb1)]))
        (r1spc, r2spc) = (len(idx_1) - r1cmn, 0)
    (s1, s2) = (r1[idx_1], r2[idx_2])
    ndtype = _get_fieldspec(r1k.dtype)
    for fname, fdtype in _get_fieldspec(r1.dtype):
        if fname not in key:
            ndtype.append((fname, fdtype))
    for fname, fdtype in _get_fieldspec(r2.dtype):
        names = list(name for name, dtype in ndtype)
        try:
            nameidx = names.index(fname)
        except ValueError:
            ndtype.append((fname, fdtype))
        else:
            _, cdtype = ndtype[nameidx]
            if fname in key:
                ndtype[nameidx] = (fname, max(fdtype, cdtype))
            else:
                ndtype[nameidx:nameidx + 1] = [
                    (fname + r1postfix, cdtype),
                    (fname + r2postfix, fdtype)
                ]
    ndtype = np.dtype(ndtype)
    cmn = max(r1cmn, r2cmn)
    output = ma.masked_all((cmn + r1spc + r2spc,), dtype=ndtype)
    names = output.dtype.names
    for f in r1names:
        selected = s1[f]
        if f not in names or (f in r2names and not r2postfix and f not in key):
            f += r1postfix
        current = output[f]
        current[:r1cmn] = selected[:r1cmn]
        if jointype in ('outer', 'leftouter'):
            current[cmn:cmn + r1spc] = selected[r1cmn:]
    for f in r2names:
        selected = s2[f]
        if f not in names or (f in r1names and not r1postfix and f not in key):
            f += r2postfix
        current = output[f]
        current[:r2cmn] = selected[:r2cmn]
        if (jointype == 'outer') and r2spc:
            current[-r2spc:] = selected[r2cmn:]
    output.sort(order=key)
    kwargs = dict(usemask=usemask, asrecarray=asrecarray)
    return _fix_output(_fix_defaults(output, defaults), **kwargs)
def _rec_join_dispatcher(
        key, r1, r2, jointype=None, r1postfix=None, r2postfix=None,
        defaults=None):
    return (r1, r2)
@array_function_dispatch(_rec_join_dispatcher)
def rec_join(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2',
             defaults=None):
    kwargs = dict(jointype=jointype, r1postfix=r1postfix, r2postfix=r2postfix,
                  defaults=defaults, usemask=False, asrecarray=True)
    return join_by(key, r1, r2, **kwargs)
