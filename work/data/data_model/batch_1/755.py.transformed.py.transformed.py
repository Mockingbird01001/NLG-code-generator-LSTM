import sys
import os
import re
import functools
import itertools
import warnings
import weakref
import contextlib
from operator import itemgetter, index as opindex
from collections.abc import Mapping
import numpy as np
from . import format
from ._datasource import DataSource
from numpy.core import overrides
from numpy.core.multiarray import packbits, unpackbits
from numpy.core.overrides import set_array_function_like_doc, set_module
from numpy.core._internal import recursive
from ._iotools import (
    LineSplitter, NameValidator, StringConverter, ConverterError,
    ConverterLockError, ConversionWarning, _is_string_like,
    has_nested_fields, flatten_dtype, easy_dtype, _decode_line
    )
from numpy.compat import (
    asbytes, asstr, asunicode, os_fspath, os_PathLike,
    pickle, contextlib_nullcontext
    )
@set_module('numpy')
def loads(*args, **kwargs):
    warnings.warn(
        "np.loads is deprecated, use pickle.loads instead",
        DeprecationWarning, stacklevel=2)
    return pickle.loads(*args, **kwargs)
__all__ = [
    'savetxt', 'loadtxt', 'genfromtxt', 'ndfromtxt', 'mafromtxt',
    'recfromtxt', 'recfromcsv', 'load', 'loads', 'save', 'savez',
    'savez_compressed', 'packbits', 'unpackbits', 'fromregex', 'DataSource'
    ]
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')
class BagObj:
    def __init__(self, obj):
        self._obj = weakref.proxy(obj)
    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, '_obj')[key]
        except KeyError:
            raise AttributeError(key) from None
    def __dir__(self):
        return list(object.__getattribute__(self, '_obj').keys())
def zipfile_factory(file, *args, **kwargs):
    if not hasattr(file, 'read'):
        file = os_fspath(file)
    import zipfile
    kwargs['allowZip64'] = True
    return zipfile.ZipFile(file, *args, **kwargs)
class NpzFile(Mapping):
    zip = None
    fid = None
    def __init__(self, fid, own_fid=False, allow_pickle=False,
                 pickle_kwargs=None):
        _zip = zipfile_factory(fid)
        self._files = _zip.namelist()
        self.files = []
        self.allow_pickle = allow_pickle
        self.pickle_kwargs = pickle_kwargs
        for x in self._files:
            if x.endswith('.npy'):
                self.files.append(x[:-4])
            else:
                self.files.append(x)
        self.zip = _zip
        self.f = BagObj(self)
        if own_fid:
            self.fid = fid
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def close(self):
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None
        self.f = None
    def __del__(self):
        self.close()
    def __iter__(self):
        return iter(self.files)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, key):
        member = False
        if key in self._files:
            member = True
        elif key in self.files:
            member = True
            key += '.npy'
        if member:
            bytes = self.zip.open(key)
            magic = bytes.read(len(format.MAGIC_PREFIX))
            bytes.close()
            if magic == format.MAGIC_PREFIX:
                bytes = self.zip.open(key)
                return format.read_array(bytes,
                                         allow_pickle=self.allow_pickle,
                                         pickle_kwargs=self.pickle_kwargs)
            else:
                return self.zip.read(key)
        else:
            raise KeyError("%s is not a file in the archive" % key)
    def iteritems(self):
        warnings.warn(
            "NpzFile.iteritems is deprecated in python 3, to match the "
            "removal of dict.itertems. Use .items() instead.",
            DeprecationWarning, stacklevel=2)
        return self.items()
    def iterkeys(self):
        warnings.warn(
            "NpzFile.iterkeys is deprecated in python 3, to match the "
            "removal of dict.iterkeys. Use .keys() instead.",
            DeprecationWarning, stacklevel=2)
        return self.keys()
@set_module('numpy')
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
         encoding='ASCII'):
    if encoding not in ('ASCII', 'latin1', 'bytes'):
        raise ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")
    pickle_kwargs = dict(encoding=encoding, fix_imports=fix_imports)
    with contextlib.ExitStack() as stack:
        if hasattr(file, 'read'):
            fid = file
            own_fid = False
        else:
            fid = stack.enter_context(open(os_fspath(file), "rb"))
            own_fid = True
        _ZIP_PREFIX = b'PK\x03\x04'
        _ZIP_SUFFIX = b'PK\x05\x06'
        N = len(format.MAGIC_PREFIX)
        magic = fid.read(N)
        fid.seek(-min(N, len(magic)), 1)
        if magic.startswith(_ZIP_PREFIX) or magic.startswith(_ZIP_SUFFIX):
            stack.pop_all()
            ret = NpzFile(fid, own_fid=own_fid, allow_pickle=allow_pickle,
                          pickle_kwargs=pickle_kwargs)
            return ret
        elif magic == format.MAGIC_PREFIX:
            if mmap_mode:
                return format.open_memmap(file, mode=mmap_mode)
            else:
                return format.read_array(fid, allow_pickle=allow_pickle,
                                         pickle_kwargs=pickle_kwargs)
        else:
            if not allow_pickle:
                raise ValueError("Cannot load file containing pickled data "
                                 "when allow_pickle=False")
            try:
                return pickle.load(fid, **pickle_kwargs)
            except Exception as e:
                raise IOError(
                    "Failed to interpret file %s as a pickle" % repr(file)) from e
def _save_dispatcher(file, arr, allow_pickle=None, fix_imports=None):
    return (arr,)
@array_function_dispatch(_save_dispatcher)
def save(file, arr, allow_pickle=True, fix_imports=True):
    if hasattr(file, 'write'):
        file_ctx = contextlib_nullcontext(file)
    else:
        file = os_fspath(file)
        if not file.endswith('.npy'):
            file = file + '.npy'
        file_ctx = open(file, "wb")
    with file_ctx as fid:
        arr = np.asanyarray(arr)
        format.write_array(fid, arr, allow_pickle=allow_pickle,
                           pickle_kwargs=dict(fix_imports=fix_imports))
def _savez_dispatcher(file, *args, **kwds):
    yield from args
    yield from kwds.values()
@array_function_dispatch(_savez_dispatcher)
def savez(file, *args, **kwds):
    _savez(file, args, kwds, False)
def _savez_compressed_dispatcher(file, *args, **kwds):
    yield from args
    yield from kwds.values()
@array_function_dispatch(_savez_compressed_dispatcher)
def savez_compressed(file, *args, **kwds):
    _savez(file, args, kwds, True)
def _savez(file, args, kwds, compress, allow_pickle=True, pickle_kwargs=None):
    import zipfile
    if not hasattr(file, 'write'):
        file = os_fspath(file)
        if not file.endswith('.npz'):
            file = file + '.npz'
    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError(
                "Cannot use un-named variables and keyword %s" % key)
        namedict[key] = val
    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED
    zipf = zipfile_factory(file, mode="w", compression=compression)
    for key, val in namedict.items():
        fname = key + '.npy'
        val = np.asanyarray(val)
        with zipf.open(fname, 'w', force_zip64=True) as fid:
            format.write_array(fid, val,
                               allow_pickle=allow_pickle,
                               pickle_kwargs=pickle_kwargs)
    zipf.close()
def _getconv(dtype):
    def floatconv(x):
        x.lower()
        if '0x' in x:
            return float.fromhex(x)
        return float(x)
    typ = dtype.type
    if issubclass(typ, np.bool_):
        return lambda x: bool(int(x))
    if issubclass(typ, np.uint64):
        return np.uint64
    if issubclass(typ, np.int64):
        return np.int64
    if issubclass(typ, np.integer):
        return lambda x: int(float(x))
    elif issubclass(typ, np.longdouble):
        return np.longdouble
    elif issubclass(typ, np.floating):
        return floatconv
    elif issubclass(typ, complex):
        return lambda x: complex(asstr(x).replace('+-', '-'))
    elif issubclass(typ, np.bytes_):
        return asbytes
    elif issubclass(typ, np.unicode_):
        return asunicode
    else:
        return asstr
_loadtxt_chunksize = 50000
def _loadtxt_dispatcher(fname, dtype=None, comments=None, delimiter=None,
                        converters=None, skiprows=None, usecols=None, unpack=None,
                        ndmin=None, encoding=None, max_rows=None, *, like=None):
    return (like,)
@set_array_function_like_doc
@set_module('numpy')
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding='bytes', max_rows=None, *, like=None):
    if like is not None:
        return _loadtxt_with_like(
            fname, dtype=dtype, comments=comments, delimiter=delimiter,
            converters=converters, skiprows=skiprows, usecols=usecols,
            unpack=unpack, ndmin=ndmin, encoding=encoding,
            max_rows=max_rows, like=like
        )
    @recursive
    def flatten_dtype_internal(self, dt):
        if dt.names is None:
            shape = dt.shape
            if len(shape) == 0:
                return ([dt.base], None)
            else:
                packing = [(shape[-1], list)]
                if len(shape) > 1:
                    for dim in dt.shape[-2::-1]:
                        packing = [(dim*packing[0][0], packing*dim)]
                return ([dt.base] * int(np.prod(dt.shape)), packing)
        else:
            types = []
            packing = []
            for field in dt.names:
                tp, bytes = dt.fields[field]
                flat_dt, flat_packing = self(tp)
                types.extend(flat_dt)
                if tp.ndim > 0:
                    packing.extend(flat_packing)
                else:
                    packing.append((len(flat_dt), flat_packing))
            return (types, packing)
    @recursive
    def pack_items(self, items, packing):
        if packing is None:
            return items[0]
        elif packing is tuple:
            return tuple(items)
        elif packing is list:
            return list(items)
        else:
            start = 0
            ret = []
            for length, subpacking in packing:
                ret.append(self(items[start:start+length], subpacking))
                start += length
            return tuple(ret)
    def split_line(line):
        line = _decode_line(line, encoding=encoding)
        if comments is not None:
            line = regex_comments.split(line, maxsplit=1)[0]
        line = line.strip('\r\n')
        return line.split(delimiter) if line else []
    def read_data(chunk_size):
        X = []
        line_iter = itertools.chain([first_line], fh)
        line_iter = itertools.islice(line_iter, max_rows)
        for i, line in enumerate(line_iter):
            vals = split_line(line)
            if len(vals) == 0:
                continue
            if usecols:
                vals = [vals[j] for j in usecols]
            if len(vals) != N:
                line_num = i + skiprows + 1
                raise ValueError("Wrong number of columns at line %d"
                                 % line_num)
            items = [conv(val) for (conv, val) in zip(converters, vals)]
            items = pack_items(items, packing)
            X.append(items)
            if len(X) > chunk_size:
                yield X
                X = []
        if X:
            yield X
    if ndmin not in [0, 1, 2]:
        raise ValueError('Illegal value of ndmin keyword: %s' % ndmin)
    if comments is not None:
        if isinstance(comments, (str, bytes)):
            comments = [comments]
        comments = [_decode_line(x) for x in comments]
        comments = (re.escape(comment) for comment in comments)
        regex_comments = re.compile('|'.join(comments))
    if delimiter is not None:
        delimiter = _decode_line(delimiter)
    user_converters = converters
    byte_converters = False
    if encoding == 'bytes':
        encoding = None
        byte_converters = True
    if usecols is not None:
        try:
            usecols_as_list = list(usecols)
        except TypeError:
            usecols_as_list = [usecols]
        for col_idx in usecols_as_list:
            try:
                opindex(col_idx)
            except TypeError as e:
                e.args = (
                    "usecols must be an int or a sequence of ints but "
                    "it contains at least one element of type %s" %
                    type(col_idx),
                    )
                raise
        usecols = usecols_as_list
    dtype = np.dtype(dtype)
    defconv = _getconv(dtype)
    dtype_types, packing = flatten_dtype_internal(dtype)
    fown = False
    try:
        if isinstance(fname, os_PathLike):
            fname = os_fspath(fname)
        if _is_string_like(fname):
            fh = np.lib._datasource.open(fname, 'rt', encoding=encoding)
            fencoding = getattr(fh, 'encoding', 'latin1')
            fh = iter(fh)
            fown = True
        else:
            fh = iter(fname)
            fencoding = getattr(fname, 'encoding', 'latin1')
    except TypeError as e:
        raise ValueError(
            'fname must be a string, file handle, or generator'
        ) from e
    if encoding is not None:
        fencoding = encoding
    elif fencoding is None:
        import locale
        fencoding = locale.getpreferredencoding()
    try:
        for i in range(skiprows):
            next(fh)
        first_vals = None
        try:
            while not first_vals:
                first_line = next(fh)
                first_vals = split_line(first_line)
        except StopIteration:
            first_line = ''
            first_vals = []
            warnings.warn('loadtxt: Empty input file: "%s"' % fname,
                          stacklevel=2)
        N = len(usecols or first_vals)
        if len(dtype_types) > 1:
            converters = [_getconv(dt) for dt in dtype_types]
        else:
            converters = [defconv for i in range(N)]
            if N > 1:
                packing = [(N, tuple)]
        for i, conv in (user_converters or {}).items():
            if usecols:
                try:
                    i = usecols.index(i)
                except ValueError:
                    continue
            if byte_converters:
                def tobytes_first(x, conv):
                    if type(x) is bytes:
                        return conv(x)
                    return conv(x.encode("latin1"))
                converters[i] = functools.partial(tobytes_first, conv=conv)
            else:
                converters[i] = conv
        converters = [conv if conv is not bytes else
                      lambda x: x.encode(fencoding) for conv in converters]
        X = None
        for x in read_data(_loadtxt_chunksize):
            if X is None:
                X = np.array(x, dtype)
            else:
                nshape = list(X.shape)
                pos = nshape[0]
                nshape[0] += len(x)
                X.resize(nshape, refcheck=False)
                X[pos:, ...] = x
    finally:
        if fown:
            fh.close()
    if X is None:
        X = np.array([], dtype)
    if X.ndim == 3 and X.shape[:2] == (1, 1):
        X.shape = (1, -1)
    if X.ndim > ndmin:
        X = np.squeeze(X)
    if X.ndim < ndmin:
        if ndmin == 1:
            X = np.atleast_1d(X)
        elif ndmin == 2:
            X = np.atleast_2d(X).T
    if unpack:
        if len(dtype_types) > 1:
            return [X[field] for field in dtype.names]
        else:
            return X.T
    else:
        return X
_loadtxt_with_like = array_function_dispatch(
    _loadtxt_dispatcher
)(loadtxt)
def _savetxt_dispatcher(fname, X, fmt=None, delimiter=None, newline=None,
                        header=None, footer=None, comments=None,
                        encoding=None):
    return (X,)
@array_function_dispatch(_savetxt_dispatcher)
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
    if isinstance(fmt, bytes):
        fmt = asstr(fmt)
    delimiter = asstr(delimiter)
    class WriteWrap:
        def __init__(self, fh, encoding):
            self.fh = fh
            self.encoding = encoding
            self.do_write = self.first_write
        def close(self):
            self.fh.close()
        def write(self, v):
            self.do_write(v)
        def write_bytes(self, v):
            if isinstance(v, bytes):
                self.fh.write(v)
            else:
                self.fh.write(v.encode(self.encoding))
        def write_normal(self, v):
            self.fh.write(asunicode(v))
        def first_write(self, v):
            try:
                self.write_normal(v)
                self.write = self.write_normal
            except TypeError:
                self.write_bytes(v)
                self.write = self.write_bytes
    own_fh = False
    if isinstance(fname, os_PathLike):
        fname = os_fspath(fname)
    if _is_string_like(fname):
        open(fname, 'wt').close()
        fh = np.lib._datasource.open(fname, 'wt', encoding=encoding)
        own_fh = True
    elif hasattr(fname, 'write'):
        fh = WriteWrap(fname, encoding or 'latin1')
    else:
        raise ValueError('fname must be a string or file handle')
    try:
        X = np.asarray(X)
        if X.ndim == 0 or X.ndim > 2:
            raise ValueError(
                "Expected 1D or 2D array, got %dD array instead" % X.ndim)
        elif X.ndim == 1:
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1
            else:
                ncol = len(X.dtype.names)
        else:
            ncol = X.shape[1]
        iscomplex_X = np.iscomplexobj(X)
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = asstr(delimiter).join(map(asstr, fmt))
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError('fmt has wrong number of %% formats:  %s' % fmt)
            if n_fmt_chars == 1:
                if iscomplex_X:
                    fmt = [' (%s+%sj)' % (fmt, fmt), ] * ncol
                else:
                    fmt = [fmt, ] * ncol
                format = delimiter.join(fmt)
            elif iscomplex_X and n_fmt_chars != (2 * ncol):
                raise error
            elif ((not iscomplex_X) and n_fmt_chars != ncol):
                raise error
            else:
                format = fmt
        else:
            raise ValueError('invalid fmt: %r' % (fmt,))
        if len(header) > 0:
            header = header.replace('\n', '\n' + comments)
            fh.write(comments + header + newline)
        if iscomplex_X:
            for row in X:
                row2 = []
                for number in row:
                    row2.append(number.real)
                    row2.append(number.imag)
                s = format % tuple(row2) + newline
                fh.write(s.replace('+-', '-'))
        else:
            for row in X:
                try:
                    v = format % tuple(row) + newline
                except TypeError as e:
                    raise TypeError("Mismatch between array dtype ('%s') and "
                                    "format specifier ('%s')"
                                    % (str(X.dtype), format)) from e
                fh.write(v)
        if len(footer) > 0:
            footer = footer.replace('\n', '\n' + comments)
            fh.write(comments + footer + newline)
    finally:
        if own_fh:
            fh.close()
@set_module('numpy')
def fromregex(file, regexp, dtype, encoding=None):
    own_fh = False
    if not hasattr(file, "read"):
        file = np.lib._datasource.open(file, 'rt', encoding=encoding)
        own_fh = True
    try:
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        content = file.read()
        if isinstance(content, bytes) and isinstance(regexp, np.compat.unicode):
            regexp = asbytes(regexp)
        elif isinstance(content, np.compat.unicode) and isinstance(regexp, bytes):
            regexp = asstr(regexp)
        if not hasattr(regexp, 'match'):
            regexp = re.compile(regexp)
        seq = regexp.findall(content)
        if seq and not isinstance(seq[0], tuple):
            newdtype = np.dtype(dtype[dtype.names[0]])
            output = np.array(seq, dtype=newdtype)
            output.dtype = dtype
        else:
            output = np.array(seq, dtype=dtype)
        return output
    finally:
        if own_fh:
            file.close()
def _genfromtxt_dispatcher(fname, dtype=None, comments=None, delimiter=None,
                           skip_header=None, skip_footer=None, converters=None,
                           missing_values=None, filling_values=None, usecols=None,
                           names=None, excludelist=None, deletechars=None,
                           replace_space=None, autostrip=None, case_sensitive=None,
                           defaultfmt=None, unpack=None, usemask=None, loose=None,
                           invalid_raise=None, max_rows=None, encoding=None, *,
                           like=None):
    return (like,)
@set_array_function_like_doc
@set_module('numpy')
               skip_header=0, skip_footer=0, converters=None,
               missing_values=None, filling_values=None, usecols=None,
               names=None, excludelist=None,
               deletechars=''.join(sorted(NameValidator.defaultdeletechars)),
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt="f%i", unpack=None, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding='bytes', *,
               like=None):
    if like is not None:
        return _genfromtxt_with_like(
            fname, dtype=dtype, comments=comments, delimiter=delimiter,
            skip_header=skip_header, skip_footer=skip_footer,
            converters=converters, missing_values=missing_values,
            filling_values=filling_values, usecols=usecols, names=names,
            excludelist=excludelist, deletechars=deletechars,
            replace_space=replace_space, autostrip=autostrip,
            case_sensitive=case_sensitive, defaultfmt=defaultfmt,
            unpack=unpack, usemask=usemask, loose=loose,
            invalid_raise=invalid_raise, max_rows=max_rows, encoding=encoding,
            like=like
        )
    if max_rows is not None:
        if skip_footer:
            raise ValueError(
                    "The keywords 'skip_footer' and 'max_rows' can not be "
                    "specified at the same time.")
        if max_rows < 1:
            raise ValueError("'max_rows' must be at least 1.")
    if usemask:
        from numpy.ma import MaskedArray, make_mask_descr
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        raise TypeError(
            "The input argument 'converter' should be a valid dictionary "
            "(got '%s' instead)" % type(user_converters))
    if encoding == 'bytes':
        encoding = None
        byte_converters = True
    else:
        byte_converters = False
    try:
        if isinstance(fname, os_PathLike):
            fname = os_fspath(fname)
        if isinstance(fname, str):
            fid = np.lib._datasource.open(fname, 'rt', encoding=encoding)
            fid_ctx = contextlib.closing(fid)
        else:
            fid = fname
            fid_ctx = contextlib_nullcontext(fid)
        fhd = iter(fid)
    except TypeError as e:
        raise TypeError(
            "fname must be a string, filehandle, list of strings, "
            "or generator. Got %s instead." % type(fname)) from e
    with fid_ctx:
        split_line = LineSplitter(delimiter=delimiter, comments=comments,
                                  autostrip=autostrip, encoding=encoding)
        validate_names = NameValidator(excludelist=excludelist,
                                       deletechars=deletechars,
                                       case_sensitive=case_sensitive,
                                       replace_space=replace_space)
        try:
            for i in range(skip_header):
                next(fhd)
            first_values = None
            while not first_values:
                first_line = _decode_line(next(fhd), encoding)
                if (names is True) and (comments is not None):
                    if comments in first_line:
                        first_line = (
                            ''.join(first_line.split(comments)[1:]))
                first_values = split_line(first_line)
        except StopIteration:
            first_line = ''
            first_values = []
            warnings.warn('genfromtxt: Empty input file: "%s"' % fname, stacklevel=2)
        if names is True:
            fval = first_values[0].strip()
            if comments is not None:
                if fval in comments:
                    del first_values[0]
        if usecols is not None:
            try:
                usecols = [_.strip() for _ in usecols.split(",")]
            except AttributeError:
                try:
                    usecols = list(usecols)
                except TypeError:
                    usecols = [usecols, ]
        nbcols = len(usecols or first_values)
        if names is True:
            names = validate_names([str(_.strip()) for _ in first_values])
            first_line = ''
        elif _is_string_like(names):
            names = validate_names([_.strip() for _ in names.split(',')])
        elif names:
            names = validate_names(names)
        if dtype is not None:
            dtype = easy_dtype(dtype, defaultfmt=defaultfmt, names=names,
                               excludelist=excludelist,
                               deletechars=deletechars,
                               case_sensitive=case_sensitive,
                               replace_space=replace_space)
        if names is not None:
            names = list(names)
        if usecols:
            for (i, current) in enumerate(usecols):
                if _is_string_like(current):
                    usecols[i] = names.index(current)
                elif current < 0:
                    usecols[i] = current + len(first_values)
            if (dtype is not None) and (len(dtype) > nbcols):
                descr = dtype.descr
                dtype = np.dtype([descr[_] for _ in usecols])
                names = list(dtype.names)
            elif (names is not None) and (len(names) > nbcols):
                names = [names[_] for _ in usecols]
        elif (names is not None) and (dtype is not None):
            names = list(dtype.names)
        user_missing_values = missing_values or ()
        if isinstance(user_missing_values, bytes):
            user_missing_values = user_missing_values.decode('latin1')
        missing_values = [list(['']) for _ in range(nbcols)]
        if isinstance(user_missing_values, dict):
            for (key, val) in user_missing_values.items():
                if _is_string_like(key):
                    try:
                        key = names.index(key)
                    except ValueError:
                        continue
                if usecols:
                    try:
                        key = usecols.index(key)
                    except ValueError:
                        pass
                if isinstance(val, (list, tuple)):
                    val = [str(_) for _ in val]
                else:
                    val = [str(val), ]
                if key is None:
                    for miss in missing_values:
                        miss.extend(val)
                else:
                    missing_values[key].extend(val)
        elif isinstance(user_missing_values, (list, tuple)):
            for (value, entry) in zip(user_missing_values, missing_values):
                value = str(value)
                if value not in entry:
                    entry.append(value)
        elif isinstance(user_missing_values, str):
            user_value = user_missing_values.split(",")
            for entry in missing_values:
                entry.extend(user_value)
        else:
            for entry in missing_values:
                entry.extend([str(user_missing_values)])
        user_filling_values = filling_values
        if user_filling_values is None:
            user_filling_values = []
        filling_values = [None] * nbcols
        if isinstance(user_filling_values, dict):
            for (key, val) in user_filling_values.items():
                if _is_string_like(key):
                    try:
                        key = names.index(key)
                    except ValueError:
                        continue
                if usecols:
                    try:
                        key = usecols.index(key)
                    except ValueError:
                        pass
                filling_values[key] = val
        elif isinstance(user_filling_values, (list, tuple)):
            n = len(user_filling_values)
            if (n <= nbcols):
                filling_values[:n] = user_filling_values
            else:
                filling_values = user_filling_values[:nbcols]
        else:
            filling_values = [user_filling_values] * nbcols
        if dtype is None:
            converters = [StringConverter(None, missing_values=miss, default=fill)
                          for (miss, fill) in zip(missing_values, filling_values)]
        else:
            dtype_flat = flatten_dtype(dtype, flatten_base=True)
            if len(dtype_flat) > 1:
                zipit = zip(dtype_flat, missing_values, filling_values)
                converters = [StringConverter(dt, locked=True,
                                              missing_values=miss, default=fill)
                              for (dt, miss, fill) in zipit]
            else:
                zipit = zip(missing_values, filling_values)
                converters = [StringConverter(dtype, locked=True,
                                              missing_values=miss, default=fill)
                              for (miss, fill) in zipit]
        uc_update = []
        for (j, conv) in user_converters.items():
            if _is_string_like(j):
                try:
                    j = names.index(j)
                    i = j
                except ValueError:
                    continue
            elif usecols:
                try:
                    i = usecols.index(j)
                except ValueError:
                    continue
            else:
                i = j
            if len(first_line):
                testing_value = first_values[j]
            else:
                testing_value = None
            if conv is bytes:
                user_conv = asbytes
            elif byte_converters:
                def tobytes_first(x, conv):
                    if type(x) is bytes:
                        return conv(x)
                    return conv(x.encode("latin1"))
                user_conv = functools.partial(tobytes_first, conv=conv)
            else:
                user_conv = conv
            converters[i].update(user_conv, locked=True,
                                 testing_value=testing_value,
                                 default=filling_values[i],
                                 missing_values=missing_values[i],)
            uc_update.append((i, user_conv))
        user_converters.update(uc_update)
        rows = []
        append_to_rows = rows.append
        if usemask:
            masks = []
            append_to_masks = masks.append
        invalid = []
        append_to_invalid = invalid.append
        for (i, line) in enumerate(itertools.chain([first_line, ], fhd)):
            values = split_line(line)
            nbvalues = len(values)
            if nbvalues == 0:
                continue
            if usecols:
                try:
                    values = [values[_] for _ in usecols]
                except IndexError:
                    append_to_invalid((i + skip_header + 1, nbvalues))
                    continue
            elif nbvalues != nbcols:
                append_to_invalid((i + skip_header + 1, nbvalues))
                continue
            append_to_rows(tuple(values))
            if usemask:
                append_to_masks(tuple([v.strip() in m
                                       for (v, m) in zip(values,
                                                         missing_values)]))
            if len(rows) == max_rows:
                break
    if dtype is None:
        for (i, converter) in enumerate(converters):
            current_column = [itemgetter(i)(_m) for _m in rows]
            try:
                converter.iterupgrade(current_column)
            except ConverterLockError:
                current_column = map(itemgetter(i), rows)
                for (j, value) in enumerate(current_column):
                    try:
                        converter.upgrade(value)
                    except (ConverterError, ValueError):
                        errmsg %= (j + 1 + skip_header, value)
                        raise ConverterError(errmsg)
    nbinvalid = len(invalid)
    if nbinvalid > 0:
        nbrows = len(rows) + nbinvalid - skip_footer
        if skip_footer > 0:
            nbinvalid_skipped = len([_ for _ in invalid
                                     if _[0] > nbrows + skip_header])
            invalid = invalid[:nbinvalid - nbinvalid_skipped]
            skip_footer -= nbinvalid_skipped
        errmsg = [template % (i, nb)
                  for (i, nb) in invalid]
        if len(errmsg):
            errmsg.insert(0, "Some errors were detected !")
            errmsg = "\n".join(errmsg)
            if invalid_raise:
                raise ValueError(errmsg)
            else:
                warnings.warn(errmsg, ConversionWarning, stacklevel=2)
    if skip_footer > 0:
        rows = rows[:-skip_footer]
        if usemask:
            masks = masks[:-skip_footer]
    if loose:
        rows = list(
            zip(*[[conv._loose_call(_r) for _r in map(itemgetter(i), rows)]
                  for (i, conv) in enumerate(converters)]))
    else:
        rows = list(
            zip(*[[conv._strict_call(_r) for _r in map(itemgetter(i), rows)]
                  for (i, conv) in enumerate(converters)]))
    data = rows
    if dtype is None:
        column_types = [conv.type for conv in converters]
        strcolidx = [i for (i, v) in enumerate(column_types)
                     if v == np.unicode_]
        if byte_converters and strcolidx:
            warnings.warn(
                "Reading unicode strings without specifying the encoding "
                "argument is deprecated. Set the encoding, use None for the "
                "system default.",
                np.VisibleDeprecationWarning, stacklevel=2)
            def encode_unicode_cols(row_tup):
                row = list(row_tup)
                for i in strcolidx:
                    row[i] = row[i].encode('latin1')
                return tuple(row)
            try:
                data = [encode_unicode_cols(r) for r in data]
            except UnicodeEncodeError:
                pass
            else:
                for i in strcolidx:
                    column_types[i] = np.bytes_
        sized_column_types = column_types[:]
        for i, col_type in enumerate(column_types):
            if np.issubdtype(col_type, np.character):
                n_chars = max(len(row[i]) for row in data)
                sized_column_types[i] = (col_type, n_chars)
        if names is None:
            base = {
                c_type
                for c, c_type in zip(converters, column_types)
                if c._checked}
            if len(base) == 1:
                uniform_type, = base
                (ddtype, mdtype) = (uniform_type, bool)
            else:
                ddtype = [(defaultfmt % i, dt)
                          for (i, dt) in enumerate(sized_column_types)]
                if usemask:
                    mdtype = [(defaultfmt % i, bool)
                              for (i, dt) in enumerate(sized_column_types)]
        else:
            ddtype = list(zip(names, sized_column_types))
            mdtype = list(zip(names, [bool] * len(sized_column_types)))
        output = np.array(data, dtype=ddtype)
        if usemask:
            outputmask = np.array(masks, dtype=mdtype)
    else:
        if names and dtype.names is not None:
            dtype.names = names
        if len(dtype_flat) > 1:
            if 'O' in (_.char for _ in dtype_flat):
                if has_nested_fields(dtype):
                    raise NotImplementedError(
                        "Nested fields involving objects are not supported...")
                else:
                    output = np.array(data, dtype=dtype)
            else:
                rows = np.array(data, dtype=[('', _) for _ in dtype_flat])
                output = rows.view(dtype)
            if usemask:
                rowmasks = np.array(
                    masks, dtype=np.dtype([('', bool) for t in dtype_flat]))
                mdtype = make_mask_descr(dtype)
                outputmask = rowmasks.view(mdtype)
        else:
            if user_converters:
                ishomogeneous = True
                descr = []
                for i, ttype in enumerate([conv.type for conv in converters]):
                    if i in user_converters:
                        ishomogeneous &= (ttype == dtype.type)
                        if np.issubdtype(ttype, np.character):
                            ttype = (ttype, max(len(row[i]) for row in data))
                        descr.append(('', ttype))
                    else:
                        descr.append(('', dtype))
                if not ishomogeneous:
                    if len(descr) > 1:
                        dtype = np.dtype(descr)
                    else:
                        dtype = np.dtype(ttype)
            output = np.array(data, dtype)
            if usemask:
                if dtype.names is not None:
                    mdtype = [(_, bool) for _ in dtype.names]
                else:
                    mdtype = bool
                outputmask = np.array(masks, dtype=mdtype)
    names = output.dtype.names
    if usemask and names:
        for (name, conv) in zip(names, converters):
            missing_values = [conv(_) for _ in conv.missing_values
                              if _ != '']
            for mval in missing_values:
                outputmask[name] |= (output[name] == mval)
    if usemask:
        output = output.view(MaskedArray)
        output._mask = outputmask
    output = np.squeeze(output)
    if unpack:
        if names is None:
            return output.T
        elif len(names) == 1:
            return output[names[0]]
        else:
            return [output[field] for field in names]
    return output
_genfromtxt_with_like = array_function_dispatch(
    _genfromtxt_dispatcher
)(genfromtxt)
def ndfromtxt(fname, **kwargs):
    kwargs['usemask'] = False
    warnings.warn(
        "np.ndfromtxt is a deprecated alias of np.genfromtxt, "
        "prefer the latter.",
        DeprecationWarning, stacklevel=2)
    return genfromtxt(fname, **kwargs)
def mafromtxt(fname, **kwargs):
    kwargs['usemask'] = True
    warnings.warn(
        "np.mafromtxt is a deprecated alias of np.genfromtxt, "
        "prefer the latter.",
        DeprecationWarning, stacklevel=2)
    return genfromtxt(fname, **kwargs)
def recfromtxt(fname, **kwargs):
    kwargs.setdefault("dtype", None)
    usemask = kwargs.get('usemask', False)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output
def recfromcsv(fname, **kwargs):
    kwargs.setdefault("case_sensitive", "lower")
    kwargs.setdefault("names", True)
    kwargs.setdefault("delimiter", ",")
    kwargs.setdefault("dtype", None)
    output = genfromtxt(fname, **kwargs)
    usemask = kwargs.get("usemask", False)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output
