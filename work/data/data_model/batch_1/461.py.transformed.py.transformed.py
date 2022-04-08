
import numpy
import io
import warnings
from numpy.lib.utils import safe_eval
from numpy.compat import (
    isfileobj, os_fspath, pickle
    )
__all__ = []
MAGIC_PREFIX = b'\x93NUMPY'
MAGIC_LEN = len(MAGIC_PREFIX) + 2
ARRAY_ALIGN = 64
BUFFER_SIZE = 2**18
_header_size_info = {
    (1, 0): ('<H', 'latin1'),
    (2, 0): ('<I', 'latin1'),
    (3, 0): ('<I', 'utf8'),
}
def _check_version(version):
    if version not in [(1, 0), (2, 0), (3, 0), None]:
        msg = "we only support format version (1,0), (2,0), and (3,0), not %s"
        raise ValueError(msg % (version,))
def magic(major, minor):
    if major < 0 or major > 255:
        raise ValueError("major version must be 0 <= major < 256")
    if minor < 0 or minor > 255:
        raise ValueError("minor version must be 0 <= minor < 256")
    return MAGIC_PREFIX + bytes([major, minor])
def read_magic(fp):
    magic_str = _read_bytes(fp, MAGIC_LEN, "magic string")
    if magic_str[:-2] != MAGIC_PREFIX:
        msg = "the magic string is not correct; expected %r, got %r"
        raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    major, minor = magic_str[-2:]
    return major, minor
def _has_metadata(dt):
    if dt.metadata is not None:
        return True
    elif dt.names is not None:
        return any(_has_metadata(dt[k]) for k in dt.names)
    elif dt.subdtype is not None:
        return _has_metadata(dt.base)
    else:
        return False
def dtype_to_descr(dtype):
    if _has_metadata(dtype):
        warnings.warn("metadata on a dtype may be saved or ignored, but will "
                      "raise if saved when read. Use another form of storage.",
                      UserWarning, stacklevel=2)
    if dtype.names is not None:
        return dtype.descr
    else:
        return dtype.str
def descr_to_dtype(descr):
    if isinstance(descr, str):
        return numpy.dtype(descr)
    elif isinstance(descr, tuple):
        dt = descr_to_dtype(descr[0])
        return numpy.dtype((dt, descr[1]))
    titles = []
    names = []
    formats = []
    offsets = []
    offset = 0
    for field in descr:
        if len(field) == 2:
            name, descr_str = field
            dt = descr_to_dtype(descr_str)
        else:
            name, descr_str, shape = field
            dt = numpy.dtype((descr_to_dtype(descr_str), shape))
        is_pad = (name == '' and dt.type is numpy.void and dt.names is None)
        if not is_pad:
            title, name = name if isinstance(name, tuple) else (None, name)
            titles.append(title)
            names.append(name)
            formats.append(dt)
            offsets.append(offset)
        offset += dt.itemsize
    return numpy.dtype({'names': names, 'formats': formats, 'titles': titles,
                        'offsets': offsets, 'itemsize': offset})
def header_data_from_array_1_0(array):
    d = {'shape': array.shape}
    if array.flags.c_contiguous:
        d['fortran_order'] = False
    elif array.flags.f_contiguous:
        d['fortran_order'] = True
    else:
        d['fortran_order'] = False
    d['descr'] = dtype_to_descr(array.dtype)
    return d
def _wrap_header(header, version):
    import struct
    assert version is not None
    fmt, encoding = _header_size_info[version]
    if not isinstance(header, bytes):
        header = header.encode(encoding)
    hlen = len(header) + 1
    padlen = ARRAY_ALIGN - ((MAGIC_LEN + struct.calcsize(fmt) + hlen) % ARRAY_ALIGN)
    try:
        header_prefix = magic(*version) + struct.pack(fmt, hlen + padlen)
    except struct.error:
        msg = "Header length {} too big for version={}".format(hlen, version)
        raise ValueError(msg)
    return header_prefix + header + b' '*padlen + b'\n'
def _wrap_header_guess_version(header):
    try:
        return _wrap_header(header, (1, 0))
    except ValueError:
        pass
    try:
        ret = _wrap_header(header, (2, 0))
    except UnicodeEncodeError:
        pass
    else:
        warnings.warn("Stored array in format 2.0. It can only be"
                      "read by NumPy >= 1.9", UserWarning, stacklevel=2)
        return ret
    header = _wrap_header(header, (3, 0))
    warnings.warn("Stored array in format 3.0. It can only be "
                  "read by NumPy >= 1.17", UserWarning, stacklevel=2)
    return header
def _write_array_header(fp, d, version=None):
    header = ["{"]
    for key, value in sorted(d.items()):
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)
    header = _filter_header(header)
    if version is None:
        header = _wrap_header_guess_version(header)
    else:
        header = _wrap_header(header, version)
    fp.write(header)
def write_array_header_1_0(fp, d):
    _write_array_header(fp, d, (1, 0))
def write_array_header_2_0(fp, d):
    _write_array_header(fp, d, (2, 0))
def read_array_header_1_0(fp):
    return _read_array_header(fp, version=(1, 0))
def read_array_header_2_0(fp):
    return _read_array_header(fp, version=(2, 0))
def _filter_header(s):
    import tokenize
    from io import StringIO
    tokens = []
    last_token_was_number = False
    for token in tokenize.generate_tokens(StringIO(s).readline):
        token_type = token[0]
        token_string = token[1]
        if (last_token_was_number and
                token_type == tokenize.NAME and
                token_string == "L"):
            continue
        else:
            tokens.append(token)
        last_token_was_number = (token_type == tokenize.NUMBER)
    return tokenize.untokenize(tokens)
def _read_array_header(fp, version):
    import struct
    hinfo = _header_size_info.get(version)
    if hinfo is None:
        raise ValueError("Invalid version {!r}".format(version))
    hlength_type, encoding = hinfo
    hlength_str = _read_bytes(fp, struct.calcsize(hlength_type), "array header length")
    header_length = struct.unpack(hlength_type, hlength_str)[0]
    header = _read_bytes(fp, header_length, "array header")
    header = header.decode(encoding)
    header = _filter_header(header)
    try:
        d = safe_eval(header)
    except SyntaxError as e:
        msg = "Cannot parse header: {!r}\nException: {!r}"
        raise ValueError(msg.format(header, e))
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: {!r}"
        raise ValueError(msg.format(d))
    keys = sorted(d.keys())
    if keys != ['descr', 'fortran_order', 'shape']:
        msg = "Header does not contain the correct keys: {!r}"
        raise ValueError(msg.format(keys))
    if (not isinstance(d['shape'], tuple) or
            not numpy.all([isinstance(x, int) for x in d['shape']])):
        msg = "shape is not valid: {!r}"
        raise ValueError(msg.format(d['shape']))
    if not isinstance(d['fortran_order'], bool):
        msg = "fortran_order is not a valid bool: {!r}"
        raise ValueError(msg.format(d['fortran_order']))
    try:
        dtype = descr_to_dtype(d['descr'])
    except TypeError:
        msg = "descr is not a valid dtype descriptor: {!r}"
        raise ValueError(msg.format(d['descr']))
    return d['shape'], d['fortran_order'], dtype
def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
    _check_version(version)
    _write_array_header(fp, header_data_from_array_1_0(array), version)
    if array.itemsize == 0:
        buffersize = 0
    else:
        buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)
    if array.dtype.hasobject:
        if not allow_pickle:
            raise ValueError("Object arrays cannot be saved when "
                             "allow_pickle=False")
        if pickle_kwargs is None:
            pickle_kwargs = {}
        pickle.dump(array, fp, protocol=3, **pickle_kwargs)
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        if isfileobj(fp):
            array.T.tofile(fp)
        else:
            for chunk in numpy.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='F'):
                fp.write(chunk.tobytes('C'))
    else:
        if isfileobj(fp):
            array.tofile(fp)
        else:
            for chunk in numpy.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='C'):
                fp.write(chunk.tobytes('C'))
def read_array(fp, allow_pickle=False, pickle_kwargs=None):
    version = read_magic(fp)
    _check_version(version)
    shape, fortran_order, dtype = _read_array_header(fp, version)
    if len(shape) == 0:
        count = 1
    else:
        count = numpy.multiply.reduce(shape, dtype=numpy.int64)
    if dtype.hasobject:
        if not allow_pickle:
            raise ValueError("Object arrays cannot be loaded when "
                             "allow_pickle=False")
        if pickle_kwargs is None:
            pickle_kwargs = {}
        try:
            array = pickle.load(fp, **pickle_kwargs)
        except UnicodeError as err:
            raise UnicodeError("Unpickling a python object failed: %r\n"
                               "You may need to pass the encoding= option "
                               "to numpy.load" % (err,)) from err
    else:
        if isfileobj(fp):
            array = numpy.fromfile(fp, dtype=dtype, count=count)
        else:
            array = numpy.ndarray(count, dtype=dtype)
            if dtype.itemsize > 0:
                max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)
                for i in range(0, count, max_read_count):
                    read_count = min(max_read_count, count - i)
                    read_size = int(read_count * dtype.itemsize)
                    data = _read_bytes(fp, read_size, "array data")
                    array[i:i+read_count] = numpy.frombuffer(data, dtype=dtype,
                                                             count=read_count)
        if fortran_order:
            array.shape = shape[::-1]
            array = array.transpose()
        else:
            array.shape = shape
    return array
def open_memmap(filename, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=None):
    if isfileobj(filename):
        raise ValueError("Filename must be a string or a path-like object."
                         "  Memmap cannot use existing file handles.")
    if 'w' in mode:
        _check_version(version)
        dtype = numpy.dtype(dtype)
        if dtype.hasobject:
            msg = "Array can't be memory-mapped: Python objects in dtype."
            raise ValueError(msg)
        d = dict(
            descr=dtype_to_descr(dtype),
            fortran_order=fortran_order,
            shape=shape,
        )
        with open(os_fspath(filename), mode+'b') as fp:
            _write_array_header(fp, d, version)
            offset = fp.tell()
    else:
        with open(os_fspath(filename), 'rb') as fp:
            version = read_magic(fp)
            _check_version(version)
            shape, fortran_order, dtype = _read_array_header(fp, version)
            if dtype.hasobject:
                msg = "Array can't be memory-mapped: Python objects in dtype."
                raise ValueError(msg)
            offset = fp.tell()
    if fortran_order:
        order = 'F'
    else:
        order = 'C'
    if mode == 'w+':
        mode = 'r+'
    marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order,
        mode=mode, offset=offset)
    return marray
def _read_bytes(fp, size, error_template="ran out of data"):
    data = bytes()
    while True:
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data
