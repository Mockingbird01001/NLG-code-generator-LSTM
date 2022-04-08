import numpy as np
from .numeric import uint8, ndarray, dtype
from numpy.compat import (
    os_fspath, contextlib_nullcontext, is_pathlib_path
)
from numpy.core.overrides import set_module
__all__ = ['memmap']
dtypedescr = dtype
valid_filemodes = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+", "w+"]
mode_equivalents = {
    "readonly":"r",
    "copyonwrite":"c",
    "readwrite":"r+",
    "write":"w+"
    }
@set_module('numpy')
class memmap(ndarray):
    __array_priority__ = -100.0
    def __new__(subtype, filename, dtype=uint8, mode='r+', offset=0,
                shape=None, order='C'):
        import mmap
        import os.path
        try:
            mode = mode_equivalents[mode]
        except KeyError as e:
            if mode not in valid_filemodes:
                raise ValueError(
                    "mode must be one of {!r} (got {!r})"
                    .format(valid_filemodes + list(mode_equivalents.keys()), mode)
                ) from None
        if mode == 'w+' and shape is None:
            raise ValueError("shape must be given")
        if hasattr(filename, 'read'):
            f_ctx = contextlib_nullcontext(filename)
        else:
            f_ctx = open(os_fspath(filename), ('r' if mode == 'c' else mode)+'b')
        with f_ctx as fid:
            fid.seek(0, 2)
            flen = fid.tell()
            descr = dtypedescr(dtype)
            _dbytes = descr.itemsize
            if shape is None:
                bytes = flen - offset
                if bytes % _dbytes:
                    raise ValueError("Size of available data is not a "
                            "multiple of the data-type size.")
                size = bytes // _dbytes
                shape = (size,)
            else:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                size = np.intp(1)
                for k in shape:
                    size *= k
            bytes = int(offset + size*_dbytes)
            if mode in ('w+', 'r+') and flen < bytes:
                fid.seek(bytes - 1, 0)
                fid.write(b'\0')
                fid.flush()
            if mode == 'c':
                acc = mmap.ACCESS_COPY
            elif mode == 'r':
                acc = mmap.ACCESS_READ
            else:
                acc = mmap.ACCESS_WRITE
            start = offset - offset % mmap.ALLOCATIONGRANULARITY
            bytes -= start
            array_offset = offset - start
            mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
            self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
                                   offset=array_offset, order=order)
            self._mmap = mm
            self.offset = offset
            self.mode = mode
            if is_pathlib_path(filename):
                self.filename = filename.resolve()
            elif hasattr(fid, "name") and isinstance(fid.name, str):
                self.filename = os.path.abspath(fid.name)
            else:
                self.filename = None
        return self
    def __array_finalize__(self, obj):
        if hasattr(obj, '_mmap') and np.may_share_memory(self, obj):
            self._mmap = obj._mmap
            self.filename = obj.filename
            self.offset = obj.offset
            self.mode = obj.mode
        else:
            self._mmap = None
            self.filename = None
            self.offset = None
            self.mode = None
    def flush(self):
        if self.base is not None and hasattr(self.base, 'flush'):
            self.base.flush()
    def __array_wrap__(self, arr, context=None):
        arr = super(memmap, self).__array_wrap__(arr, context)
        if self is arr or type(self) is not memmap:
            return arr
        if arr.shape == ():
            return arr[()]
        return arr.view(np.ndarray)
    def __getitem__(self, index):
        res = super(memmap, self).__getitem__(index)
        if type(res) is memmap and res._mmap is None:
            return res.view(type=ndarray)
        return res
