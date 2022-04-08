
from numpy.core.overrides import set_module
def _unpack_tuple(tup):
    if len(tup) == 1:
        return tup[0]
    else:
        return tup
def _display_as_base(cls):
    assert issubclass(cls, Exception)
    cls.__name__ = cls.__base__.__name__
    return cls
class UFuncTypeError(TypeError):
    def __init__(self, ufunc):
        self.ufunc = ufunc
@_display_as_base
class _UFuncBinaryResolutionError(UFuncTypeError):
    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc)
        self.dtypes = tuple(dtypes)
        assert len(self.dtypes) == 2
    def __str__(self):
        return (
            "ufunc {!r} cannot use operands with types {!r} and {!r}"
        ).format(
            self.ufunc.__name__, *self.dtypes
        )
@_display_as_base
class _UFuncNoLoopError(UFuncTypeError):
    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc)
        self.dtypes = tuple(dtypes)
    def __str__(self):
        return (
            "ufunc {!r} did not contain a loop with signature matching types "
            "{!r} -> {!r}"
        ).format(
            self.ufunc.__name__,
            _unpack_tuple(self.dtypes[:self.ufunc.nin]),
            _unpack_tuple(self.dtypes[self.ufunc.nin:])
        )
@_display_as_base
class _UFuncCastingError(UFuncTypeError):
    def __init__(self, ufunc, casting, from_, to):
        super().__init__(ufunc)
        self.casting = casting
        self.from_ = from_
        self.to = to
@_display_as_base
class _UFuncInputCastingError(_UFuncCastingError):
    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.in_i = i
    def __str__(self):
        i_str = "{} ".format(self.in_i) if self.ufunc.nin != 1 else ""
        return (
            "Cannot cast ufunc {!r} input {}from {!r} to {!r} with casting "
            "rule {!r}"
        ).format(
            self.ufunc.__name__, i_str, self.from_, self.to, self.casting
        )
@_display_as_base
class _UFuncOutputCastingError(_UFuncCastingError):
    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.out_i = i
    def __str__(self):
        i_str = "{} ".format(self.out_i) if self.ufunc.nout != 1 else ""
        return (
            "Cannot cast ufunc {!r} output {}from {!r} to {!r} with casting "
            "rule {!r}"
        ).format(
            self.ufunc.__name__, i_str, self.from_, self.to, self.casting
        )
@set_module('numpy')
class TooHardError(RuntimeError):
    pass
@set_module('numpy')
class AxisError(ValueError, IndexError):
    def __init__(self, axis, ndim=None, msg_prefix=None):
        if ndim is None and msg_prefix is None:
            msg = axis
        else:
            msg = ("axis {} is out of bounds for array of dimension {}"
                   .format(axis, ndim))
            if msg_prefix is not None:
                msg = "{}: {}".format(msg_prefix, msg)
        super(AxisError, self).__init__(msg)
@_display_as_base
class _ArrayMemoryError(MemoryError):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
    @property
    def _total_size(self):
        num_bytes = self.dtype.itemsize
        for dim in self.shape:
            num_bytes *= dim
        return num_bytes
    @staticmethod
    def _size_to_string(num_bytes):
        LOG2_STEP = 10
        STEP = 1024
        units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']
        unit_i = max(num_bytes.bit_length() - 1, 1) // LOG2_STEP
        unit_val = 1 << (unit_i * LOG2_STEP)
        n_units = num_bytes / unit_val
        del unit_val
        if round(n_units) == STEP:
            unit_i += 1
            n_units /= STEP
        if unit_i >= len(units):
            new_unit_i = len(units) - 1
            n_units *= 1 << ((unit_i - new_unit_i) * LOG2_STEP)
            unit_i = new_unit_i
        unit_name = units[unit_i]
        if unit_i == 0:
            return '{:.0f} {}'.format(n_units, unit_name)
        elif round(n_units) < 1000:
        else:
    def __str__(self):
        size_str = self._size_to_string(self._total_size)
        return (
            "Unable to allocate {} for an array with shape {} and data type {}"
            .format(size_str, self.shape, self.dtype)
        )
