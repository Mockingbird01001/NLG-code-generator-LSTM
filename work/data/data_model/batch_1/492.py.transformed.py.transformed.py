
__all__ = ['finfo', 'iinfo']
import warnings
from .machar import MachAr
from .overrides import set_module
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf
from .umath import log10, exp2
from . import umath
def _fr0(a):
    if a.ndim == 0:
        a = a.copy()
        a.shape = (1,)
    return a
def _fr1(a):
    if a.size == 1:
        a = a.copy()
        a.shape = ()
    return a
class MachArLike:
    def __init__(self,
                 ftype,
                 *, eps, epsneg, huge, tiny, ibeta, **kwargs):
        params = _MACHAR_PARAMS[ftype]
        float_conv = lambda v: array([v], ftype)
        float_to_float = lambda v : _fr1(float_conv(v))
        float_to_str = lambda v: (params['fmt'] % array(_fr0(v)[0], ftype))
        self.title = params['title']
        self.epsilon = self.eps = float_to_float(eps)
        self.epsneg = float_to_float(epsneg)
        self.xmax = self.huge = float_to_float(huge)
        self.xmin = self.tiny = float_to_float(tiny)
        self.ibeta = params['itype'](ibeta)
        self.__dict__.update(kwargs)
        self.precision = int(-log10(self.eps))
        self.resolution = float_to_float(float_conv(10) ** (-self.precision))
        self._str_eps = float_to_str(self.eps)
        self._str_epsneg = float_to_str(self.epsneg)
        self._str_xmin = float_to_str(self.xmin)
        self._str_xmax = float_to_str(self.xmax)
        self._str_resolution = float_to_str(self.resolution)
_convert_to_float = {
    ntypes.csingle: ntypes.single,
    ntypes.complex_: ntypes.float_,
    ntypes.clongfloat: ntypes.longfloat
    }
_title_fmt = 'numpy {} precision floating point number'
_MACHAR_PARAMS = {
    ntypes.double: dict(
        itype = ntypes.int64,
        fmt = '%24.16e',
        title = _title_fmt.format('double')),
    ntypes.single: dict(
        itype = ntypes.int32,
        fmt = '%15.7e',
        title = _title_fmt.format('single')),
    ntypes.longdouble: dict(
        itype = ntypes.longlong,
        fmt = '%s',
        title = _title_fmt.format('long double')),
    ntypes.half: dict(
        itype = ntypes.int16,
        fmt = '%12.5e',
        title = _title_fmt.format('half'))}
_KNOWN_TYPES = {}
def _register_type(machar, bytepat):
    _KNOWN_TYPES[bytepat] = machar
_float_ma = {}
def _register_known_types():
    f16 = ntypes.float16
    float16_ma = MachArLike(f16,
                            machep=-10,
                            negep=-11,
                            minexp=-14,
                            maxexp=16,
                            it=10,
                            iexp=5,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(f16(-10)),
                            epsneg=exp2(f16(-11)),
                            huge=f16(65504),
                            tiny=f16(2 ** -14))
    _register_type(float16_ma, b'f\xae')
    _float_ma[16] = float16_ma
    f32 = ntypes.float32
    float32_ma = MachArLike(f32,
                            machep=-23,
                            negep=-24,
                            minexp=-126,
                            maxexp=128,
                            it=23,
                            iexp=8,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(f32(-23)),
                            epsneg=exp2(f32(-24)),
                            huge=f32((1 - 2 ** -24) * 2**128),
                            tiny=exp2(f32(-126)))
    _register_type(float32_ma, b'\xcd\xcc\xcc\xbd')
    _float_ma[32] = float32_ma
    f64 = ntypes.float64
    epsneg_f64 = 2.0 ** -53.0
    tiny_f64 = 2.0 ** -1022.0
    float64_ma = MachArLike(f64,
                            machep=-52,
                            negep=-53,
                            minexp=-1022,
                            maxexp=1024,
                            it=52,
                            iexp=11,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=2.0 ** -52.0,
                            epsneg=epsneg_f64,
                            huge=(1.0 - epsneg_f64) / tiny_f64 * f64(4),
                            tiny=tiny_f64)
    _register_type(float64_ma, b'\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    _float_ma[64] = float64_ma
    ld = ntypes.longdouble
    epsneg_f128 = exp2(ld(-113))
    tiny_f128 = exp2(ld(-16382))
    with numeric.errstate(all='ignore'):
        huge_f128 = (ld(1) - epsneg_f128) / tiny_f128 * ld(4)
    float128_ma = MachArLike(ld,
                             machep=-112,
                             negep=-113,
                             minexp=-16382,
                             maxexp=16384,
                             it=112,
                             iexp=15,
                             ibeta=2,
                             irnd=5,
                             ngrd=0,
                             eps=exp2(ld(-112)),
                             epsneg=epsneg_f128,
                             huge=huge_f128,
                             tiny=tiny_f128)
    _register_type(float128_ma,
        b'\x9a\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\xfb\xbf')
    _register_type(float128_ma,
        b'\x9a\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\xfb\xbf')
    _float_ma[128] = float128_ma
    epsneg_f80 = exp2(ld(-64))
    tiny_f80 = exp2(ld(-16382))
    with numeric.errstate(all='ignore'):
        huge_f80 = (ld(1) - epsneg_f80) / tiny_f80 * ld(4)
    float80_ma = MachArLike(ld,
                            machep=-63,
                            negep=-64,
                            minexp=-16382,
                            maxexp=16384,
                            it=63,
                            iexp=15,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(ld(-63)),
                            epsneg=epsneg_f80,
                            huge=huge_f80,
                            tiny=tiny_f80)
    _register_type(float80_ma, b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf')
    _float_ma[80] = float80_ma
    huge_dd = (umath.nextafter(ld(inf), ld(0))
                if hasattr(umath, 'nextafter')
                else float64_ma.huge)
    float_dd_ma = MachArLike(ld,
                              machep=-105,
                              negep=-106,
                              minexp=-1022,
                              maxexp=1024,
                              it=105,
                              iexp=11,
                              ibeta=2,
                              irnd=5,
                              ngrd=0,
                              eps=exp2(ld(-105)),
                              epsneg= exp2(ld(-106)),
                              huge=huge_dd,
                              tiny=exp2(ld(-1022)))
    _register_type(float_dd_ma,
        b'\x9a\x99\x99\x99\x99\x99Y<\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    _register_type(float_dd_ma,
        b'\x9a\x99\x99\x99\x99\x99\xb9\xbf\x9a\x99\x99\x99\x99\x99Y<')
    _float_ma['dd'] = float_dd_ma
def _get_machar(ftype):
    params = _MACHAR_PARAMS.get(ftype)
    if params is None:
        raise ValueError(repr(ftype))
    key = ftype('-0.1').newbyteorder('<').tobytes()
    ma_like = None
    if ftype == ntypes.longdouble:
        ma_like = _KNOWN_TYPES.get(key[:10])
    if ma_like is None:
        ma_like = _KNOWN_TYPES.get(key)
    if ma_like is not None:
        return ma_like
    warnings.warn(
        'Signature {} for {} does not match any known type: '
        'falling back to type probe function'.format(key, ftype),
        UserWarning, stacklevel=2)
    return _discovered_machar(ftype)
def _discovered_machar(ftype):
    params = _MACHAR_PARAMS[ftype]
    return MachAr(lambda v: array([v], ftype),
                  lambda v:_fr0(v.astype(params['itype']))[0],
                  lambda v:array(_fr0(v)[0], ftype),
                  lambda v: params['fmt'] % array(_fr0(v)[0], ftype),
                  params['title'])
@set_module('numpy')
class finfo:
    _finfo_cache = {}
    def __new__(cls, dtype):
        try:
            dtype = numeric.dtype(dtype)
        except TypeError:
            dtype = numeric.dtype(type(dtype))
        obj = cls._finfo_cache.get(dtype, None)
        if obj is not None:
            return obj
        dtypes = [dtype]
        newdtype = numeric.obj2sctype(dtype)
        if newdtype is not dtype:
            dtypes.append(newdtype)
            dtype = newdtype
        if not issubclass(dtype, numeric.inexact):
            raise ValueError("data type %r not inexact" % (dtype))
        obj = cls._finfo_cache.get(dtype, None)
        if obj is not None:
            return obj
        if not issubclass(dtype, numeric.floating):
            newdtype = _convert_to_float[dtype]
            if newdtype is not dtype:
                dtypes.append(newdtype)
                dtype = newdtype
        obj = cls._finfo_cache.get(dtype, None)
        if obj is not None:
            return obj
        obj = object.__new__(cls)._init(dtype)
        for dt in dtypes:
            cls._finfo_cache[dt] = obj
        return obj
    def _init(self, dtype):
        self.dtype = numeric.dtype(dtype)
        machar = _get_machar(dtype)
        for word in ['precision', 'iexp',
                     'maxexp', 'minexp', 'negep',
                     'machep']:
            setattr(self, word, getattr(machar, word))
        for word in ['tiny', 'resolution', 'epsneg']:
            setattr(self, word, getattr(machar, word).flat[0])
        self.bits = self.dtype.itemsize * 8
        self.max = machar.huge.flat[0]
        self.min = -self.max
        self.eps = machar.eps.flat[0]
        self.nexp = machar.iexp
        self.nmant = machar.it
        self.machar = machar
        self._str_tiny = machar._str_xmin.strip()
        self._str_max = machar._str_xmax.strip()
        self._str_epsneg = machar._str_epsneg.strip()
        self._str_eps = machar._str_eps.strip()
        self._str_resolution = machar._str_resolution.strip()
        return self
    def __str__(self):
        fmt = (
            'Machine parameters for %(dtype)s\n'
            '---------------------------------------------------------------\n'
            'precision = %(precision)3s   resolution = %(_str_resolution)s\n'
            'machep = %(machep)6s   eps =        %(_str_eps)s\n'
            'negep =  %(negep)6s   epsneg =     %(_str_epsneg)s\n'
            'minexp = %(minexp)6s   tiny =       %(_str_tiny)s\n'
            'maxexp = %(maxexp)6s   max =        %(_str_max)s\n'
            'nexp =   %(nexp)6s   min =        -max\n'
            '---------------------------------------------------------------\n'
            )
        return fmt % self.__dict__
    def __repr__(self):
        c = self.__class__.__name__
        d = self.__dict__.copy()
        d['klass'] = c
        return (("%(klass)s(resolution=%(resolution)s, min=-%(_str_max)s,"
                 " max=%(_str_max)s, dtype=%(dtype)s)") % d)
@set_module('numpy')
class iinfo:
    _min_vals = {}
    _max_vals = {}
    def __init__(self, int_type):
        try:
            self.dtype = numeric.dtype(int_type)
        except TypeError:
            self.dtype = numeric.dtype(type(int_type))
        self.kind = self.dtype.kind
        self.bits = self.dtype.itemsize * 8
        self.key = "%s%d" % (self.kind, self.bits)
        if self.kind not in 'iu':
            raise ValueError("Invalid integer data type %r." % (self.kind,))
    @property
    def min(self):
        if self.kind == 'u':
            return 0
        else:
            try:
                val = iinfo._min_vals[self.key]
            except KeyError:
                val = int(-(1 << (self.bits-1)))
                iinfo._min_vals[self.key] = val
            return val
    @property
    def max(self):
        try:
            val = iinfo._max_vals[self.key]
        except KeyError:
            if self.kind == 'u':
                val = int((1 << self.bits) - 1)
            else:
                val = int((1 << (self.bits-1)) - 1)
            iinfo._max_vals[self.key] = val
        return val
    def __str__(self):
        fmt = (
            'Machine parameters for %(dtype)s\n'
            '---------------------------------------------------------------\n'
            'min = %(min)s\n'
            'max = %(max)s\n'
            '---------------------------------------------------------------\n'
            )
        return fmt % {'dtype': self.dtype, 'min': self.min, 'max': self.max}
    def __repr__(self):
        return "%s(min=%s, max=%s, dtype=%s)" % (self.__class__.__name__,
                                    self.min, self.max, self.dtype)
