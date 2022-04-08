
import sys
import warnings
import copy
import textwrap
from numpy.distutils.misc_util import mingw32
C_ABI_VERSION = 0x01000009
C_API_VERSION = 0x0000000e
class MismatchCAPIWarning(Warning):
    pass
def is_released(config):
    from distutils.version import LooseVersion
    v = config.get_version('../version.py')
    if v is None:
        raise ValueError("Could not get version")
    pv = LooseVersion(vstring=v).version
    if len(pv) > 3:
        return False
    return True
def get_api_versions(apiversion, codegen_dir):
    sys.path.insert(0, codegen_dir)
    try:
        m = __import__('genapi')
        numpy_api = __import__('numpy_api')
        curapi_hash = m.fullapi_hash(numpy_api.full_api)
        apis_hash = m.get_versions_hash()
    finally:
        del sys.path[0]
    return curapi_hash, apis_hash[apiversion]
def check_api_version(apiversion, codegen_dir):
    curapi_hash, api_hash = get_api_versions(apiversion, codegen_dir)
    if not curapi_hash == api_hash:
        msg = ("API mismatch detected, the C API version "
               "numbers have to be updated. Current C api version is %d, "
               "with checksum %s, but recorded checksum for C API version %d "
               "in core/codegen_dir/cversions.txt is %s. If functions were "
               "added in the C API, you have to update C_API_VERSION in %s."
               )
        warnings.warn(msg % (apiversion, curapi_hash, apiversion, api_hash,
                             __file__),
                      MismatchCAPIWarning, stacklevel=2)
MANDATORY_FUNCS = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs",
        "floor", "ceil", "sqrt", "log10", "log", "exp", "asin",
        "acos", "atan", "fmod", 'modf', 'frexp', 'ldexp']
OPTIONAL_STDFUNCS = ["expm1", "log1p", "acosh", "asinh", "atanh",
        "rint", "trunc", "exp2", "log2", "hypot", "atan2", "pow",
        "copysign", "nextafter", "ftello", "fseeko",
        "strtoll", "strtoull", "cbrt", "strtold_l", "fallocate",
        "backtrace", "madvise"]
OPTIONAL_HEADERS = [
                "xmmintrin.h",
                "emmintrin.h",
                "immintrin.h",
                "features.h",
                "xlocale.h",
                "dlfcn.h",
                "sys/mman.h",
]
OPTIONAL_INTRINSICS = [("__builtin_isnan", '5.'),
                       ("__builtin_isinf", '5.'),
                       ("__builtin_isfinite", '5.'),
                       ("__builtin_bswap32", '5u'),
                       ("__builtin_bswap64", '5u'),
                       ("__builtin_expect", '5, 0'),
                       ("__builtin_mul_overflow", '5, 5, (int*)5'),
                       ("_m_from_int64", '0', "emmintrin.h"),
                       ("_mm_load_ps", '(float*)0', "xmmintrin.h"),
                       ("_mm_prefetch", '(float*)0, _MM_HINT_NTA',
                        "xmmintrin.h"),
                       ("_mm_load_pd", '(double*)0', "emmintrin.h"),
                       ("__builtin_prefetch", "(float*)0, 0, 3"),
                       ("__asm__ volatile", '"vpand %xmm1, %xmm2, %xmm3"',
                        "stdio.h", "LINK_AVX"),
                       ("__asm__ volatile", '"vpand %ymm1, %ymm2, %ymm3"',
                        "stdio.h", "LINK_AVX2"),
                       ("__asm__ volatile", '"vpaddd %zmm1, %zmm2, %zmm3"',
                        "stdio.h", "LINK_AVX512F"),
                       ("__asm__ volatile", '"vfpclasspd $0x40, %zmm15, %k6\\n"\
                                             "vmovdqu8 %xmm0, %xmm1\\n"\
                                             "vpbroadcastmb2q %k0, %xmm0\\n"',
                        "stdio.h", "LINK_AVX512_SKX"),
                       ("__asm__ volatile", '"xgetbv"', "stdio.h", "XGETBV"),
                       ]
OPTIONAL_FUNCTION_ATTRIBUTES = [('__attribute__((optimize("unroll-loops")))',
                                'attribute_optimize_unroll_loops'),
                                ('__attribute__((optimize("O3")))',
                                 'attribute_optimize_opt_3'),
                                ('__attribute__((nonnull (1)))',
                                 'attribute_nonnull'),
                                ('__attribute__((target ("avx")))',
                                 'attribute_target_avx'),
                                ('__attribute__((target ("avx2")))',
                                 'attribute_target_avx2'),
                                ('__attribute__((target ("avx512f")))',
                                 'attribute_target_avx512f'),
                                ('__attribute__((target ("avx512f,avx512dq,avx512bw,avx512vl,avx512cd")))',
                                 'attribute_target_avx512_skx'),
                                ]
OPTIONAL_FUNCTION_ATTRIBUTES_WITH_INTRINSICS = [('__attribute__((target("avx2,fma")))',
                                'attribute_target_avx2_with_intrinsics',
                                '__m256 temp = _mm256_set1_ps(1.0); temp = \
                                _mm256_fmadd_ps(temp, temp, temp)',
                                'immintrin.h'),
                                ('__attribute__((target("avx512f")))',
                                'attribute_target_avx512f_with_intrinsics',
                                '__m512i temp = _mm512_castps_si512(_mm512_set1_ps(1.0))',
                                'immintrin.h'),
                                ('__attribute__((target ("avx512f,avx512dq,avx512bw,avx512vl,avx512cd")))',
                                'attribute_target_avx512_skx_with_intrinsics',
                                '__mmask8 temp = _mm512_fpclass_pd_mask(_mm512_set1_pd(1.0), 0x01);\
                                __m512i temp = _mm512_castps_si512(_mm512_set1_ps(1.0));\
                                _mm_mask_storeu_epi8(NULL, 0xFF, _mm_broadcastmb_epi64(temp))',
                                'immintrin.h'),
                                ]
OPTIONAL_VARIABLE_ATTRIBUTES = ["__thread", "__declspec(thread)"]
OPTIONAL_STDFUNCS_MAYBE = [
    "expm1", "log1p", "acosh", "atanh", "asinh", "hypot", "copysign",
    "ftello", "fseeko"
    ]
C99_FUNCS = [
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs", "floor", "ceil",
    "rint", "trunc", "sqrt", "log10", "log", "log1p", "exp", "expm1",
    "asin", "acos", "atan", "asinh", "acosh", "atanh", "hypot", "atan2",
    "pow", "fmod", "modf", 'frexp', 'ldexp', "exp2", "log2", "copysign",
    "nextafter", "cbrt"
    ]
C99_FUNCS_SINGLE = [f + 'f' for f in C99_FUNCS]
C99_FUNCS_EXTENDED = [f + 'l' for f in C99_FUNCS]
C99_COMPLEX_TYPES = [
    'complex double', 'complex float', 'complex long double'
    ]
C99_COMPLEX_FUNCS = [
    "cabs", "cacos", "cacosh", "carg", "casin", "casinh", "catan",
    "catanh", "ccos", "ccosh", "cexp", "cimag", "clog", "conj", "cpow",
    "cproj", "creal", "csin", "csinh", "csqrt", "ctan", "ctanh"
    ]
def fname2def(name):
    return "HAVE_%s" % name.upper()
def sym2def(symbol):
    define = symbol.replace(' ', '')
    return define.upper()
def type2def(symbol):
    define = symbol.replace(' ', '_')
    return define.upper()
def check_long_double_representation(cmd):
    cmd._check_compiler()
    body = LONG_DOUBLE_REPRESENTATION_SRC % {'type': 'long double'}
    if sys.platform == "win32" and not mingw32():
        try:
            cmd.compiler.compile_options.remove("/GL")
        except (AttributeError, ValueError):
            pass
    elif (sys.platform != "win32"
            and cmd.compiler.compiler_type.startswith('intel')
            and '-ipo' in cmd.compiler.cc_exe):
        newcompiler = cmd.compiler.cc_exe.replace(' -ipo', '')
        cmd.compiler.set_executables(
            compiler=newcompiler,
            compiler_so=newcompiler,
            compiler_cxx=newcompiler,
            linker_exe=newcompiler,
            linker_so=newcompiler + ' -shared'
        )
    src, obj = cmd._compile(body, None, None, 'c')
    try:
        ltype = long_double_representation(pyod(obj))
        return ltype
    except ValueError:
        body = body.replace('struct', 'volatile struct')
        body += "int main(void) { return foo.before[0]; }\n"
        src, obj = cmd._compile(body, None, None, 'c')
        cmd.temp_files.append("_configtest")
        cmd.compiler.link_executable([obj], "_configtest")
        ltype = long_double_representation(pyod("_configtest"))
        return ltype
    finally:
        cmd._clean()
LONG_DOUBLE_REPRESENTATION_SRC = r"""
/* "before" is 16 bytes to ensure there's no padding between it and "x".
 *    We're not expecting any "long double" bigger than 16 bytes or with
 *       alignment requirements stricter than 16 bytes.  */
typedef %(type)s test_type;
struct {
        char         before[16];
        test_type    x;
        char         after[8];
} foo = {
        { '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
          '\001', '\043', '\105', '\147', '\211', '\253', '\315', '\357' },
        -123456789.0,
        { '\376', '\334', '\272', '\230', '\166', '\124', '\062', '\020' }
};
"""
def pyod(filename):
    out = []
    with open(filename, 'rb') as fid:
        yo2 = [oct(o)[2:] for o in fid.read()]
    for i in range(0, len(yo2), 16):
        line = ['%07d' % int(oct(i)[2:])]
        line.extend(['%03d' % int(c) for c in yo2[i:i+16]])
        out.append(" ".join(line))
    return out
_BEFORE_SEQ = ['000', '000', '000', '000', '000', '000', '000', '000',
              '001', '043', '105', '147', '211', '253', '315', '357']
_AFTER_SEQ = ['376', '334', '272', '230', '166', '124', '062', '020']
_IEEE_DOUBLE_BE = ['301', '235', '157', '064', '124', '000', '000', '000']
_IEEE_DOUBLE_LE = _IEEE_DOUBLE_BE[::-1]
_INTEL_EXTENDED_12B = ['000', '000', '000', '000', '240', '242', '171', '353',
                       '031', '300', '000', '000']
_INTEL_EXTENDED_16B = ['000', '000', '000', '000', '240', '242', '171', '353',
                       '031', '300', '000', '000', '000', '000', '000', '000']
_MOTOROLA_EXTENDED_12B = ['300', '031', '000', '000', '353', '171',
                          '242', '240', '000', '000', '000', '000']
_IEEE_QUAD_PREC_BE = ['300', '031', '326', '363', '105', '100', '000', '000',
                      '000', '000', '000', '000', '000', '000', '000', '000']
_IEEE_QUAD_PREC_LE = _IEEE_QUAD_PREC_BE[::-1]
_IBM_DOUBLE_DOUBLE_BE = (['301', '235', '157', '064', '124', '000', '000', '000'] +
                     ['000'] * 8)
_IBM_DOUBLE_DOUBLE_LE = (['000', '000', '000', '124', '064', '157', '235', '301'] +
                     ['000'] * 8)
def long_double_representation(lines):
    read = [''] * 32
    saw = None
    for line in lines:
        for w in line.split()[1:]:
            read.pop(0)
            read.append(w)
            if read[-8:] == _AFTER_SEQ:
                saw = copy.copy(read)
                if read[:12] == _BEFORE_SEQ[4:]:
                    if read[12:-8] == _INTEL_EXTENDED_12B:
                        return 'INTEL_EXTENDED_12_BYTES_LE'
                    if read[12:-8] == _MOTOROLA_EXTENDED_12B:
                        return 'MOTOROLA_EXTENDED_12_BYTES_BE'
                elif read[:8] == _BEFORE_SEQ[8:]:
                    if read[8:-8] == _INTEL_EXTENDED_16B:
                        return 'INTEL_EXTENDED_16_BYTES_LE'
                    elif read[8:-8] == _IEEE_QUAD_PREC_BE:
                        return 'IEEE_QUAD_BE'
                    elif read[8:-8] == _IEEE_QUAD_PREC_LE:
                        return 'IEEE_QUAD_LE'
                    elif read[8:-8] == _IBM_DOUBLE_DOUBLE_LE:
                        return 'IBM_DOUBLE_DOUBLE_LE'
                    elif read[8:-8] == _IBM_DOUBLE_DOUBLE_BE:
                        return 'IBM_DOUBLE_DOUBLE_BE'
                elif read[:16] == _BEFORE_SEQ:
                    if read[16:-8] == _IEEE_DOUBLE_LE:
                        return 'IEEE_DOUBLE_LE'
                    elif read[16:-8] == _IEEE_DOUBLE_BE:
                        return 'IEEE_DOUBLE_BE'
    if saw is not None:
        raise ValueError("Unrecognized format (%s)" % saw)
    else:
        raise ValueError("Could not lock sequences (%s)" % saw)
def check_for_right_shift_internal_compiler_error(cmd):
    cmd._check_compiler()
    has_optimize = cmd.try_compile(textwrap.dedent("""\
        __attribute__((optimize("O3"))) void right_shift() {}
        """), None, None)
    if not has_optimize:
        return False
    no_err = cmd.try_compile(textwrap.dedent("""\
        typedef long the_type;  /* fails also for unsigned and long long */
        __attribute__((optimize("O3"))) void right_shift(the_type in2, the_type *ip1, int n) {
            for (int i = 0; i < n; i++) {
                if (in2 < (the_type)sizeof(the_type) * 8) {
                    ip1[i] = ip1[i] >> in2;
                }
            }
        }
        """), None, None)
    return not no_err
