
import sys, io, os, re, textwrap, pprint, inspect, atexit, subprocess
class _Config:
    conf_nocache = False
    conf_noopt = False
    conf_cache_factors = None
    conf_tmp_path = None
    conf_check_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "checks"
    )
    conf_target_groups = {}
    conf_c_prefix = 'NPY_'
    conf_c_prefix_ = 'NPY__'
    conf_cc_flags = dict(
        gcc = dict(
            native = '-march=native',
            opt = '-O3',
            werror = '-Werror'
        ),
        clang = dict(
            native = '-march=native',
            opt = "-O3",
            werror = '-Werror'
        ),
        icc = dict(
            native = '-xHost',
            opt = '-O3',
            werror = '-Werror'
        ),
        iccw = dict(
            native = '/QxHost',
            opt = '/O3',
            werror = '/Werror'
        ),
        msvc = dict(
            native = None,
            opt = '/O2',
            werror = '/WX'
        )
    )
    conf_min_features = dict(
        x86 = "SSE SSE2",
        x64 = "SSE SSE2 SSE3",
        ppc64 = '',
        ppc64le = "VSX VSX2",
        armhf = '',
        aarch64 = "NEON NEON_FP16 NEON_VFPV4 ASIMD"
    )
    conf_features = dict(
        SSE = dict(
            interest=1, headers="xmmintrin.h",
            implies="SSE2"
        ),
        SSE2   = dict(interest=2, implies="SSE", headers="emmintrin.h"),
        SSE3   = dict(interest=3, implies="SSE2", headers="pmmintrin.h"),
        SSSE3  = dict(interest=4, implies="SSE3", headers="tmmintrin.h"),
        SSE41  = dict(interest=5, implies="SSSE3", headers="smmintrin.h"),
        POPCNT = dict(interest=6, implies="SSE41", headers="popcntintrin.h"),
        SSE42  = dict(interest=7, implies="POPCNT"),
        AVX    = dict(
            interest=8, implies="SSE42", headers="immintrin.h",
            implies_detect=False
        ),
        XOP    = dict(interest=9, implies="AVX", headers="x86intrin.h"),
        FMA4   = dict(interest=10, implies="AVX", headers="x86intrin.h"),
        F16C   = dict(interest=11, implies="AVX"),
        FMA3   = dict(interest=12, implies="F16C"),
        AVX2   = dict(interest=13, implies="F16C"),
        AVX512F = dict(
            interest=20, implies="FMA3 AVX2", implies_detect=False,
            extra_checks="AVX512F_REDUCE"
        ),
        AVX512CD = dict(interest=21, implies="AVX512F"),
        AVX512_KNL = dict(
            interest=40, implies="AVX512CD", group="AVX512ER AVX512PF",
            detect="AVX512_KNL", implies_detect=False
        ),
        AVX512_KNM = dict(
            interest=41, implies="AVX512_KNL",
            group="AVX5124FMAPS AVX5124VNNIW AVX512VPOPCNTDQ",
            detect="AVX512_KNM", implies_detect=False
        ),
        AVX512_SKX = dict(
            interest=42, implies="AVX512CD", group="AVX512VL AVX512BW AVX512DQ",
            detect="AVX512_SKX", implies_detect=False,
            extra_checks="AVX512BW_MASK AVX512DQ_MASK"
        ),
        AVX512_CLX = dict(
            interest=43, implies="AVX512_SKX", group="AVX512VNNI",
            detect="AVX512_CLX"
        ),
        AVX512_CNL = dict(
            interest=44, implies="AVX512_SKX", group="AVX512IFMA AVX512VBMI",
            detect="AVX512_CNL", implies_detect=False
        ),
        AVX512_ICL = dict(
            interest=45, implies="AVX512_CLX AVX512_CNL",
            group="AVX512VBMI2 AVX512BITALG AVX512VPOPCNTDQ",
            detect="AVX512_ICL", implies_detect=False
        ),
        VSX = dict(interest=1, headers="altivec.h"),
        VSX2 = dict(interest=2, implies="VSX", implies_detect=False),
        VSX3 = dict(interest=3, implies="VSX2", implies_detect=False),
        NEON  = dict(interest=1, headers="arm_neon.h"),
        NEON_FP16 = dict(interest=2, implies="NEON"),
        NEON_VFPV4 = dict(interest=3, implies="NEON_FP16"),
        ASIMD = dict(interest=4, implies="NEON_FP16 NEON_VFPV4", implies_detect=False),
        ASIMDHP = dict(interest=5, implies="ASIMD"),
        ASIMDDP = dict(interest=6, implies="ASIMD"),
        ASIMDFHM = dict(interest=7, implies="ASIMDHP"),
    )
    def conf_features_partial(self):
        if self.cc_noopt:
            return {}
        on_x86 = self.cc_on_x86 or self.cc_on_x64
        is_unix = self.cc_is_gcc or self.cc_is_clang
        if on_x86 and is_unix: return dict(
            SSE    = dict(flags="-msse"),
            SSE2   = dict(flags="-msse2"),
            SSE3   = dict(flags="-msse3"),
            SSSE3  = dict(flags="-mssse3"),
            SSE41  = dict(flags="-msse4.1"),
            POPCNT = dict(flags="-mpopcnt"),
            SSE42  = dict(flags="-msse4.2"),
            AVX    = dict(flags="-mavx"),
            F16C   = dict(flags="-mf16c"),
            XOP    = dict(flags="-mxop"),
            FMA4   = dict(flags="-mfma4"),
            FMA3   = dict(flags="-mfma"),
            AVX2   = dict(flags="-mavx2"),
            AVX512F = dict(flags="-mavx512f"),
            AVX512CD = dict(flags="-mavx512cd"),
            AVX512_KNL = dict(flags="-mavx512er -mavx512pf"),
            AVX512_KNM = dict(
                flags="-mavx5124fmaps -mavx5124vnniw -mavx512vpopcntdq"
            ),
            AVX512_SKX = dict(flags="-mavx512vl -mavx512bw -mavx512dq"),
            AVX512_CLX = dict(flags="-mavx512vnni"),
            AVX512_CNL = dict(flags="-mavx512ifma -mavx512vbmi"),
            AVX512_ICL = dict(
                flags="-mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq"
            )
        )
        if on_x86 and self.cc_is_icc: return dict(
            SSE    = dict(flags="-msse"),
            SSE2   = dict(flags="-msse2"),
            SSE3   = dict(flags="-msse3"),
            SSSE3  = dict(flags="-mssse3"),
            SSE41  = dict(flags="-msse4.1"),
            POPCNT = {},
            SSE42  = dict(flags="-msse4.2"),
            AVX    = dict(flags="-mavx"),
            F16C   = {},
            XOP    = dict(disable="Intel Compiler doesn't support it"),
            FMA4   = dict(disable="Intel Compiler doesn't support it"),
            FMA3 = dict(
                implies="F16C AVX2", flags="-march=core-avx2"
            ),
            AVX2 = dict(implies="FMA3", flags="-march=core-avx2"),
            AVX512F = dict(
                implies="AVX2 AVX512CD", flags="-march=common-avx512"
            ),
            AVX512CD = dict(
                implies="AVX2 AVX512F", flags="-march=common-avx512"
            ),
            AVX512_KNL = dict(flags="-xKNL"),
            AVX512_KNM = dict(flags="-xKNM"),
            AVX512_SKX = dict(flags="-xSKYLAKE-AVX512"),
            AVX512_CLX = dict(flags="-xCASCADELAKE"),
            AVX512_CNL = dict(flags="-xCANNONLAKE"),
            AVX512_ICL = dict(flags="-xICELAKE-CLIENT"),
        )
        if on_x86 and self.cc_is_iccw: return dict(
            SSE    = dict(flags="/arch:SSE"),
            SSE2   = dict(flags="/arch:SSE2"),
            SSE3   = dict(flags="/arch:SSE3"),
            SSSE3  = dict(flags="/arch:SSSE3"),
            SSE41  = dict(flags="/arch:SSE4.1"),
            POPCNT = {},
            SSE42  = dict(flags="/arch:SSE4.2"),
            AVX    = dict(flags="/arch:AVX"),
            F16C   = {},
            XOP    = dict(disable="Intel Compiler doesn't support it"),
            FMA4   = dict(disable="Intel Compiler doesn't support it"),
            FMA3 = dict(
                implies="F16C AVX2", flags="/arch:CORE-AVX2"
            ),
            AVX2 = dict(
                implies="FMA3", flags="/arch:CORE-AVX2"
            ),
            AVX512F = dict(
                implies="AVX2 AVX512CD", flags="/Qx:COMMON-AVX512"
            ),
            AVX512CD = dict(
                implies="AVX2 AVX512F", flags="/Qx:COMMON-AVX512"
            ),
            AVX512_KNL = dict(flags="/Qx:KNL"),
            AVX512_KNM = dict(flags="/Qx:KNM"),
            AVX512_SKX = dict(flags="/Qx:SKYLAKE-AVX512"),
            AVX512_CLX = dict(flags="/Qx:CASCADELAKE"),
            AVX512_CNL = dict(flags="/Qx:CANNONLAKE"),
            AVX512_ICL = dict(flags="/Qx:ICELAKE-CLIENT")
        )
        if on_x86 and self.cc_is_msvc: return dict(
            SSE    = dict(flags="/arch:SSE"),
            SSE2   = dict(flags="/arch:SSE2"),
            SSE3   = {},
            SSSE3  = {},
            SSE41  = {},
            POPCNT = dict(headers="nmmintrin.h"),
            SSE42  = {},
            AVX    = dict(flags="/arch:AVX"),
            F16C   = {},
            XOP    = dict(headers="ammintrin.h"),
            FMA4   = dict(headers="ammintrin.h"),
            FMA3 = dict(
                implies="F16C AVX2", flags="/arch:AVX2"
            ),
            AVX2 = dict(
                implies="F16C FMA3", flags="/arch:AVX2"
            ),
            AVX512F = dict(
                implies="AVX2 AVX512CD AVX512_SKX", flags="/arch:AVX512"
            ),
            AVX512CD = dict(
                implies="AVX512F AVX512_SKX", flags="/arch:AVX512"
            ),
            AVX512_KNL = dict(
                disable="MSVC compiler doesn't support it"
            ),
            AVX512_KNM = dict(
                disable="MSVC compiler doesn't support it"
            ),
            AVX512_SKX = dict(flags="/arch:AVX512"),
            AVX512_CLX = {},
            AVX512_CNL = {},
            AVX512_ICL = {}
        )
        on_power = self.cc_on_ppc64le or self.cc_on_ppc64
        if on_power:
            partial = dict(
                VSX = dict(
                    implies=("VSX2" if self.cc_on_ppc64le else ""),
                    flags="-mvsx"
                ),
                VSX2 = dict(
                    flags="-mcpu=power8", implies_detect=False
                ),
                VSX3 = dict(
                    flags="-mcpu=power9 -mtune=power9", implies_detect=False
                )
            )
            if self.cc_is_clang:
                partial["VSX"]["flags"]  = "-maltivec -mvsx"
                partial["VSX2"]["flags"] = "-mpower8-vector"
                partial["VSX3"]["flags"] = "-mpower9-vector"
            return partial
        if self.cc_on_aarch64 and is_unix: return dict(
            NEON = dict(
                implies="NEON_FP16 NEON_VFPV4 ASIMD", autovec=True
            ),
            NEON_FP16 = dict(
                implies="NEON NEON_VFPV4 ASIMD", autovec=True
            ),
            NEON_VFPV4 = dict(
                implies="NEON NEON_FP16 ASIMD", autovec=True
            ),
            ASIMD = dict(
                implies="NEON NEON_FP16 NEON_VFPV4", autovec=True
            ),
            ASIMDHP = dict(
                flags="-march=armv8.2-a+fp16"
            ),
            ASIMDDP = dict(
                flags="-march=armv8.2-a+dotprod"
            ),
            ASIMDFHM = dict(
                flags="-march=armv8.2-a+fp16fml"
            ),
        )
        if self.cc_on_armhf and is_unix: return dict(
            NEON = dict(
                flags="-mfpu=neon"
            ),
            NEON_FP16 = dict(
                flags="-mfpu=neon-fp16 -mfp16-format=ieee"
            ),
            NEON_VFPV4 = dict(
                flags="-mfpu=neon-vfpv4",
            ),
            ASIMD = dict(
                flags="-mfpu=neon-fp-armv8 -march=armv8-a+simd",
            ),
            ASIMDHP = dict(
                flags="-march=armv8.2-a+fp16"
            ),
            ASIMDDP = dict(
                flags="-march=armv8.2-a+dotprod",
            ),
            ASIMDFHM = dict(
                flags="-march=armv8.2-a+fp16fml"
            )
        )
        return {}
    def __init__(self):
        if self.conf_tmp_path is None:
            import tempfile, shutil
            tmp = tempfile.mkdtemp()
            def rm_temp():
                try:
                    shutil.rmtree(tmp)
                except IOError:
                    pass
            atexit.register(rm_temp)
            self.conf_tmp_path = tmp
        if self.conf_cache_factors is None:
            self.conf_cache_factors = [
                os.path.getmtime(__file__),
                self.conf_nocache
            ]
class _Distutils:
    def __init__(self, ccompiler):
        self._ccompiler = ccompiler
    def dist_compile(self, sources, flags, **kwargs):
        assert(isinstance(sources, list))
        assert(isinstance(flags, list))
        flags = kwargs.pop("extra_postargs", []) + flags
        return self._ccompiler.compile(
            sources, extra_postargs=flags, **kwargs
        )
    def dist_test(self, source, flags):
        assert(isinstance(source, str))
        from distutils.errors import CompileError
        cc = self._ccompiler;
        bk_spawn = getattr(cc, 'spawn', None)
        if bk_spawn:
            cc_type = getattr(self._ccompiler, "compiler_type", "")
            if cc_type in ("msvc",):
                setattr(cc, 'spawn', self._dist_test_spawn_paths)
            else:
                setattr(cc, 'spawn', self._dist_test_spawn)
        test = False
        try:
            self.dist_compile(
                [source], flags, output_dir=self.conf_tmp_path
            )
            test = True
        except CompileError as e:
            self.dist_log(str(e), stderr=True)
        if bk_spawn:
            setattr(cc, 'spawn', bk_spawn)
        return test
    def dist_info(self):
        if hasattr(self, "_dist_info"):
            return self._dist_info
        cc_type = getattr(self._ccompiler, "compiler_type", '')
        if cc_type in ("intelem", "intelemw"):
            platform = "x86_64"
        elif cc_type in ("intel", "intelw", "intele"):
            platform = "x86"
        else:
            from distutils.util import get_platform
            platform = get_platform()
        cc_info = getattr(self._ccompiler, "compiler", getattr(self._ccompiler, "compiler_so", ''))
        if not cc_type or cc_type == "unix":
            if hasattr(cc_info, "__iter__"):
                compiler = cc_info[0]
            else:
                compiler = str(cc_info)
        else:
            compiler = cc_type
        if hasattr(cc_info, "__iter__") and len(cc_info) > 1:
            extra_args = ' '.join(cc_info[1:])
        else:
            extra_args  = os.environ.get("CFLAGS", "")
            extra_args += os.environ.get("CPPFLAGS", "")
        self._dist_info = (platform, compiler, extra_args)
        return self._dist_info
    @staticmethod
    def dist_error(*args):
        from distutils.errors import CompileError
        raise CompileError(_Distutils._dist_str(*args))
    @staticmethod
    def dist_fatal(*args):
        from distutils.errors import DistutilsError
        raise DistutilsError(_Distutils._dist_str(*args))
    @staticmethod
    def dist_log(*args, stderr=False):
        from numpy.distutils import log
        out = _Distutils._dist_str(*args)
        if stderr:
            log.warn(out)
        else:
            log.info(out)
    @staticmethod
    def dist_load_module(name, path):
        from numpy.compat import npy_load_module
        try:
            return npy_load_module(name, path)
        except Exception as e:
            _Distutils.dist_log(e, stderr=True)
        return None
    @staticmethod
    def _dist_str(*args):
        def to_str(arg):
            if not isinstance(arg, str) and hasattr(arg, '__iter__'):
                ret = []
                for a in arg:
                    ret.append(to_str(a))
                return '('+ ' '.join(ret) + ')'
            return str(arg)
        stack = inspect.stack()[2]
        start = "CCompilerOpt.%s[%d] : " % (stack.function, stack.lineno)
        out = ' '.join([
            to_str(a)
            for a in (*args,)
        ])
        return start + out
    def _dist_test_spawn_paths(self, cmd, display=None):
        if not hasattr(self._ccompiler, "_paths"):
            self._dist_test_spawn(cmd)
            return
        old_path = os.getenv("path")
        try:
            os.environ["path"] = self._ccompiler._paths
            self._dist_test_spawn(cmd)
        finally:
            os.environ["path"] = old_path
    _dist_warn_regex = re.compile(
        ".*("
        "warning D9002|"
        "invalid argument for option"
        ").*"
    )
    @staticmethod
    def _dist_test_spawn(cmd, display=None):
        from distutils.errors import CompileError
        try:
            o = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                        universal_newlines=True)
            if o and re.match(_Distutils._dist_warn_regex, o):
                _Distutils.dist_error(
                    "Flags in command", cmd ,"aren't supported by the compiler"
                    ", output -> \n%s" % o
                )
        except subprocess.CalledProcessError as exc:
            o = exc.output
            s = exc.returncode
        except OSError:
            o = b''
            s = 127
        else:
            return None
        _Distutils.dist_error(
            "Command", cmd, "failed with exit status %d output -> \n%s" % (
            s, o
        ))
_share_cache = {}
class _Cache:
    _cache_ignore = re.compile("^(_|conf_)")
    def __init__(self, cache_path=None, *factors):
        self.cache_me = {}
        self.cache_private = set()
        self.cache_infile = False
        if self.conf_nocache:
            self.dist_log("cache is disabled by `Config`")
            return
        chash = self.cache_hash(*factors, *self.conf_cache_factors)
        if cache_path:
            if os.path.exists(cache_path):
                self.dist_log("load cache from file ->", cache_path)
                cache_mod = self.dist_load_module("cache", cache_path)
                if not cache_mod:
                    self.dist_log(
                        "unable to load the cache file as a module",
                        stderr=True
                    )
                elif not hasattr(cache_mod, "hash") or                     not hasattr(cache_mod, "data"):
                    self.dist_log("invalid cache file", stderr=True)
                elif chash == cache_mod.hash:
                    self.dist_log("hit the file cache")
                    for attr, val in cache_mod.data.items():
                        setattr(self, attr, val)
                    self.cache_infile = True
                else:
                    self.dist_log("miss the file cache")
            atexit.register(self._cache_write, cache_path, chash)
        if not self.cache_infile:
            other_cache = _share_cache.get(chash)
            if other_cache:
                self.dist_log("hit the memory cache")
                for attr, val in other_cache.__dict__.items():
                    if attr in other_cache.cache_private or                               re.match(self._cache_ignore, attr):
                        continue
                    setattr(self, attr, val)
        _share_cache[chash] = self
    def __del__(self):
        pass
    def _cache_write(self, cache_path, cache_hash):
        self.dist_log("write cache to path ->", cache_path)
        for attr in list(self.__dict__.keys()):
            if re.match(self._cache_ignore, attr):
                self.__dict__.pop(attr)
        d = os.path.dirname(cache_path)
        if not os.path.exists(d):
            os.makedirs(d)
        repr_dict = pprint.pformat(self.__dict__, compact=True)
        with open(cache_path, "w") as f:
            f.write(textwrap.dedent("""\
            (distutils/ccompiler_opt.py)
            hash = {}
            data = \\
            """).format(cache_hash))
            f.write(repr_dict)
    def cache_hash(self, *factors):
        chash = 0
        for f in factors:
            for char in str(f):
                chash  = ord(char) + (chash << 6) + (chash << 16) - chash
                chash &= 0xFFFFFFFF
        return chash
    @staticmethod
    def me(cb):
        def cache_wrap_me(self, *args, **kwargs):
            cache_key = str((
                cb.__name__, *args, *kwargs.keys(), *kwargs.values()
            ))
            if cache_key in self.cache_me:
                return self.cache_me[cache_key]
            ccb = cb(self, *args, **kwargs)
            self.cache_me[cache_key] = ccb
            return ccb
        return cache_wrap_me
class _CCompiler(object):
    def __init__(self):
        if hasattr(self, "cc_is_cached"):
            return
        detect_arch = (
            ("cc_on_x64",      ".*(x|x86_|amd)64.*"),
            ("cc_on_x86",      ".*(win32|x86|i386|i686).*"),
            ("cc_on_ppc64le",  ".*(powerpc|ppc)64(el|le).*"),
            ("cc_on_ppc64",    ".*(powerpc|ppc)64.*"),
            ("cc_on_aarch64",  ".*(aarch64|arm64).*"),
            ("cc_on_armhf",    ".*arm.*"),
            ("cc_on_noarch",    ""),
        )
        detect_compiler = (
            ("cc_is_gcc",     r".*(gcc|gnu\-g).*"),
            ("cc_is_clang",    ".*clang.*"),
            ("cc_is_iccw",     ".*(intelw|intelemw|iccw).*"),
            ("cc_is_icc",      ".*(intel|icc).*"),
            ("cc_is_msvc",     ".*msvc.*"),
            ("cc_is_nocc",     ""),
        )
        detect_args = (
           ("cc_has_debug",  ".*(O0|Od|ggdb|coverage|debug:full).*"),
           ("cc_has_native", ".*(-march=native|-xHost|/QxHost).*"),
           ("cc_noopt", ".*DISABLE_OPT.*"),
        )
        dist_info = self.dist_info()
        platform, compiler_info, extra_args = dist_info
        for section in (detect_arch, detect_compiler, detect_args):
            for attr, rgex in section:
                setattr(self, attr, False)
        for detect, searchin in ((detect_arch, platform), (detect_compiler, compiler_info)):
            for attr, rgex in detect:
                if rgex and not re.match(rgex, searchin, re.IGNORECASE):
                    continue
                setattr(self, attr, True)
                break
        for attr, rgex in detect_args:
            if rgex and not re.match(rgex, extra_args, re.IGNORECASE):
                continue
            setattr(self, attr, True)
        if self.cc_on_noarch:
            self.dist_log(
                "unable to detect CPU architecture which lead to disable the optimization. "
                f"check dist_info:<<\n{dist_info}\n>>",
                stderr=True
            )
            self.cc_noopt = True
        if self.conf_noopt:
            self.dist_log("Optimization is disabled by the Config", stderr=True)
            self.cc_noopt = True
        if self.cc_is_nocc:
            self.dist_log(
                "unable to detect compiler type which leads to treating it as GCC. "
                "this is a normal behavior if you're using gcc-like compiler such as MinGW or IBM/XLC."
                f"check dist_info:<<\n{dist_info}\n>>",
                stderr=True
            )
            self.cc_is_gcc = True
        self.cc_march = "unknown"
        for arch in ("x86", "x64", "ppc64", "ppc64le", "armhf", "aarch64"):
            if getattr(self, "cc_on_" + arch):
                self.cc_march = arch
                break
        self.cc_name = "unknown"
        for name in ("gcc", "clang", "iccw", "icc", "msvc"):
            if getattr(self, "cc_is_" + name):
                self.cc_name = name
                break
        self.cc_flags = {}
        compiler_flags = self.conf_cc_flags.get(self.cc_name)
        if compiler_flags is None:
            self.dist_fatal(
                "undefined flag for compiler '%s', "
                "leave an empty dict instead" % self.cc_name
            )
        for name, flags in compiler_flags.items():
            self.cc_flags[name] = nflags = []
            if flags:
                assert(isinstance(flags, str))
                flags = flags.split()
                for f in flags:
                    if self.cc_test_flags([f]):
                        nflags.append(f)
        self.cc_is_cached = True
    @_Cache.me
    def cc_test_flags(self, flags):
        assert(isinstance(flags, list))
        self.dist_log("testing flags", flags)
        test_path = os.path.join(self.conf_check_path, "test_flags.c")
        test = self.dist_test(test_path, flags)
        if not test:
            self.dist_log("testing failed", stderr=True)
        return test
    def cc_normalize_flags(self, flags):
        assert(isinstance(flags, list))
        if self.cc_is_gcc or self.cc_is_clang or self.cc_is_icc:
            return self._cc_normalize_unix(flags)
        if self.cc_is_msvc or self.cc_is_iccw:
            return self._cc_normalize_win(flags)
        return flags
    _cc_normalize_unix_mrgx = re.compile(
        r"^(-mcpu=|-march=|-x[A-Z0-9\-])"
    )
    _cc_normalize_unix_frgx = re.compile(
        r"^(?!(-mcpu=|-march=|-x[A-Z0-9\-]))(?!-m[a-z0-9\-\.]*.$)"
    )
    _cc_normalize_unix_krgx = re.compile(
        r"^(-mfpu|-mtune)"
    )
    _cc_normalize_arch_ver = re.compile(
        r"[0-9.]"
    )
    def _cc_normalize_unix(self, flags):
        def ver_flags(f):
            tokens = f.split('+')
            ver = float('0' + ''.join(
                re.findall(self._cc_normalize_arch_ver, tokens[0])
            ))
            return ver, tokens[0], tokens[1:]
        if len(flags) <= 1:
            return flags
        for i, cur_flag in enumerate(reversed(flags)):
            if not re.match(self._cc_normalize_unix_mrgx, cur_flag):
                continue
            lower_flags = flags[:-(i+1)]
            upper_flags = flags[-i:]
            filterd = list(filter(
                self._cc_normalize_unix_frgx.search, lower_flags
            ))
            ver, arch, subflags = ver_flags(cur_flag)
            if ver > 0 and len(subflags) > 0:
                for xflag in lower_flags:
                    xver, _, xsubflags = ver_flags(xflag)
                    if ver == xver:
                        subflags = xsubflags + subflags
                cur_flag = arch + '+' + '+'.join(subflags)
            flags = filterd + [cur_flag]
            if i > 0:
                flags += upper_flags
            break
        final_flags = []
        matched = set()
        for f in reversed(flags):
            match = re.match(self._cc_normalize_unix_krgx, f)
            if not match:
                pass
            elif match[0] in matched:
                continue
            else:
                matched.add(match[0])
            final_flags.insert(0, f)
        return final_flags
    _cc_normalize_win_frgx = re.compile(
        r"^(?!(/arch\:|/Qx\:))"
    )
    _cc_normalize_win_mrgx = re.compile(
        r"^(/arch|/Qx:)"
    )
    def _cc_normalize_win(self, flags):
        for i, f in enumerate(reversed(flags)):
            if not re.match(self._cc_normalize_win_mrgx, f):
                continue
            i += 1
            return list(filter(
                self._cc_normalize_win_frgx.search, flags[:-i]
            )) + flags[-i:]
        return flags
class _Feature:
    def __init__(self):
        if hasattr(self, "feature_is_cached"):
            return
        self.feature_supported = pfeatures = self.conf_features_partial()
        for feature_name in list(pfeatures.keys()):
            feature  = pfeatures[feature_name]
            cfeature = self.conf_features[feature_name]
            feature.update({
                k:v for k,v in cfeature.items() if k not in feature
            })
            disabled = feature.get("disable")
            if disabled is not None:
                pfeatures.pop(feature_name)
                self.dist_log(
                    "feature '%s' is disabled," % feature_name,
                    disabled, stderr=True
                )
                continue
            for option in (
                "implies", "group", "detect", "headers", "flags", "extra_checks"
            ) :
                oval = feature.get(option)
                if isinstance(oval, str):
                    feature[option] = oval.split()
        self.feature_min = set()
        min_f = self.conf_min_features.get(self.cc_march, "")
        for F in min_f.upper().split():
            if F in self.feature_supported:
                self.feature_min.add(F)
        self.feature_is_cached = True
    def feature_names(self, names=None, force_flags=None):
        assert(
            names is None or (
                not isinstance(names, str) and
                hasattr(names, "__iter__")
            )
        )
        assert(force_flags is None or isinstance(force_flags, list))
        if names is None:
            names = self.feature_supported.keys()
        supported_names = set()
        for f in names:
            if self.feature_is_supported(f, force_flags=force_flags):
                supported_names.add(f)
        return supported_names
    def feature_is_exist(self, name):
        assert(name.isupper())
        return name in self.conf_features
    def feature_sorted(self, names, reverse=False):
        def sort_cb(k):
            if isinstance(k, str):
                return self.feature_supported[k]["interest"]
            rank = max([self.feature_supported[f]["interest"] for f in k])
            rank += len(k) -1
            return rank
        return sorted(names, reverse=reverse, key=sort_cb)
    def feature_implies(self, names, keep_origins=False):
        def get_implies(name, _caller=set()):
            implies = set()
            d = self.feature_supported[name]
            for i in d.get("implies", []):
                implies.add(i)
                if i in _caller:
                    continue
                _caller.add(name)
                implies = implies.union(get_implies(i, _caller))
            return implies
        if isinstance(names, str):
            implies = get_implies(names)
            names = [names]
        else:
            assert(hasattr(names, "__iter__"))
            implies = set()
            for n in names:
                implies = implies.union(get_implies(n))
        if not keep_origins:
            implies.difference_update(names)
        return implies
    def feature_implies_c(self, names):
        if isinstance(names, str):
            names = set((names,))
        else:
            names = set(names)
        return names.union(self.feature_implies(names))
    def feature_ahead(self, names):
        assert(
            not isinstance(names, str)
            and hasattr(names, '__iter__')
        )
        implies = self.feature_implies(names, keep_origins=True)
        ahead = [n for n in names if n not in implies]
        if len(ahead) == 0:
            ahead = self.feature_sorted(names, reverse=True)[:1]
        return ahead
    def feature_untied(self, names):
        assert(
            not isinstance(names, str)
            and hasattr(names, '__iter__')
        )
        final = []
        for n in names:
            implies = self.feature_implies(n)
            tied = [
                nn for nn in final
                if nn in implies and n in self.feature_implies(nn)
            ]
            if tied:
                tied = self.feature_sorted(tied + [n])
                if n not in tied[1:]:
                    continue
                final.remove(tied[:1][0])
            final.append(n)
        return final
    def feature_get_til(self, names, keyisfalse):
        def til(tnames):
            tnames = self.feature_implies_c(tnames)
            tnames = self.feature_sorted(tnames, reverse=True)
            for i, n in enumerate(tnames):
                if not self.feature_supported[n].get(keyisfalse, True):
                    tnames = tnames[:i+1]
                    break
            return tnames
        if isinstance(names, str) or len(names) <= 1:
            names = til(names)
            names.reverse()
            return names
        names = self.feature_ahead(names)
        names = {t for n in names for t in til(n)}
        return self.feature_sorted(names)
    def feature_detect(self, names):
        names = self.feature_get_til(names, "implies_detect")
        detect = []
        for n in names:
            d = self.feature_supported[n]
            detect += d.get("detect", d.get("group", [n]))
        return detect
    @_Cache.me
    def feature_flags(self, names):
        names = self.feature_sorted(self.feature_implies_c(names))
        flags = []
        for n in names:
            d = self.feature_supported[n]
            f = d.get("flags", [])
            if not f or not self.cc_test_flags(f):
                continue
            flags += f
        return self.cc_normalize_flags(flags)
    @_Cache.me
    def feature_test(self, name, force_flags=None):
        if force_flags is None:
            force_flags = self.feature_flags(name)
        self.dist_log(
            "testing feature '%s' with flags (%s)" % (
            name, ' '.join(force_flags)
        ))
        test_path = os.path.join(
            self.conf_check_path, "cpu_%s.c" % name.lower()
        )
        if not os.path.exists(test_path):
            self.dist_fatal("feature test file is not exist", test_path)
        test = self.dist_test(test_path, force_flags + self.cc_flags["werror"])
        if not test:
            self.dist_log("testing failed", stderr=True)
        return test
    @_Cache.me
    def feature_is_supported(self, name, force_flags=None):
        assert(name.isupper())
        assert(force_flags is None or isinstance(force_flags, list))
        supported = name in self.feature_supported
        if supported:
            for impl in self.feature_implies(name):
                if not self.feature_test(impl, force_flags):
                    return False
            if not self.feature_test(name, force_flags):
                return False
        return supported
    @_Cache.me
    def feature_can_autovec(self, name):
        assert(isinstance(name, str))
        d = self.feature_supported[name]
        can = d.get("autovec", None)
        if can is None:
            valid_flags = [
                self.cc_test_flags([f]) for f in d.get("flags", [])
            ]
            can = valid_flags and any(valid_flags)
        return can
    @_Cache.me
    def feature_extra_checks(self, name):
        assert isinstance(name, str)
        d = self.feature_supported[name]
        extra_checks = d.get("extra_checks", [])
        if not extra_checks:
            return []
        self.dist_log("Testing extra checks for feature '%s'" % name, extra_checks)
        flags = self.feature_flags(name)
        available = []
        not_available = []
        for chk in extra_checks:
            test_path = os.path.join(
                self.conf_check_path, "extra_%s.c" % chk.lower()
            )
            if not os.path.exists(test_path):
                self.dist_fatal("extra check file does not exist", test_path)
            is_supported = self.dist_test(test_path, flags + self.cc_flags["werror"])
            if is_supported:
                available.append(chk)
            else:
                not_available.append(chk)
        if not_available:
            self.dist_log("testing failed for checks", not_available, stderr=True)
        return available
    def feature_c_preprocessor(self, feature_name, tabs=0):
        assert(feature_name.isupper())
        feature = self.feature_supported.get(feature_name)
        assert(feature is not None)
        prepr = [
            "/** %s **/" % feature_name,
        ]
        prepr += [
        ]
        extra_defs = feature.get("group", [])
        extra_defs += self.feature_extra_checks(feature_name)
        for edef in extra_defs:
            prepr += [
            ]
        if tabs > 0:
            prepr = [('\t'*tabs) + l for l in prepr]
        return '\n'.join(prepr)
class _Parse:
    def __init__(self, cpu_baseline, cpu_dispatch):
        self._parse_policies = dict(
            KEEP_BASELINE = (
                None, self._parse_policy_not_keepbase,
                []
            ),
            KEEP_SORT = (
                self._parse_policy_keepsort,
                self._parse_policy_not_keepsort,
                []
            ),
            MAXOPT = (
                self._parse_policy_maxopt, None,
                []
            ),
            WERROR = (
                self._parse_policy_werror, None,
                []
            ),
            AUTOVEC = (
                self._parse_policy_autovec, None,
                ["MAXOPT"]
            )
        )
        if hasattr(self, "parse_is_cached"):
            return
        self.parse_baseline_names = []
        self.parse_baseline_flags = []
        self.parse_dispatch_names = []
        self.parse_target_groups = {}
        if self.cc_noopt:
            cpu_baseline = cpu_dispatch = None
        self.dist_log("check requested baseline")
        if cpu_baseline is not None:
            cpu_baseline = self._parse_arg_features("cpu_baseline", cpu_baseline)
            baseline_names = self.feature_names(cpu_baseline)
            self.parse_baseline_flags = self.feature_flags(baseline_names)
            self.parse_baseline_names = self.feature_sorted(
                self.feature_implies_c(baseline_names)
            )
        self.dist_log("check requested dispatch-able features")
        if cpu_dispatch is not None:
            cpu_dispatch_ = self._parse_arg_features("cpu_dispatch", cpu_dispatch)
            cpu_dispatch = {
                f for f in cpu_dispatch_
                if f not in self.parse_baseline_names
            }
            conflict_baseline = cpu_dispatch_.difference(cpu_dispatch)
            self.parse_dispatch_names = self.feature_sorted(
                self.feature_names(cpu_dispatch)
            )
            if len(conflict_baseline) > 0:
                self.dist_log(
                    "skip features", conflict_baseline, "since its part of baseline"
                )
        self.dist_log("initialize targets groups")
        for group_name, tokens in self.conf_target_groups.items():
            self.dist_log("parse target group", group_name)
            GROUP_NAME = group_name.upper()
            if not tokens or not tokens.strip():
                self.parse_target_groups[GROUP_NAME] = (
                    False, [], []
                )
                continue
            has_baseline, features, extra_flags =                self._parse_target_tokens(tokens)
            self.parse_target_groups[GROUP_NAME] = (
                has_baseline, features, extra_flags
            )
        self.parse_is_cached = True
    def parse_targets(self, source):
        self.dist_log("looking for '@targets' inside -> ", source)
        with open(source) as fd:
            tokens = ""
            max_to_reach = 1000
            start_with = "@targets"
            start_pos = -1
            end_with = "*/"
            end_pos = -1
            for current_line, line in enumerate(fd):
                if current_line == max_to_reach:
                    self.dist_fatal("reached the max of lines")
                    break
                if start_pos == -1:
                    start_pos = line.find(start_with)
                    if start_pos == -1:
                        continue
                    start_pos += len(start_with)
                tokens += line
                end_pos = line.find(end_with)
                if end_pos != -1:
                    end_pos += len(tokens) - len(line)
                    break
        if start_pos == -1:
            self.dist_fatal("expected to find '%s' within a C comment" % start_with)
        if end_pos == -1:
            self.dist_fatal("expected to end with '%s'" % end_with)
        tokens = tokens[start_pos:end_pos]
        return self._parse_target_tokens(tokens)
    _parse_regex_arg = re.compile(r'\s|[,]|([+-])')
    def _parse_arg_features(self, arg_name, req_features):
        if not isinstance(req_features, str):
            self.dist_fatal("expected a string in '%s'" % arg_name)
        final_features = set()
        tokens = list(filter(None, re.split(self._parse_regex_arg, req_features)))
        append = True
        for tok in tokens:
                self.dist_fatal(
                    arg_name, "target groups and policies "
                    "aren't allowed from arguments, "
                    "only from dispatch-able sources"
                )
            if tok == '+':
                append = True
                continue
            if tok == '-':
                append = False
                continue
            TOK = tok.upper()
            features_to = set()
            if TOK == "NONE":
                pass
            elif TOK == "NATIVE":
                native = self.cc_flags["native"]
                if not native:
                    self.dist_fatal(arg_name,
                        "native option isn't supported by the compiler"
                    )
                features_to = self.feature_names(force_flags=native)
            elif TOK == "MAX":
                features_to = self.feature_supported.keys()
            elif TOK == "MIN":
                features_to = self.feature_min
            else:
                if TOK in self.feature_supported:
                    features_to.add(TOK)
                else:
                    if not self.feature_is_exist(TOK):
                        self.dist_fatal(arg_name,
                            ", '%s' isn't a known feature or option" % tok
                        )
            if append:
                final_features = final_features.union(features_to)
            else:
                final_features = final_features.difference(features_to)
            append = True
        return final_features
    _parse_regex_target = re.compile(r'\s|[*,/]|([()])')
    def _parse_target_tokens(self, tokens):
        assert(isinstance(tokens, str))
        final_targets = []
        extra_flags = []
        has_baseline = False
        skipped  = set()
        policies = set()
        multi_target = None
        tokens = list(filter(None, re.split(self._parse_regex_target, tokens)))
        if not tokens:
            self.dist_fatal("expected one token at least")
        for tok in tokens:
            TOK = tok.upper()
            ch = tok[0]
            if ch in ('+', '-'):
                self.dist_fatal(
                    "+/- are 'not' allowed from target's groups or @targets, "
                    "only from cpu_baseline and cpu_dispatch parms"
                )
            elif ch == '$':
                if multi_target is not None:
                    self.dist_fatal(
                        "policies aren't allowed inside multi-target '()'"
                        ", only CPU features"
                    )
                policies.add(self._parse_token_policy(TOK))
                if multi_target is not None:
                    self.dist_fatal(
                        "target groups aren't allowed inside multi-target '()'"
                        ", only CPU features"
                    )
                has_baseline, final_targets, extra_flags =                self._parse_token_group(TOK, has_baseline, final_targets, extra_flags)
            elif ch == '(':
                if multi_target is not None:
                    self.dist_fatal("unclosed multi-target, missing ')'")
                multi_target = set()
            elif ch == ')':
                if multi_target is None:
                    self.dist_fatal("multi-target opener '(' wasn't found")
                targets = self._parse_multi_target(multi_target)
                if targets is None:
                    skipped.add(tuple(multi_target))
                else:
                    if len(targets) == 1:
                        targets = targets[0]
                    if targets and targets not in final_targets:
                        final_targets.append(targets)
                multi_target = None
            else:
                if TOK == "BASELINE":
                    if multi_target is not None:
                        self.dist_fatal("baseline isn't allowed inside multi-target '()'")
                    has_baseline = True
                    continue
                if multi_target is not None:
                    multi_target.add(TOK)
                    continue
                if not self.feature_is_exist(TOK):
                    self.dist_fatal("invalid target name '%s'" % TOK)
                is_enabled = (
                    TOK in self.parse_baseline_names or
                    TOK in self.parse_dispatch_names
                )
                if  is_enabled:
                    if TOK not in final_targets:
                        final_targets.append(TOK)
                    continue
                skipped.add(TOK)
        if multi_target is not None:
            self.dist_fatal("unclosed multi-target, missing ')'")
        if skipped:
            self.dist_log(
                "skip targets", skipped,
                "not part of baseline or dispatch-able features"
            )
        final_targets = self.feature_untied(final_targets)
        for p in list(policies):
            _, _, deps = self._parse_policies[p]
            for d in deps:
                if d in policies:
                    continue
                self.dist_log(
                    "policy '%s' force enables '%s'" % (
                    p, d
                ))
                policies.add(d)
        for p, (have, nhave, _) in self._parse_policies.items():
            func = None
            if p in policies:
                func = have
                self.dist_log("policy '%s' is ON" % p)
            else:
                func = nhave
            if not func:
                continue
            has_baseline, final_targets, extra_flags = func(
                has_baseline, final_targets, extra_flags
            )
        return has_baseline, final_targets, extra_flags
    def _parse_token_policy(self, token):
        if len(token) <= 1 or token[-1:] == token[0]:
            self.dist_fatal("'$' must stuck in the begin of policy name")
        token = token[1:]
        if token not in self._parse_policies:
            self.dist_fatal(
                "'%s' is an invalid policy name, available policies are" % token,
                self._parse_policies.keys()
            )
        return token
    def _parse_token_group(self, token, has_baseline, final_targets, extra_flags):
        if len(token) <= 1 or token[-1:] == token[0]:
        token = token[1:]
        ghas_baseline, gtargets, gextra_flags = self.parse_target_groups.get(
            token, (False, None, [])
        )
        if gtargets is None:
            self.dist_fatal(
                "'%s' is an invalid target group name, " % token +                "available target groups are",
                self.parse_target_groups.keys()
            )
        if ghas_baseline:
            has_baseline = True
        final_targets += [f for f in gtargets if f not in final_targets]
        extra_flags += [f for f in gextra_flags if f not in extra_flags]
        return has_baseline, final_targets, extra_flags
    def _parse_multi_target(self, targets):
        if not targets:
            self.dist_fatal("empty multi-target '()'")
        if not all([
            self.feature_is_exist(tar) for tar in targets
        ]) :
            self.dist_fatal("invalid target name in multi-target", targets)
        if not all([
            (
                tar in self.parse_baseline_names or
                tar in self.parse_dispatch_names
            )
            for tar in targets
        ]) :
            return None
        targets = self.feature_ahead(targets)
        if not targets:
            return None
        targets = self.feature_sorted(targets)
        targets = tuple(targets)
        return targets
    def _parse_policy_not_keepbase(self, has_baseline, final_targets, extra_flags):
        skipped = []
        for tar in final_targets[:]:
            is_base = False
            if isinstance(tar, str):
                is_base = tar in self.parse_baseline_names
            else:
                is_base = all([
                    f in self.parse_baseline_names
                    for f in tar
                ])
            if is_base:
                skipped.append(tar)
                final_targets.remove(tar)
        if skipped:
            self.dist_log("skip baseline features", skipped)
        return has_baseline, final_targets, extra_flags
    def _parse_policy_keepsort(self, has_baseline, final_targets, extra_flags):
        self.dist_log(
            "policy 'keep_sort' is on, dispatch-able targets", final_targets, "\n"
            "are 'not' sorted depend on the highest interest but"
            "as specified in the dispatch-able source or the extra group"
        )
        return has_baseline, final_targets, extra_flags
    def _parse_policy_not_keepsort(self, has_baseline, final_targets, extra_flags):
        final_targets = self.feature_sorted(final_targets, reverse=True)
        return has_baseline, final_targets, extra_flags
    def _parse_policy_maxopt(self, has_baseline, final_targets, extra_flags):
        if self.cc_has_debug:
            self.dist_log("debug mode is detected, policy 'maxopt' is skipped.")
        elif self.cc_noopt:
            self.dist_log("optimization is disabled, policy 'maxopt' is skipped.")
        else:
            flags = self.cc_flags["opt"]
            if not flags:
                self.dist_log(
                    "current compiler doesn't support optimization flags, "
                    "policy 'maxopt' is skipped", stderr=True
                )
            else:
                extra_flags += flags
        return has_baseline, final_targets, extra_flags
    def _parse_policy_werror(self, has_baseline, final_targets, extra_flags):
        flags = self.cc_flags["werror"]
        if not flags:
            self.dist_log(
                "current compiler doesn't support werror flags, "
                "warnings will 'not' treated as errors", stderr=True
            )
        else:
            self.dist_log("compiler warnings are treated as errors")
            extra_flags += flags
        return has_baseline, final_targets, extra_flags
    def _parse_policy_autovec(self, has_baseline, final_targets, extra_flags):
        skipped = []
        for tar in final_targets[:]:
            if isinstance(tar, str):
                can = self.feature_can_autovec(tar)
            else:
                can = all([
                    self.feature_can_autovec(t)
                    for t in tar
                ])
            if not can:
                final_targets.remove(tar)
                skipped.append(tar)
        if skipped:
            self.dist_log("skip non auto-vectorized features", skipped)
        return has_baseline, final_targets, extra_flags
class CCompilerOpt(_Config, _Distutils, _Cache, _CCompiler, _Feature, _Parse):
    def __init__(self, ccompiler, cpu_baseline="min", cpu_dispatch="max", cache_path=None):
        _Config.__init__(self)
        _Distutils.__init__(self, ccompiler)
        _Cache.__init__(self, cache_path, self.dist_info(), cpu_baseline, cpu_dispatch)
        _CCompiler.__init__(self)
        _Feature.__init__(self)
        if not self.cc_noopt and self.cc_has_native:
            self.dist_log(
                "native flag is specified through environment variables. "
                "force cpu-baseline='native'"
            )
            cpu_baseline = "native"
        _Parse.__init__(self, cpu_baseline, cpu_dispatch)
        self._requested_baseline = cpu_baseline
        self._requested_dispatch = cpu_dispatch
        self.sources_status = getattr(self, "sources_status", {})
        self.cache_private.add("sources_status")
        self.hit_cache = hasattr(self, "hit_cache")
    def is_cached(self):
        return self.cache_infile and self.hit_cache
    def cpu_baseline_flags(self):
        return self.parse_baseline_flags
    def cpu_baseline_names(self):
        return self.parse_baseline_names
    def cpu_dispatch_names(self):
        return self.parse_dispatch_names
    def try_dispatch(self, sources, src_dir=None, **kwargs):
        to_compile = {}
        baseline_flags = self.cpu_baseline_flags()
        include_dirs = kwargs.setdefault("include_dirs", [])
        for src in sources:
            output_dir = os.path.dirname(src)
            if src_dir:
                if not output_dir.startswith(src_dir):
                    output_dir = os.path.join(src_dir, output_dir)
                if output_dir not in include_dirs:
                    include_dirs.append(output_dir)
            has_baseline, targets, extra_flags = self.parse_targets(src)
            nochange = self._generate_config(output_dir, src, targets, has_baseline)
            for tar in targets:
                tar_src = self._wrap_target(output_dir, src, tar, nochange=nochange)
                flags = tuple(extra_flags + self.feature_flags(tar))
                to_compile.setdefault(flags, []).append(tar_src)
            if has_baseline:
                flags = tuple(extra_flags + baseline_flags)
                to_compile.setdefault(flags, []).append(src)
            self.sources_status[src] = (has_baseline, targets)
        objects = []
        for flags, srcs in to_compile.items():
            objects += self.dist_compile(srcs, list(flags), **kwargs)
        return objects
    def generate_dispatch_header(self, header_path):
        self.dist_log("generate CPU dispatch header: (%s)" % header_path)
        baseline_names = self.cpu_baseline_names()
        dispatch_names = self.cpu_dispatch_names()
        baseline_len = len(baseline_names)
        dispatch_len = len(dispatch_names)
        header_dir = os.path.dirname(header_path)
        if not os.path.exists(header_dir):
            self.dist_log(
                f"dispatch header dir {header_dir} does not exist, creating it",
                stderr=True
            )
            os.makedirs(header_dir)
        with open(header_path, 'w') as f:
            baseline_calls = ' \\\n'.join([
                (
                    "\t%sWITH_CPU_EXPAND_(MACRO_TO_CALL(%s, __VA_ARGS__))"
                ) % (self.conf_c_prefix, f)
                for f in baseline_names
            ])
            dispatch_calls = ' \\\n'.join([
                (
                    "\t%sWITH_CPU_EXPAND_(MACRO_TO_CALL(%s, __VA_ARGS__))"
                ) % (self.conf_c_prefix, f)
                for f in dispatch_names
            ])
            f.write(textwrap.dedent("""\
                /*
                 * AUTOGENERATED DON'T EDIT
                 * Please make changes to the code generator (distutils/ccompiler_opt.py)
                */
                {baseline_calls}
                {dispatch_calls}
            """).format(
                pfx=self.conf_c_prefix, baseline_str=" ".join(baseline_names),
                dispatch_str=" ".join(dispatch_names), baseline_len=baseline_len,
                dispatch_len=dispatch_len, baseline_calls=baseline_calls,
                dispatch_calls=dispatch_calls
            ))
            baseline_pre = ''
            for name in baseline_names:
                baseline_pre += self.feature_c_preprocessor(name, tabs=1) + '\n'
            dispatch_pre = ''
            for name in dispatch_names:
                dispatch_pre += textwrap.dedent("""\
                {pre}
                """).format(
                    pfx=self.conf_c_prefix_, name=name, pre=self.feature_c_preprocessor(
                    name, tabs=1
                ))
            f.write(textwrap.dedent("""\
            /******* baseline features *******/
            {baseline_pre}
            /******* dispatch features *******/
            {dispatch_pre}
            """).format(
                pfx=self.conf_c_prefix_, baseline_pre=baseline_pre,
                dispatch_pre=dispatch_pre
            ))
    def report(self, full=False):
        report = []
        platform_rows = []
        baseline_rows = []
        dispatch_rows = []
        report.append(("Platform", platform_rows))
        report.append(("", ""))
        report.append(("CPU baseline", baseline_rows))
        report.append(("", ""))
        report.append(("CPU dispatch", dispatch_rows))
        platform_rows.append(("Architecture", (
            "unsupported" if self.cc_on_noarch else self.cc_march)
        ))
        platform_rows.append(("Compiler", (
            "unix-like"   if self.cc_is_nocc   else self.cc_name)
        ))
        if self.cc_noopt:
            baseline_rows.append(("Requested", "optimization disabled"))
        else:
            baseline_rows.append(("Requested", repr(self._requested_baseline)))
        baseline_names = self.cpu_baseline_names()
        baseline_rows.append((
            "Enabled", (' '.join(baseline_names) if baseline_names else "none")
        ))
        baseline_flags = self.cpu_baseline_flags()
        baseline_rows.append((
            "Flags", (' '.join(baseline_flags) if baseline_flags else "none")
        ))
        extra_checks = []
        for name in baseline_names:
            extra_checks += self.feature_extra_checks(name)
        baseline_rows.append((
            "Extra checks", (' '.join(extra_checks) if extra_checks else "none")
        ))
        if self.cc_noopt:
            baseline_rows.append(("Requested", "optimization disabled"))
        else:
            dispatch_rows.append(("Requested", repr(self._requested_dispatch)))
        dispatch_names = self.cpu_dispatch_names()
        dispatch_rows.append((
            "Enabled", (' '.join(dispatch_names) if dispatch_names else "none")
        ))
        target_sources = {}
        for source, (_, targets) in self.sources_status.items():
            for tar in targets:
                target_sources.setdefault(tar, []).append(source)
        if not full or not target_sources:
            generated = ""
            for tar in self.feature_sorted(target_sources):
                sources = target_sources[tar]
                name = tar if isinstance(tar, str) else '(%s)' % ' '.join(tar)
                generated += name + "[%d] " % len(sources)
            dispatch_rows.append(("Generated", generated[:-1] if generated else "none"))
        else:
            dispatch_rows.append(("Generated", ''))
            for tar in self.feature_sorted(target_sources):
                sources = target_sources[tar]
                pretty_name = tar if isinstance(tar, str) else '(%s)' % ' '.join(tar)
                flags = ' '.join(self.feature_flags(tar))
                implies = ' '.join(self.feature_sorted(self.feature_implies(tar)))
                detect = ' '.join(self.feature_detect(tar))
                extra_checks = []
                for name in ((tar,) if isinstance(tar, str) else tar):
                    extra_checks += self.feature_extra_checks(name)
                extra_checks = (' '.join(extra_checks) if extra_checks else "none")
                dispatch_rows.append(('', ''))
                dispatch_rows.append((pretty_name, implies))
                dispatch_rows.append(("Flags", flags))
                dispatch_rows.append(("Extra checks", extra_checks))
                dispatch_rows.append(("Detect", detect))
                for src in sources:
                    dispatch_rows.append(("", src))
        text = []
        secs_len = [len(secs) for secs, _ in report]
        cols_len = [len(col) for _, rows in report for col, _ in rows]
        tab = ' ' * 2
        pad =  max(max(secs_len), max(cols_len))
        for sec, rows in report:
            if not sec:
                text.append("")
                continue
            sec += ' ' * (pad - len(sec))
            text.append(sec + tab + ': ')
            for col, val in rows:
                col += ' ' * (pad - len(col))
                text.append(tab + col + ': ' + val)
        return '\n'.join(text)
    def _wrap_target(self, output_dir, dispatch_src, target, nochange=False):
        assert(isinstance(target, (str, tuple)))
        if isinstance(target, str):
            ext_name = target_name = target
        else:
            ext_name = '.'.join(target)
            target_name = '__'.join(target)
        wrap_path = os.path.join(output_dir, os.path.basename(dispatch_src))
        wrap_path = "{0}.{2}{1}".format(*os.path.splitext(wrap_path), ext_name.lower())
        if nochange and os.path.exists(wrap_path):
            return wrap_path
        self.dist_log("wrap dispatch-able target -> ", wrap_path)
        features = self.feature_sorted(self.feature_implies_c(target))
        target_defs = [target_join + f for f in features]
        target_defs = '\n'.join(target_defs)
        with open(wrap_path, "w") as fd:
            fd.write(textwrap.dedent("""\
            /**
             * AUTOGENERATED DON'T EDIT
             * Please make changes to the code generator \
             (distutils/ccompiler_opt.py)
             */
            {target_defs}
            """).format(
                pfx=self.conf_c_prefix_, target_name=target_name,
                path=os.path.abspath(dispatch_src), target_defs=target_defs
            ))
        return wrap_path
    def _generate_config(self, output_dir, dispatch_src, targets, has_baseline=False):
        config_path = os.path.basename(dispatch_src).replace(".c", ".h")
        config_path = os.path.join(output_dir, config_path)
        cache_hash = self.cache_hash(targets, has_baseline)
        try:
            with open(config_path) as f:
                last_hash = f.readline().split("cache_hash:")
                if len(last_hash) == 2 and int(last_hash[1]) == cache_hash:
                    return True
        except IOError:
            pass
        self.dist_log("generate dispatched config -> ", config_path)
        dispatch_calls = []
        for tar in targets:
            if isinstance(tar, str):
                target_name = tar
            else:
                target_name = '__'.join([t for t in tar])
            req_detect = self.feature_detect(tar)
            req_detect = '&&'.join([
                "CHK(%s)" % f for f in req_detect
            ])
            dispatch_calls.append(
                "\t%sCPU_DISPATCH_EXPAND_(CB((%s), %s, __VA_ARGS__))" % (
                self.conf_c_prefix_, req_detect, target_name
            ))
        dispatch_calls = ' \\\n'.join(dispatch_calls)
        if has_baseline:
            baseline_calls = (
                "\t%sCPU_DISPATCH_EXPAND_(CB(__VA_ARGS__))"
            ) % self.conf_c_prefix_
        else:
            baseline_calls = ''
        with open(config_path, "w") as fd:
            fd.write(textwrap.dedent("""\
            // cache_hash:{cache_hash}
            /**
             * AUTOGENERATED DON'T EDIT
             * Please make changes to the code generator (distutils/ccompiler_opt.py)
             */
            {baseline_calls}
            {dispatch_calls}
            """).format(
                pfx=self.conf_c_prefix_, baseline_calls=baseline_calls,
                dispatch_calls=dispatch_calls, cache_hash=cache_hash
            ))
        return False
def new_ccompiler_opt(compiler, dispatch_hpath, **kwargs):
    opt = CCompilerOpt(compiler, **kwargs)
    if not os.path.exists(dispatch_hpath) or not opt.is_cached():
        opt.generate_dispatch_header(dispatch_hpath)
    return opt
