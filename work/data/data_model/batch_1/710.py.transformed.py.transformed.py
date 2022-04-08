
import json
from io import open
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
import sys
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.packaging.version import LegacyVersion
from setuptools.extern.more_itertools import unique_everseen
from .monkey import get_unpatched
if platform.system() == 'Windows':
    import winreg
    from os import environ
else:
    class winreg:
        HKEY_USERS = None
        HKEY_CURRENT_USER = None
        HKEY_LOCAL_MACHINE = None
        HKEY_CLASSES_ROOT = None
    environ = dict()
_msvc9_suppress_errors = (
    ImportError,
    distutils.errors.DistutilsPlatformError,
)
try:
    from distutils.msvc9compiler import Reg
except _msvc9_suppress_errors:
    pass
def msvc9_find_vcvarsall(version):
    vc_base = r'Software\%sMicrosoft\DevDiv\VCForPython\%0.1f'
    key = vc_base % ('', version)
    try:
        productdir = Reg.get_value(key, "installdir")
    except KeyError:
        try:
            key = vc_base % ('Wow6432Node\\', version)
            productdir = Reg.get_value(key, "installdir")
        except KeyError:
            productdir = None
    if productdir:
        vcvarsall = join(productdir, "vcvarsall.bat")
        if isfile(vcvarsall):
            return vcvarsall
    return get_unpatched(msvc9_find_vcvarsall)(version)
def msvc9_query_vcvarsall(ver, arch='x86', *args, **kwargs):
    try:
        orig = get_unpatched(msvc9_query_vcvarsall)
        return orig(ver, arch, *args, **kwargs)
    except distutils.errors.DistutilsPlatformError:
        pass
    except ValueError:
        pass
    try:
        return EnvironmentInfo(arch, ver).return_env()
    except distutils.errors.DistutilsPlatformError as exc:
        _augment_exception(exc, ver, arch)
        raise
def _msvc14_find_vc2015():
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"Software\Microsoft\VisualStudio\SxS\VC7",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_32KEY
        )
    except OSError:
        return None, None
    best_version = 0
    best_dir = None
    with key:
        for i in itertools.count():
            try:
                v, vc_dir, vt = winreg.EnumValue(key, i)
            except OSError:
                break
            if v and vt == winreg.REG_SZ and isdir(vc_dir):
                try:
                    version = int(float(v))
                except (ValueError, TypeError):
                    continue
                if version >= 14 and version > best_version:
                    best_version, best_dir = version, vc_dir
    return best_version, best_dir
def _msvc14_find_vc2017():
    root = environ.get("ProgramFiles(x86)") or environ.get("ProgramFiles")
    if not root:
        return None, None
    try:
        path = subprocess.check_output([
            join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe"),
            "-latest",
            "-prerelease",
            "-requiresAny",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-requires", "Microsoft.VisualStudio.Workload.WDExpress",
            "-property", "installationPath",
            "-products", "*",
        ]).decode(encoding="mbcs", errors="strict").strip()
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
        return None, None
    path = join(path, "VC", "Auxiliary", "Build")
    if isdir(path):
        return 15, path
    return None, None
PLAT_SPEC_TO_RUNTIME = {
    'x86': 'x86',
    'x86_amd64': 'x64',
    'x86_arm': 'arm',
    'x86_arm64': 'arm64'
}
def _msvc14_find_vcvarsall(plat_spec):
    _, best_dir = _msvc14_find_vc2017()
    vcruntime = None
    if plat_spec in PLAT_SPEC_TO_RUNTIME:
        vcruntime_plat = PLAT_SPEC_TO_RUNTIME[plat_spec]
    else:
        vcruntime_plat = 'x64' if 'amd64' in plat_spec else 'x86'
    if best_dir:
        vcredist = join(best_dir, "..", "..", "redist", "MSVC", "**",
                        vcruntime_plat, "Microsoft.VC14*.CRT",
                        "vcruntime140.dll")
        try:
            import glob
            vcruntime = glob.glob(vcredist, recursive=True)[-1]
        except (ImportError, OSError, LookupError):
            vcruntime = None
    if not best_dir:
        best_version, best_dir = _msvc14_find_vc2015()
        if best_version:
            vcruntime = join(best_dir, 'redist', vcruntime_plat,
                             "Microsoft.VC140.CRT", "vcruntime140.dll")
    if not best_dir:
        return None, None
    vcvarsall = join(best_dir, "vcvarsall.bat")
    if not isfile(vcvarsall):
        return None, None
    if not vcruntime or not isfile(vcruntime):
        vcruntime = None
    return vcvarsall, vcruntime
def _msvc14_get_vc_env(plat_spec):
    if "DISTUTILS_USE_SDK" in environ:
        return {
            key.lower(): value
            for key, value in environ.items()
        }
    vcvarsall, vcruntime = _msvc14_find_vcvarsall(plat_spec)
    if not vcvarsall:
        raise distutils.errors.DistutilsPlatformError(
            "Unable to find vcvarsall.bat"
        )
    try:
        out = subprocess.check_output(
            'cmd /u /c "{}" {} && set'.format(vcvarsall, plat_spec),
            stderr=subprocess.STDOUT,
        ).decode('utf-16le', errors='replace')
    except subprocess.CalledProcessError as exc:
        raise distutils.errors.DistutilsPlatformError(
            "Error executing {}".format(exc.cmd)
        ) from exc
    env = {
        key.lower(): value
        for key, _, value in
        (line.partition('=') for line in out.splitlines())
        if key and value
    }
    if vcruntime:
        env['py_vcruntime_redist'] = vcruntime
    return env
def msvc14_get_vc_env(plat_spec):
    try:
        return _msvc14_get_vc_env(plat_spec)
    except distutils.errors.DistutilsPlatformError as exc:
        _augment_exception(exc, 14.0)
        raise
def msvc14_gen_lib_options(*args, **kwargs):
    if "numpy.distutils" in sys.modules:
        import numpy as np
        if LegacyVersion(np.__version__) < LegacyVersion('1.11.2'):
            return np.distutils.ccompiler.gen_lib_options(*args, **kwargs)
    return get_unpatched(msvc14_gen_lib_options)(*args, **kwargs)
def _augment_exception(exc, version, arch=''):
    message = exc.args[0]
    if "vcvarsall" in message.lower() or "visual c" in message.lower():
        tmpl = 'Microsoft Visual C++ {version:0.1f} or greater is required.'
        message = tmpl.format(**locals())
        msdownload = 'www.microsoft.com/download/details.aspx?id=%d'
        if version == 9.0:
            if arch.lower().find('ia64') > -1:
                message += ' Get it with "Microsoft Windows SDK 7.0"'
            else:
                message += ' Get it from http://aka.ms/vcpython27'
        elif version == 10.0:
            message += ' Get it with "Microsoft Windows SDK 7.1": '
            message += msdownload % 8279
        elif version >= 14.0:
            message += (' Get it with "Microsoft C++ Build Tools": '
                        r'https://visualstudio.microsoft.com'
                        r'/visual-cpp-build-tools/')
    exc.args = (message, )
class PlatformInfo:
    current_cpu = environ.get('processor_architecture', '').lower()
    def __init__(self, arch):
        self.arch = arch.lower().replace('x64', 'amd64')
    @property
    def target_cpu(self):
        return self.arch[self.arch.find('_') + 1:]
    def target_is_x86(self):
        return self.target_cpu == 'x86'
    def current_is_x86(self):
        return self.current_cpu == 'x86'
    def current_dir(self, hidex86=False, x64=False):
        return (
            '' if (self.current_cpu == 'x86' and hidex86) else
            r'\x64' if (self.current_cpu == 'amd64' and x64) else
            r'\%s' % self.current_cpu
        )
    def target_dir(self, hidex86=False, x64=False):
        return (
            '' if (self.target_cpu == 'x86' and hidex86) else
            r'\x64' if (self.target_cpu == 'amd64' and x64) else
            r'\%s' % self.target_cpu
        )
    def cross_dir(self, forcex86=False):
        current = 'x86' if forcex86 else self.current_cpu
        return (
            '' if self.target_cpu == current else
            self.target_dir().replace('\\', '\\%s_' % current)
        )
class RegistryInfo:
    HKEYS = (winreg.HKEY_USERS,
             winreg.HKEY_CURRENT_USER,
             winreg.HKEY_LOCAL_MACHINE,
             winreg.HKEY_CLASSES_ROOT)
    def __init__(self, platform_info):
        self.pi = platform_info
    @property
    def visualstudio(self):
        return 'VisualStudio'
    @property
    def sxs(self):
        return join(self.visualstudio, 'SxS')
    @property
    def vc(self):
        return join(self.sxs, 'VC7')
    @property
    def vs(self):
        return join(self.sxs, 'VS7')
    @property
    def vc_for_python(self):
        return r'DevDiv\VCForPython'
    @property
    def microsoft_sdk(self):
        return 'Microsoft SDKs'
    @property
    def windows_sdk(self):
        return join(self.microsoft_sdk, 'Windows')
    @property
    def netfx_sdk(self):
        return join(self.microsoft_sdk, 'NETFXSDK')
    @property
    def windows_kits_roots(self):
        return r'Windows Kits\Installed Roots'
    def microsoft(self, key, x86=False):
        node64 = '' if self.pi.current_is_x86() or x86 else 'Wow6432Node'
        return join('Software', node64, 'Microsoft', key)
    def lookup(self, key, name):
        key_read = winreg.KEY_READ
        openkey = winreg.OpenKey
        closekey = winreg.CloseKey
        ms = self.microsoft
        for hkey in self.HKEYS:
            bkey = None
            try:
                bkey = openkey(hkey, ms(key), 0, key_read)
            except (OSError, IOError):
                if not self.pi.current_is_x86():
                    try:
                        bkey = openkey(hkey, ms(key, True), 0, key_read)
                    except (OSError, IOError):
                        continue
                else:
                    continue
            try:
                return winreg.QueryValueEx(bkey, name)[0]
            except (OSError, IOError):
                pass
            finally:
                if bkey:
                    closekey(bkey)
class SystemInfo:
    WinDir = environ.get('WinDir', '')
    ProgramFiles = environ.get('ProgramFiles', '')
    ProgramFilesx86 = environ.get('ProgramFiles(x86)', ProgramFiles)
    def __init__(self, registry_info, vc_ver=None):
        self.ri = registry_info
        self.pi = self.ri.pi
        self.known_vs_paths = self.find_programdata_vs_vers()
        self.vs_ver = self.vc_ver = (
            vc_ver or self._find_latest_available_vs_ver())
    def _find_latest_available_vs_ver(self):
        reg_vc_vers = self.find_reg_vs_vers()
        if not (reg_vc_vers or self.known_vs_paths):
            raise distutils.errors.DistutilsPlatformError(
                'No Microsoft Visual C++ version found')
        vc_vers = set(reg_vc_vers)
        vc_vers.update(self.known_vs_paths)
        return sorted(vc_vers)[-1]
    def find_reg_vs_vers(self):
        ms = self.ri.microsoft
        vckeys = (self.ri.vc, self.ri.vc_for_python, self.ri.vs)
        vs_vers = []
        for hkey, key in itertools.product(self.ri.HKEYS, vckeys):
            try:
                bkey = winreg.OpenKey(hkey, ms(key), 0, winreg.KEY_READ)
            except (OSError, IOError):
                continue
            with bkey:
                subkeys, values, _ = winreg.QueryInfoKey(bkey)
                for i in range(values):
                    with contextlib.suppress(ValueError):
                        ver = float(winreg.EnumValue(bkey, i)[0])
                        if ver not in vs_vers:
                            vs_vers.append(ver)
                for i in range(subkeys):
                    with contextlib.suppress(ValueError):
                        ver = float(winreg.EnumKey(bkey, i))
                        if ver not in vs_vers:
                            vs_vers.append(ver)
        return sorted(vs_vers)
    def find_programdata_vs_vers(self):
        vs_versions = {}
        instances_dir =            r'C:\ProgramData\Microsoft\VisualStudio\Packages\_Instances'
        try:
            hashed_names = listdir(instances_dir)
        except (OSError, IOError):
            return vs_versions
        for name in hashed_names:
            try:
                state_path = join(instances_dir, name, 'state.json')
                with open(state_path, 'rt', encoding='utf-8') as state_file:
                    state = json.load(state_file)
                vs_path = state['installationPath']
                listdir(join(vs_path, r'VC\Tools\MSVC'))
                vs_versions[self._as_float_version(
                    state['installationVersion'])] = vs_path
            except (OSError, IOError, KeyError):
                continue
        return vs_versions
    @staticmethod
    def _as_float_version(version):
        return float('.'.join(version.split('.')[:2]))
    @property
    def VSInstallDir(self):
        default = join(self.ProgramFilesx86,
                       'Microsoft Visual Studio %0.1f' % self.vs_ver)
        return self.ri.lookup(self.ri.vs, '%0.1f' % self.vs_ver) or default
    @property
    def VCInstallDir(self):
        path = self._guess_vc() or self._guess_vc_legacy()
        if not isdir(path):
            msg = 'Microsoft Visual C++ directory not found'
            raise distutils.errors.DistutilsPlatformError(msg)
        return path
    def _guess_vc(self):
        if self.vs_ver <= 14.0:
            return ''
        try:
            vs_dir = self.known_vs_paths[self.vs_ver]
        except KeyError:
            vs_dir = self.VSInstallDir
        guess_vc = join(vs_dir, r'VC\Tools\MSVC')
        try:
            vc_ver = listdir(guess_vc)[-1]
            self.vc_ver = self._as_float_version(vc_ver)
            return join(guess_vc, vc_ver)
        except (OSError, IOError, IndexError):
            return ''
    def _guess_vc_legacy(self):
        default = join(self.ProgramFilesx86,
                       r'Microsoft Visual Studio %0.1f\VC' % self.vs_ver)
        reg_path = join(self.ri.vc_for_python, '%0.1f' % self.vs_ver)
        python_vc = self.ri.lookup(reg_path, 'installdir')
        default_vc = join(python_vc, 'VC') if python_vc else default
        return self.ri.lookup(self.ri.vc, '%0.1f' % self.vs_ver) or default_vc
    @property
    def WindowsSdkVersion(self):
        if self.vs_ver <= 9.0:
            return '7.0', '6.1', '6.0a'
        elif self.vs_ver == 10.0:
            return '7.1', '7.0a'
        elif self.vs_ver == 11.0:
            return '8.0', '8.0a'
        elif self.vs_ver == 12.0:
            return '8.1', '8.1a'
        elif self.vs_ver >= 14.0:
            return '10.0', '8.1'
    @property
    def WindowsSdkLastVersion(self):
        return self._use_last_dir_name(join(self.WindowsSdkDir, 'lib'))
    @property
    def WindowsSdkDir(self):
        sdkdir = ''
        for ver in self.WindowsSdkVersion:
            loc = join(self.ri.windows_sdk, 'v%s' % ver)
            sdkdir = self.ri.lookup(loc, 'installationfolder')
            if sdkdir:
                break
        if not sdkdir or not isdir(sdkdir):
            path = join(self.ri.vc_for_python, '%0.1f' % self.vc_ver)
            install_base = self.ri.lookup(path, 'installdir')
            if install_base:
                sdkdir = join(install_base, 'WinSDK')
        if not sdkdir or not isdir(sdkdir):
            for ver in self.WindowsSdkVersion:
                intver = ver[:ver.rfind('.')]
                path = r'Microsoft SDKs\Windows Kits\%s' % intver
                d = join(self.ProgramFiles, path)
                if isdir(d):
                    sdkdir = d
        if not sdkdir or not isdir(sdkdir):
            for ver in self.WindowsSdkVersion:
                path = r'Microsoft SDKs\Windows\v%s' % ver
                d = join(self.ProgramFiles, path)
                if isdir(d):
                    sdkdir = d
        if not sdkdir:
            sdkdir = join(self.VCInstallDir, 'PlatformSDK')
        return sdkdir
    @property
    def WindowsSDKExecutablePath(self):
        if self.vs_ver <= 11.0:
            netfxver = 35
            arch = ''
        else:
            netfxver = 40
            hidex86 = True if self.vs_ver <= 12.0 else False
            arch = self.pi.current_dir(x64=True, hidex86=hidex86)
        fx = 'WinSDK-NetFx%dTools%s' % (netfxver, arch.replace('\\', '-'))
        regpaths = []
        if self.vs_ver >= 14.0:
            for ver in self.NetFxSdkVersion:
                regpaths += [join(self.ri.netfx_sdk, ver, fx)]
        for ver in self.WindowsSdkVersion:
            regpaths += [join(self.ri.windows_sdk, 'v%sA' % ver, fx)]
        for path in regpaths:
            execpath = self.ri.lookup(path, 'installationfolder')
            if execpath:
                return execpath
    @property
    def FSharpInstallDir(self):
        return self.ri.lookup(path, 'productdir') or ''
    @property
    def UniversalCRTSdkDir(self):
        vers = ('10', '81') if self.vs_ver >= 14.0 else ()
        for ver in vers:
            sdkdir = self.ri.lookup(self.ri.windows_kits_roots,
                                    'kitsroot%s' % ver)
            if sdkdir:
                return sdkdir or ''
    @property
    def UniversalCRTSdkLastVersion(self):
        return self._use_last_dir_name(join(self.UniversalCRTSdkDir, 'lib'))
    @property
    def NetFxSdkVersion(self):
        return (('4.7.2', '4.7.1', '4.7',
                 '4.6.2', '4.6.1', '4.6',
                 '4.5.2', '4.5.1', '4.5')
                if self.vs_ver >= 14.0 else ())
    @property
    def NetFxSdkDir(self):
        sdkdir = ''
        for ver in self.NetFxSdkVersion:
            loc = join(self.ri.netfx_sdk, ver)
            sdkdir = self.ri.lookup(loc, 'kitsinstallationfolder')
            if sdkdir:
                break
        return sdkdir
    @property
    def FrameworkDir32(self):
        guess_fw = join(self.WinDir, r'Microsoft.NET\Framework')
        return self.ri.lookup(self.ri.vc, 'frameworkdir32') or guess_fw
    @property
    def FrameworkDir64(self):
        guess_fw = join(self.WinDir, r'Microsoft.NET\Framework64')
        return self.ri.lookup(self.ri.vc, 'frameworkdir64') or guess_fw
    @property
    def FrameworkVersion32(self):
        return self._find_dot_net_versions(32)
    @property
    def FrameworkVersion64(self):
        return self._find_dot_net_versions(64)
    def _find_dot_net_versions(self, bits):
        reg_ver = self.ri.lookup(self.ri.vc, 'frameworkver%d' % bits)
        dot_net_dir = getattr(self, 'FrameworkDir%d' % bits)
        ver = reg_ver or self._use_last_dir_name(dot_net_dir, 'v') or ''
        if self.vs_ver >= 12.0:
            return ver, 'v4.0'
        elif self.vs_ver >= 10.0:
            return 'v4.0.30319' if ver.lower()[:2] != 'v4' else ver, 'v3.5'
        elif self.vs_ver == 9.0:
            return 'v3.5', 'v2.0.50727'
        elif self.vs_ver == 8.0:
            return 'v3.0', 'v2.0.50727'
    @staticmethod
    def _use_last_dir_name(path, prefix=''):
        matching_dirs = (
            dir_name
            for dir_name in reversed(listdir(path))
            if isdir(join(path, dir_name)) and
            dir_name.startswith(prefix)
        )
        return next(matching_dirs, None) or ''
class EnvironmentInfo:
    def __init__(self, arch, vc_ver=None, vc_min_ver=0):
        self.pi = PlatformInfo(arch)
        self.ri = RegistryInfo(self.pi)
        self.si = SystemInfo(self.ri, vc_ver)
        if self.vc_ver < vc_min_ver:
            err = 'No suitable Microsoft Visual C++ version found'
            raise distutils.errors.DistutilsPlatformError(err)
    @property
    def vs_ver(self):
        return self.si.vs_ver
    @property
    def vc_ver(self):
        return self.si.vc_ver
    @property
    def VSTools(self):
        paths = [r'Common7\IDE', r'Common7\Tools']
        if self.vs_ver >= 14.0:
            arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            paths += [r'Common7\IDE\CommonExtensions\Microsoft\TestWindow']
            paths += [r'Team Tools\Performance Tools']
            paths += [r'Team Tools\Performance Tools%s' % arch_subdir]
        return [join(self.si.VSInstallDir, path) for path in paths]
    @property
    def VCIncludes(self):
        return [join(self.si.VCInstallDir, 'Include'),
                join(self.si.VCInstallDir, r'ATLMFC\Include')]
    @property
    def VCLibraries(self):
        if self.vs_ver >= 15.0:
            arch_subdir = self.pi.target_dir(x64=True)
        else:
            arch_subdir = self.pi.target_dir(hidex86=True)
        paths = ['Lib%s' % arch_subdir, r'ATLMFC\Lib%s' % arch_subdir]
        if self.vs_ver >= 14.0:
            paths += [r'Lib\store%s' % arch_subdir]
        return [join(self.si.VCInstallDir, path) for path in paths]
    @property
    def VCStoreRefs(self):
        if self.vs_ver < 14.0:
            return []
        return [join(self.si.VCInstallDir, r'Lib\store\references')]
    @property
    def VCTools(self):
        si = self.si
        tools = [join(si.VCInstallDir, 'VCPackages')]
        forcex86 = True if self.vs_ver <= 10.0 else False
        arch_subdir = self.pi.cross_dir(forcex86)
        if arch_subdir:
            tools += [join(si.VCInstallDir, 'Bin%s' % arch_subdir)]
        if self.vs_ver == 14.0:
            path = 'Bin%s' % self.pi.current_dir(hidex86=True)
            tools += [join(si.VCInstallDir, path)]
        elif self.vs_ver >= 15.0:
            host_dir = (r'bin\HostX86%s' if self.pi.current_is_x86() else
                        r'bin\HostX64%s')
            tools += [join(
                si.VCInstallDir, host_dir % self.pi.target_dir(x64=True))]
            if self.pi.current_cpu != self.pi.target_cpu:
                tools += [join(
                    si.VCInstallDir, host_dir % self.pi.current_dir(x64=True))]
        else:
            tools += [join(si.VCInstallDir, 'Bin')]
        return tools
    @property
    def OSLibraries(self):
        if self.vs_ver <= 10.0:
            arch_subdir = self.pi.target_dir(hidex86=True, x64=True)
            return [join(self.si.WindowsSdkDir, 'Lib%s' % arch_subdir)]
        else:
            arch_subdir = self.pi.target_dir(x64=True)
            lib = join(self.si.WindowsSdkDir, 'lib')
            libver = self._sdk_subdir
            return [join(lib, '%sum%s' % (libver, arch_subdir))]
    @property
    def OSIncludes(self):
        include = join(self.si.WindowsSdkDir, 'include')
        if self.vs_ver <= 10.0:
            return [include, join(include, 'gl')]
        else:
            if self.vs_ver >= 14.0:
                sdkver = self._sdk_subdir
            else:
                sdkver = ''
            return [join(include, '%sshared' % sdkver),
                    join(include, '%sum' % sdkver),
                    join(include, '%swinrt' % sdkver)]
    @property
    def OSLibpath(self):
        ref = join(self.si.WindowsSdkDir, 'References')
        libpath = []
        if self.vs_ver <= 9.0:
            libpath += self.OSLibraries
        if self.vs_ver >= 11.0:
            libpath += [join(ref, r'CommonConfiguration\Neutral')]
        if self.vs_ver >= 14.0:
            libpath += [
                ref,
                join(self.si.WindowsSdkDir, 'UnionMetadata'),
                join(
                    ref, 'Windows.Foundation.UniversalApiContract', '1.0.0.0'),
                join(ref, 'Windows.Foundation.FoundationContract', '1.0.0.0'),
                join(
                    ref, 'Windows.Networking.Connectivity.WwanContract',
                    '1.0.0.0'),
                join(
                    self.si.WindowsSdkDir, 'ExtensionSDKs', 'Microsoft.VCLibs',
                    '%0.1f' % self.vs_ver, 'References', 'CommonConfiguration',
                    'neutral'),
            ]
        return libpath
    @property
    def SdkTools(self):
        return list(self._sdk_tools())
    def _sdk_tools(self):
        if self.vs_ver < 15.0:
            bin_dir = 'Bin' if self.vs_ver <= 11.0 else r'Bin\x86'
            yield join(self.si.WindowsSdkDir, bin_dir)
        if not self.pi.current_is_x86():
            arch_subdir = self.pi.current_dir(x64=True)
            path = 'Bin%s' % arch_subdir
            yield join(self.si.WindowsSdkDir, path)
        if self.vs_ver in (10.0, 11.0):
            if self.pi.target_is_x86():
                arch_subdir = ''
            else:
                arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            path = r'Bin\NETFX 4.0 Tools%s' % arch_subdir
            yield join(self.si.WindowsSdkDir, path)
        elif self.vs_ver >= 15.0:
            path = join(self.si.WindowsSdkDir, 'Bin')
            arch_subdir = self.pi.current_dir(x64=True)
            sdkver = self.si.WindowsSdkLastVersion
            yield join(path, '%s%s' % (sdkver, arch_subdir))
        if self.si.WindowsSDKExecutablePath:
            yield self.si.WindowsSDKExecutablePath
    @property
    def _sdk_subdir(self):
        ucrtver = self.si.WindowsSdkLastVersion
        return ('%s\\' % ucrtver) if ucrtver else ''
    @property
    def SdkSetup(self):
        if self.vs_ver > 9.0:
            return []
        return [join(self.si.WindowsSdkDir, 'Setup')]
    @property
    def FxTools(self):
        pi = self.pi
        si = self.si
        if self.vs_ver <= 10.0:
            include32 = True
            include64 = not pi.target_is_x86() and not pi.current_is_x86()
        else:
            include32 = pi.target_is_x86() or pi.current_is_x86()
            include64 = pi.current_cpu == 'amd64' or pi.target_cpu == 'amd64'
        tools = []
        if include32:
            tools += [join(si.FrameworkDir32, ver)
                      for ver in si.FrameworkVersion32]
        if include64:
            tools += [join(si.FrameworkDir64, ver)
                      for ver in si.FrameworkVersion64]
        return tools
    @property
    def NetFxSDKLibraries(self):
        if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
            return []
        arch_subdir = self.pi.target_dir(x64=True)
        return [join(self.si.NetFxSdkDir, r'lib\um%s' % arch_subdir)]
    @property
    def NetFxSDKIncludes(self):
        if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
            return []
        return [join(self.si.NetFxSdkDir, r'include\um')]
    @property
    def VsTDb(self):
        return [join(self.si.VSInstallDir, r'VSTSDB\Deploy')]
    @property
    def MSBuild(self):
        if self.vs_ver < 12.0:
            return []
        elif self.vs_ver < 15.0:
            base_path = self.si.ProgramFilesx86
            arch_subdir = self.pi.current_dir(hidex86=True)
        else:
            base_path = self.si.VSInstallDir
            arch_subdir = ''
        path = r'MSBuild\%0.1f\bin%s' % (self.vs_ver, arch_subdir)
        build = [join(base_path, path)]
        if self.vs_ver >= 15.0:
            build += [join(base_path, path, 'Roslyn')]
        return build
    @property
    def HTMLHelpWorkshop(self):
        if self.vs_ver < 11.0:
            return []
        return [join(self.si.ProgramFilesx86, 'HTML Help Workshop')]
    @property
    def UCRTLibraries(self):
        if self.vs_ver < 14.0:
            return []
        arch_subdir = self.pi.target_dir(x64=True)
        lib = join(self.si.UniversalCRTSdkDir, 'lib')
        ucrtver = self._ucrt_subdir
        return [join(lib, '%sucrt%s' % (ucrtver, arch_subdir))]
    @property
    def UCRTIncludes(self):
        if self.vs_ver < 14.0:
            return []
        include = join(self.si.UniversalCRTSdkDir, 'include')
        return [join(include, '%sucrt' % self._ucrt_subdir)]
    @property
    def _ucrt_subdir(self):
        ucrtver = self.si.UniversalCRTSdkLastVersion
        return ('%s\\' % ucrtver) if ucrtver else ''
    @property
    def FSharp(self):
        if 11.0 > self.vs_ver > 12.0:
            return []
        return [self.si.FSharpInstallDir]
    @property
    def VCRuntimeRedist(self):
        vcruntime = 'vcruntime%d0.dll' % self.vc_ver
        arch_subdir = self.pi.target_dir(x64=True).strip('\\')
        prefixes = []
        tools_path = self.si.VCInstallDir
        redist_path = dirname(tools_path.replace(r'\Tools', r'\Redist'))
        if isdir(redist_path):
            redist_path = join(redist_path, listdir(redist_path)[-1])
            prefixes += [redist_path, join(redist_path, 'onecore')]
        prefixes += [join(tools_path, 'redist')]
        crt_dirs = ('Microsoft.VC%d.CRT' % (self.vc_ver * 10),
                    'Microsoft.VC%d.CRT' % (int(self.vs_ver) * 10))
        for prefix, crt_dir in itertools.product(prefixes, crt_dirs):
            path = join(prefix, arch_subdir, crt_dir, vcruntime)
            if isfile(path):
                return path
    def return_env(self, exists=True):
        env = dict(
            include=self._build_paths('include',
                                      [self.VCIncludes,
                                       self.OSIncludes,
                                       self.UCRTIncludes,
                                       self.NetFxSDKIncludes],
                                      exists),
            lib=self._build_paths('lib',
                                  [self.VCLibraries,
                                   self.OSLibraries,
                                   self.FxTools,
                                   self.UCRTLibraries,
                                   self.NetFxSDKLibraries],
                                  exists),
            libpath=self._build_paths('libpath',
                                      [self.VCLibraries,
                                       self.FxTools,
                                       self.VCStoreRefs,
                                       self.OSLibpath],
                                      exists),
            path=self._build_paths('path',
                                   [self.VCTools,
                                    self.VSTools,
                                    self.VsTDb,
                                    self.SdkTools,
                                    self.SdkSetup,
                                    self.FxTools,
                                    self.MSBuild,
                                    self.HTMLHelpWorkshop,
                                    self.FSharp],
                                   exists),
        )
        if self.vs_ver >= 14 and isfile(self.VCRuntimeRedist):
            env['py_vcruntime_redist'] = self.VCRuntimeRedist
        return env
    def _build_paths(self, name, spec_path_lists, exists):
        spec_paths = itertools.chain.from_iterable(spec_path_lists)
        env_paths = environ.get(name, '').split(pathsep)
        paths = itertools.chain(spec_paths, env_paths)
        extant_paths = list(filter(isdir, paths)) if exists else paths
        if not extant_paths:
            msg = "%s environment variable is empty" % name.upper()
            raise distutils.errors.DistutilsPlatformError(msg)
        unique_paths = unique_everseen(extant_paths)
        return pathsep.join(unique_paths)
