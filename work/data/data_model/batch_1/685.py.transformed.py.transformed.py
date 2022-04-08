
import os
import platform
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.version import StrictVersion
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
                                       msvc_runtime_version,
                                       msvc_runtime_major,
                                       get_build_architecture)
def get_msvcr_replacement():
    msvcr = msvc_runtime_library()
    return [] if msvcr is None else [msvcr]
distutils.cygwinccompiler.get_msvcr = get_msvcr_replacement
_START = re.compile(r'\[Ordinal/Name Pointer\] Table')
_TABLE = re.compile(r'^\s+\[([\s*[0-9]*)\] ([a-zA-Z0-9_]*)')
class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
    compiler_type = 'mingw32'
    def __init__ (self,
                  verbose=0,
                  dry_run=0,
                  force=0):
        distutils.cygwinccompiler.CygwinCCompiler.__init__ (self, verbose,
                                                            dry_run, force)
        if self.gcc_version is None:
            try:
                out_string  = subprocess.check_output(['gcc', '-dumpversion'])
            except (OSError, CalledProcessError):
                out_string = ""
            result = re.search(r'(\d+\.\d+)', out_string)
            if result:
                self.gcc_version = StrictVersion(result.group(1))
        if self.gcc_version <= "2.91.57":
            entry_point = '--entry _DllMain@12'
        else:
            entry_point = ''
        if self.linker_dll == 'dllwrap':
            self.linker = 'dllwrap'
        elif self.linker_dll == 'gcc':
            self.linker = 'g++'
        build_import_library()
        msvcr_success = build_msvcr_library()
        msvcr_dbg_success = build_msvcr_library(debug=True)
        if msvcr_success or msvcr_dbg_success:
            self.define_macro('NPY_MINGW_USE_CUSTOM_MSVCR')
        msvcr_version = msvc_runtime_version()
        if msvcr_version:
            self.define_macro('__MSVCRT_VERSION__', '0x%04i' % msvcr_version)
        if get_build_architecture() == 'AMD64':
            if self.gcc_version < "4.0":
                self.set_executables(
                    compiler='gcc -g -DDEBUG -DMS_WIN64 -mno-cygwin -O0 -Wall',
                    compiler_so='gcc -g -DDEBUG -DMS_WIN64 -mno-cygwin -O0'
                                ' -Wall -Wstrict-prototypes',
                    linker_exe='gcc -g -mno-cygwin',
                    linker_so='gcc -g -mno-cygwin -shared')
            else:
                self.set_executables(
                    compiler='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall',
                    compiler_so='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall -Wstrict-prototypes',
                    linker_exe='gcc -g',
                    linker_so='gcc -g -shared')
        else:
            if self.gcc_version <= "3.0.0":
                self.set_executables(
                    compiler='gcc -mno-cygwin -O2 -w',
                    compiler_so='gcc -mno-cygwin -mdll -O2 -w'
                                ' -Wstrict-prototypes',
                    linker_exe='g++ -mno-cygwin',
                    linker_so='%s -mno-cygwin -mdll -static %s' %
                              (self.linker, entry_point))
            elif self.gcc_version < "4.0":
                self.set_executables(
                    compiler='gcc -mno-cygwin -O2 -Wall',
                    compiler_so='gcc -mno-cygwin -O2 -Wall'
                                ' -Wstrict-prototypes',
                    linker_exe='g++ -mno-cygwin',
                    linker_so='g++ -mno-cygwin -shared')
            else:
                self.set_executables(compiler='gcc -O2 -Wall',
                                     compiler_so='gcc -O2 -Wall -Wstrict-prototypes',
                                     linker_exe='g++ ',
                                     linker_so='g++ -shared')
        self.compiler_cxx = ['g++']
        return
    def link(self,
             target_desc,
             objects,
             output_filename,
             output_dir,
             libraries,
             library_dirs,
             runtime_library_dirs,
             export_symbols = None,
             debug=0,
             extra_preargs=None,
             extra_postargs=None,
             build_temp=None,
             target_lang=None):
        runtime_library = msvc_runtime_library()
        if runtime_library:
            if not libraries:
                libraries = []
            libraries.append(runtime_library)
        args = (self,
                target_desc,
                objects,
                output_filename,
                output_dir,
                libraries,
                library_dirs,
                runtime_library_dirs,
                None,
                debug,
                extra_preargs,
                extra_postargs,
                build_temp,
                target_lang)
        if self.gcc_version < "3.0.0":
            func = distutils.cygwinccompiler.CygwinCCompiler.link
        else:
            func = UnixCCompiler.link
        func(*args[:func.__code__.co_argcount])
        return
    def object_filenames (self,
                          source_filenames,
                          strip_dir=0,
                          output_dir=''):
        if output_dir is None: output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            (base, ext) = os.path.splitext (os.path.normcase(src_name))
            drv, base = os.path.splitdrive(base)
            if drv:
                base = base[1:]
            if ext not in (self.src_extensions + ['.rc', '.res']):
                raise UnknownFileError(
                      "unknown file type '%s' (from '%s')" %                      (ext, src_name))
            if strip_dir:
                base = os.path.basename (base)
            if ext == '.res' or ext == '.rc':
                obj_names.append (os.path.join (output_dir,
                                                base + ext + self.obj_extension))
            else:
                obj_names.append (os.path.join (output_dir,
                                                base + self.obj_extension))
        return obj_names
def find_python_dll():
    stems = [sys.prefix]
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)
    elif hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix:
        stems.append(sys.real_prefix)
    sub_dirs = ['', 'lib', 'bin']
    lib_dirs = []
    for stem in stems:
        for folder in sub_dirs:
            lib_dirs.append(os.path.join(stem, folder))
    if 'SYSTEMROOT' in os.environ:
        lib_dirs.append(os.path.join(os.environ['SYSTEMROOT'], 'System32'))
    major_version, minor_version = tuple(sys.version_info[:2])
    implementation = platform.python_implementation()
    if implementation == 'CPython':
        dllname = f'python{major_version}{minor_version}.dll'
    elif implementation == 'PyPy':
        dllname = f'libpypy{major_version}-c.dll'
    else:
        dllname = 'Unknown platform {implementation}'
    print("Looking for %s" % dllname)
    for folder in lib_dirs:
        dll = os.path.join(folder, dllname)
        if os.path.exists(dll):
            return dll
    raise ValueError("%s not found in %s" % (dllname, lib_dirs))
def dump_table(dll):
    st = subprocess.check_output(["objdump.exe", "-p", dll])
    return st.split(b'\n')
def generate_def(dll, dfile):
    dump = dump_table(dll)
    for i in range(len(dump)):
        if _START.match(dump[i].decode()):
            break
    else:
        raise ValueError("Symbol table not found")
    syms = []
    for j in range(i+1, len(dump)):
        m = _TABLE.match(dump[j].decode())
        if m:
            syms.append((int(m.group(1).strip()), m.group(2)))
        else:
            break
    if len(syms) == 0:
        log.warn('No symbols found in %s' % dll)
    with open(dfile, 'w') as d:
        d.write('LIBRARY        %s\n' % os.path.basename(dll))
        d.write(';CODE          PRELOAD MOVEABLE DISCARDABLE\n')
        d.write(';DATA          PRELOAD SINGLE\n')
        d.write('\nEXPORTS\n')
        for s in syms:
            d.write('%s\n' % s[1])
def find_dll(dll_name):
    arch = {'AMD64' : 'amd64',
            'Intel' : 'x86'}[get_build_architecture()]
    def _find_dll_in_winsxs(dll_name):
        winsxs_path = os.path.join(os.environ.get('WINDIR', r'C:\WINDOWS'),
                                   'winsxs')
        if not os.path.exists(winsxs_path):
            return None
        for root, dirs, files in os.walk(winsxs_path):
            if dll_name in files and arch in root:
                return os.path.join(root, dll_name)
        return None
    def _find_dll_in_path(dll_name):
        for path in [sys.prefix] + os.environ['PATH'].split(';'):
            filepath = os.path.join(path, dll_name)
            if os.path.exists(filepath):
                return os.path.abspath(filepath)
    return _find_dll_in_winsxs(dll_name) or _find_dll_in_path(dll_name)
def build_msvcr_library(debug=False):
    if os.name != 'nt':
        return False
    msvcr_ver = msvc_runtime_major()
    if msvcr_ver is None:
        log.debug('Skip building import library: '
                  'Runtime is not compiled with MSVC')
        return False
    if msvcr_ver < 80:
        log.debug('Skip building msvcr library:'
                  ' custom functionality not present')
        return False
    msvcr_name = msvc_runtime_library()
    if debug:
        msvcr_name += 'd'
    out_name = "lib%s.a" % msvcr_name
    out_file = os.path.join(sys.prefix, 'libs', out_name)
    if os.path.isfile(out_file):
        log.debug('Skip building msvcr library: "%s" exists' %
                  (out_file,))
        return True
    msvcr_dll_name = msvcr_name + '.dll'
    dll_file = find_dll(msvcr_dll_name)
    if not dll_file:
        log.warn('Cannot build msvcr library: "%s" not found' %
                 msvcr_dll_name)
        return False
    def_name = "lib%s.def" % msvcr_name
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    log.info('Building msvcr library: "%s" (from %s)'             % (out_file, dll_file))
    generate_def(dll_file, def_file)
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    retcode = subprocess.call(cmd)
    os.remove(def_file)
    return (not retcode)
def build_import_library():
    if os.name != 'nt':
        return
    arch = get_build_architecture()
    if arch == 'AMD64':
        return _build_import_library_amd64()
    elif arch == 'Intel':
        return _build_import_library_x86()
    else:
        raise ValueError("Unhandled arch %s" % arch)
def _check_for_import_lib():
    major_version, minor_version = tuple(sys.version_info[:2])
    patterns = ['libpython%d%d.a',
                'libpython%d%d.dll.a',
                'libpython%d.%d.dll.a']
    stems = [sys.prefix]
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)
    elif hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix:
        stems.append(sys.real_prefix)
    sub_dirs = ['libs', 'lib']
    candidates = []
    for pat in patterns:
        filename = pat % (major_version, minor_version)
        for stem_dir in stems:
            for folder in sub_dirs:
                candidates.append(os.path.join(stem_dir, folder, filename))
    for fullname in candidates:
        if os.path.isfile(fullname):
            return (True, fullname)
    return (False, candidates[0])
def _build_import_library_amd64():
    out_exists, out_file = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return
    dll_file = find_python_dll()
    log.info('Building import library (arch=AMD64): "%s" (from %s)' %
             (out_file, dll_file))
    def_name = "python%d%d.def" % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    generate_def(dll_file, def_file)
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    subprocess.check_call(cmd)
def _build_import_library_x86():
    out_exists, out_file = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return
    lib_name = "python%d%d.lib" % tuple(sys.version_info[:2])
    lib_file = os.path.join(sys.prefix, 'libs', lib_name)
    if not os.path.isfile(lib_file):
        if hasattr(sys, 'base_prefix'):
            base_lib = os.path.join(sys.base_prefix, 'libs', lib_name)
        elif hasattr(sys, 'real_prefix'):
            base_lib = os.path.join(sys.real_prefix, 'libs', lib_name)
        else:
            base_lib = ''
        if os.path.isfile(base_lib):
            lib_file = base_lib
        else:
            log.warn('Cannot build import library: "%s" not found', lib_file)
            return
    log.info('Building import library (ARCH=x86): "%s"', out_file)
    from numpy.distutils import lib2def
    def_name = "python%d%d.def" % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    nm_output = lib2def.getnm(
            lib2def.DEFAULT_NM + [lib_file], shell=False)
    dlist, flist = lib2def.parse_nm(nm_output)
    with open(def_file, 'w') as fid:
        lib2def.output_def(dlist, flist, lib2def.DEF_HEADER, fid)
    dll_name = find_python_dll ()
    cmd = ["dlltool",
           "--dllname", dll_name,
           "--def", def_file,
           "--output-lib", out_file]
    status = subprocess.check_output(cmd)
    if status:
        log.warn('Failed to build import library for gcc. Linking will fail.')
    return
_MSVCRVER_TO_FULLVER = {}
if sys.platform == 'win32':
    try:
        import msvcrt
        _MSVCRVER_TO_FULLVER['80'] = "8.0.50727.42"
        _MSVCRVER_TO_FULLVER['90'] = "9.0.21022.8"
        _MSVCRVER_TO_FULLVER['100'] = "10.0.30319.460"
        _MSVCRVER_TO_FULLVER['140'] = "14.15.26726.0"
        if hasattr(msvcrt, "CRT_ASSEMBLY_VERSION"):
            major, minor, rest = msvcrt.CRT_ASSEMBLY_VERSION.split(".", 2)
            _MSVCRVER_TO_FULLVER[major + minor] = msvcrt.CRT_ASSEMBLY_VERSION
            del major, minor, rest
    except ImportError:
        log.warn('Cannot import msvcrt: using manifest will not be possible')
def msvc_manifest_xml(maj, min):
    try:
        fullver = _MSVCRVER_TO_FULLVER[str(maj * 10 + min)]
    except KeyError:
        raise ValueError("Version %d,%d of MSVCRT not supported yet" %
                         (maj, min))
    template = textwrap.dedent("""\
        <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
          <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
            <security>
              <requestedPrivileges>
                <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>
              </requestedPrivileges>
            </security>
          </trustInfo>
          <dependency>
            <dependentAssembly>
              <assemblyIdentity type="win32" name="Microsoft.VC%(maj)d%(min)d.CRT" version="%(fullver)s" processorArchitecture="*" publicKeyToken="1fc8b3b9a1e18e3b"></assemblyIdentity>
            </dependentAssembly>
          </dependency>
        </assembly>""")
    return template % {'fullver': fullver, 'maj': maj, 'min': min}
def manifest_rc(name, type='dll'):
    if type == 'dll':
        rctype = 2
    elif type == 'exe':
        rctype = 1
    else:
        raise ValueError("Type %s not supported" % type)
    return % (rctype, name)
def check_embedded_msvcr_match_linked(msver):
    maj = msvc_runtime_major()
    if maj:
        if not maj == int(msver):
            raise ValueError(
                  "Discrepancy between linked msvcr "                  "(%d) and the one about to be embedded "                  "(%d)" % (int(msver), maj))
def configtest_name(config):
    base = os.path.basename(config._gen_temp_sourcefile("yo", [], "c"))
    return os.path.splitext(base)[0]
def manifest_name(config):
    root = configtest_name(config)
    exext = config.compiler.exe_extension
    return root + exext + ".manifest"
def rc_name(config):
    root = configtest_name(config)
    return root + ".rc"
def generate_manifest(config):
    msver = get_build_msvc_version()
    if msver is not None:
        if msver >= 8:
            check_embedded_msvcr_match_linked(msver)
            ma = int(msver)
            mi = int((msver - ma) * 10)
            manxml = msvc_manifest_xml(ma, mi)
            man = open(manifest_name(config), "w")
            config.temp_files.append(manifest_name(config))
            man.write(manxml)
            man.close()
