
import os
import sys
import copy
from subprocess import Popen, PIPE, check_output
import re
from distutils.unixccompiler import UnixCCompiler
from distutils.file_util import write_file
from distutils.errors import (DistutilsExecError, CCompilerError,
        CompileError, UnknownFileError)
from distutils.version import LooseVersion
from distutils.spawn import find_executable
def get_msvcr():
    msc_pos = sys.version.find('MSC v.')
    if msc_pos != -1:
        msc_ver = sys.version[msc_pos+6:msc_pos+10]
        if msc_ver == '1300':
            return ['msvcr70']
        elif msc_ver == '1310':
            return ['msvcr71']
        elif msc_ver == '1400':
            return ['msvcr80']
        elif msc_ver == '1500':
            return ['msvcr90']
        elif msc_ver == '1600':
            return ['msvcr100']
        else:
            raise ValueError("Unknown MS Compiler version %s " % msc_ver)
class CygwinCCompiler(UnixCCompiler):
    compiler_type = 'cygwin'
    obj_extension = ".o"
    static_lib_extension = ".a"
    shared_lib_extension = ".dll"
    static_lib_format = "lib%s%s"
    shared_lib_format = "%s%s"
    exe_extension = ".exe"
    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)
        status, details = check_config_h()
        self.debug_print("Python's GCC status: %s (details: %s)" %
                         (status, details))
        if status is not CONFIG_H_OK:
            self.warn(
                "Python's pyconfig.h doesn't seem to support your compiler. "
                "Reason: %s. "
                "Compiling may fail because of undefined preprocessor macros."
                % details)
        self.gcc_version, self.ld_version, self.dllwrap_version =            get_versions()
        self.debug_print(self.compiler_type + ": gcc %s, ld %s, dllwrap %s\n" %
                         (self.gcc_version,
                          self.ld_version,
                          self.dllwrap_version) )
        if self.ld_version >= "2.10.90":
            self.linker_dll = "gcc"
        else:
            self.linker_dll = "dllwrap"
        if self.ld_version >= "2.13":
            shared_option = "-shared"
        else:
            shared_option = "-mdll -static"
        self.set_executables(compiler='gcc -mcygwin -O -Wall',
                             compiler_so='gcc -mcygwin -mdll -O -Wall',
                             compiler_cxx='g++ -mcygwin -O -Wall',
                             linker_exe='gcc -mcygwin',
                             linker_so=('%s -mcygwin %s' %
                                        (self.linker_dll, shared_option)))
        if self.gcc_version == "2.91.57":
            self.dll_libraries=["msvcrt"]
            self.warn(
                "Consider upgrading to a newer version of gcc")
        else:
            self.dll_libraries = get_msvcr()
    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        if ext == '.rc' or ext == '.res':
            try:
                self.spawn(["windres", "-i", src, "-o", obj])
            except DistutilsExecError as msg:
                raise CompileError(msg)
        else:
            try:
                self.spawn(self.compiler_so + cc_args + [src, '-o', obj] +
                           extra_postargs)
            except DistutilsExecError as msg:
                raise CompileError(msg)
    def link(self, target_desc, objects, output_filename, output_dir=None,
             libraries=None, library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        extra_preargs = copy.copy(extra_preargs or [])
        libraries = copy.copy(libraries or [])
        objects = copy.copy(objects or [])
        libraries.extend(self.dll_libraries)
        if ((export_symbols is not None) and
            (target_desc != self.EXECUTABLE or self.linker_dll == "gcc")):
            temp_dir = os.path.dirname(objects[0])
            (dll_name, dll_extension) = os.path.splitext(
                os.path.basename(output_filename))
            def_file = os.path.join(temp_dir, dll_name + ".def")
            lib_file = os.path.join(temp_dir, 'lib' + dll_name + ".a")
            contents = [
                "LIBRARY %s" % os.path.basename(output_filename),
                "EXPORTS"]
            for sym in export_symbols:
                contents.append(sym)
            self.execute(write_file, (def_file, contents),
                         "writing %s" % def_file)
            if self.linker_dll == "dllwrap":
                extra_preargs.extend(["--output-lib", lib_file])
                extra_preargs.extend(["--def", def_file])
            else:
                objects.append(def_file)
        if not debug:
            extra_preargs.append("-s")
        UnixCCompiler.link(self, target_desc, objects, output_filename,
                           output_dir, libraries, library_dirs,
                           runtime_library_dirs,
                           None,
                           debug, extra_preargs, extra_postargs, build_temp,
                           target_lang)
    def object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
        if output_dir is None:
            output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            base, ext = os.path.splitext(os.path.normcase(src_name))
            if ext not in (self.src_extensions + ['.rc','.res']):
                raise UnknownFileError("unknown file type '%s' (from '%s')" %                      (ext, src_name))
            if strip_dir:
                base = os.path.basename (base)
            if ext in ('.res', '.rc'):
                obj_names.append (os.path.join(output_dir,
                                              base + ext + self.obj_extension))
            else:
                obj_names.append (os.path.join(output_dir,
                                               base + self.obj_extension))
        return obj_names
class Mingw32CCompiler(CygwinCCompiler):
    compiler_type = 'mingw32'
    def __init__(self, verbose=0, dry_run=0, force=0):
        CygwinCCompiler.__init__ (self, verbose, dry_run, force)
        if self.ld_version >= "2.13":
            shared_option = "-shared"
        else:
            shared_option = "-mdll -static"
        if self.gcc_version <= "2.91.57":
            entry_point = '--entry _DllMain@12'
        else:
            entry_point = ''
        if is_cygwingcc():
            raise CCompilerError(
                'Cygwin gcc cannot be used with --compiler=mingw32')
        self.set_executables(compiler='gcc -O -Wall',
                             compiler_so='gcc -mdll -O -Wall',
                             compiler_cxx='g++ -O -Wall',
                             linker_exe='gcc',
                             linker_so='%s %s %s'
                                        % (self.linker_dll, shared_option,
                                           entry_point))
        self.dll_libraries=[]
        self.dll_libraries = get_msvcr()
CONFIG_H_OK = "ok"
CONFIG_H_NOTOK = "not ok"
CONFIG_H_UNCERTAIN = "uncertain"
def check_config_h():
    from distutils import sysconfig
    if "GCC" in sys.version:
        return CONFIG_H_OK, "sys.version mentions 'GCC'"
    fn = sysconfig.get_config_h_filename()
    try:
        config_h = open(fn)
        try:
            if "__GNUC__" in config_h.read():
                return CONFIG_H_OK, "'%s' mentions '__GNUC__'" % fn
            else:
                return CONFIG_H_NOTOK, "'%s' does not mention '__GNUC__'" % fn
        finally:
            config_h.close()
    except OSError as exc:
        return (CONFIG_H_UNCERTAIN,
                "couldn't read '%s': %s" % (fn, exc.strerror))
RE_VERSION = re.compile(br'(\d+\.\d+(\.\d+)*)')
def _find_exe_version(cmd):
    executable = cmd.split()[0]
    if find_executable(executable) is None:
        return None
    out = Popen(cmd, shell=True, stdout=PIPE).stdout
    try:
        out_string = out.read()
    finally:
        out.close()
    result = RE_VERSION.search(out_string)
    if result is None:
        return None
    return LooseVersion(result.group(1).decode())
def get_versions():
    commands = ['gcc -dumpversion', 'ld -v', 'dllwrap --version']
    return tuple([_find_exe_version(cmd) for cmd in commands])
def is_cygwingcc():
    out_string = check_output(['gcc', '-dumpmachine'])
    return out_string.strip().endswith(b'cygwin')
