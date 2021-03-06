import sys
from numpy.distutils.fcompiler import FCompiler
compilers = ['NVHPCFCompiler']
class NVHPCFCompiler(FCompiler):
    compiler_type = 'nv'
    description = 'NVIDIA HPC SDK'
    version_pattern = r'\s*(nvfortran|(pg(f77|f90|fortran)) \(aka nvfortran\)) (?P<version>[\d.-]+).*'
    executables = {
        'version_cmd': ["<F90>", "-V"],
        'compiler_f77': ["nvfortran"],
        'compiler_fix': ["nvfortran", "-Mfixed"],
        'compiler_f90': ["nvfortran"],
        'linker_so': ["<F90>"],
        'archiver': ["ar", "-cr"],
        'ranlib': ["ranlib"]
    }
    pic_flags = ['-fpic']
    module_dir_switch = '-module '
    module_include_switch = '-I'
    def get_flags(self):
        opt = ['-Minform=inform', '-Mnosecond_underscore']
        return self.pic_flags + opt
    def get_flags_opt(self):
        return ['-fast']
    def get_flags_debug(self):
        return ['-g']
    def get_flags_linker_so(self):
        return ["-shared", '-fpic']
    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir
if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='nv').get_version())
