import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
import distutils
from distutils.errors import DistutilsError
try:
    from threading import local as tlocal
except ImportError:
    from dummy_threading import local as tlocal
_tdata = tlocal()
_tmpdirs = []
def clean_up_temporary_directory():
    if _tmpdirs is not None:
        for d in _tmpdirs:
            try:
                shutil.rmtree(d)
            except OSError:
                pass
atexit.register(clean_up_temporary_directory)
from numpy.compat import npy_load_module
__all__ = ['Configuration', 'get_numpy_include_dirs', 'default_config_dict',
           'dict_append', 'appendpath', 'generate_config_py',
           'get_cmd', 'allpath', 'get_mathlibs',
           'terminal_has_colors', 'red_text', 'green_text', 'yellow_text',
           'blue_text', 'cyan_text', 'cyg2win32', 'mingw32', 'all_strings',
           'has_f_sources', 'has_cxx_sources', 'filter_sources',
           'get_dependencies', 'is_local_src_dir', 'get_ext_source_files',
           'get_script_files', 'get_lib_source_files', 'get_data_files',
           'dot_join', 'get_frame', 'minrelpath', 'njoin',
           'is_sequence', 'is_string', 'as_list', 'gpaths', 'get_language',
           'quote_args', 'get_build_architecture', 'get_info', 'get_pkg_info',
           'get_num_build_jobs']
class InstallableLib:
    def __init__(self, name, build_info, target_dir):
        self.name = name
        self.build_info = build_info
        self.target_dir = target_dir
def get_num_build_jobs():
    from numpy.distutils.core import get_distribution
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = multiprocessing.cpu_count()
    cpu_count = min(cpu_count, 8)
    envjobs = int(os.environ.get("NPY_NUM_BUILD_JOBS", cpu_count))
    dist = get_distribution()
    if dist is None:
        return envjobs
    cmdattr = (getattr(dist.get_command_obj('build'), 'parallel', None),
               getattr(dist.get_command_obj('build_ext'), 'parallel', None),
               getattr(dist.get_command_obj('build_clib'), 'parallel', None))
    if all(x is None for x in cmdattr):
        return envjobs
    else:
        return max(x for x in cmdattr if x is not None)
def quote_args(args):
    args = list(args)
    for i in range(len(args)):
        a = args[i]
        if ' ' in a and a[0] not in '"\'':
            args[i] = '"%s"' % (a)
    return args
def allpath(name):
    splitted = name.split('/')
    return os.path.join(*splitted)
def rel_path(path, parent_path):
    pd = os.path.realpath(os.path.abspath(parent_path))
    apath = os.path.realpath(os.path.abspath(path))
    if len(apath) < len(pd):
        return path
    if apath == pd:
        return ''
    if pd == apath[:len(pd)]:
        assert apath[len(pd)] in [os.sep], repr((path, apath[len(pd)]))
        path = apath[len(pd)+1:]
    return path
def get_path_from_frame(frame, parent_path=None):
    try:
        caller_file = eval('__file__', frame.f_globals, frame.f_locals)
        d = os.path.dirname(os.path.abspath(caller_file))
    except NameError:
        caller_name = eval('__name__', frame.f_globals, frame.f_locals)
        __import__(caller_name)
        mod = sys.modules[caller_name]
        if hasattr(mod, '__file__'):
            d = os.path.dirname(os.path.abspath(mod.__file__))
        else:
            d = os.path.abspath('.')
    if parent_path is not None:
        d = rel_path(d, parent_path)
    return d or '.'
def njoin(*path):
    paths = []
    for p in path:
        if is_sequence(p):
            paths.append(njoin(*p))
        else:
            assert is_string(p)
            paths.append(p)
    path = paths
    if not path:
        joined = ''
    else:
        joined = os.path.join(*path)
    if os.path.sep != '/':
        joined = joined.replace('/', os.path.sep)
    return minrelpath(joined)
def get_mathlibs(path=None):
    if path is not None:
        config_file = os.path.join(path, '_numpyconfig.h')
    else:
        dirs = get_numpy_include_dirs()
        for path in dirs:
            fn = os.path.join(path, '_numpyconfig.h')
            if os.path.exists(fn):
                config_file = fn
                break
        else:
            raise DistutilsError('_numpyconfig.h not found in numpy include '
                'dirs %r' % (dirs,))
    with open(config_file) as fid:
        mathlibs = []
        for line in fid:
            if line.startswith(s):
                value = line[len(s):].strip()
                if value:
                    mathlibs.extend(value.split(','))
    return mathlibs
def minrelpath(path):
    if not is_string(path):
        return path
    if '.' not in path:
        return path
    l = path.split(os.sep)
    while l:
        try:
            i = l.index('.', 1)
        except ValueError:
            break
        del l[i]
    j = 1
    while l:
        try:
            i = l.index('..', j)
        except ValueError:
            break
        if l[i-1]=='..':
            j += 1
        else:
            del l[i], l[i-1]
            j = 1
    if not l:
        return ''
    return os.sep.join(l)
def sorted_glob(fileglob):
    return sorted(glob.glob(fileglob))
def _fix_paths(paths, local_path, include_non_existing):
    assert is_sequence(paths), repr(type(paths))
    new_paths = []
    assert not is_string(paths), repr(paths)
    for n in paths:
        if is_string(n):
            if '*' in n or '?' in n:
                p = sorted_glob(n)
                p2 = sorted_glob(njoin(local_path, n))
                if p2:
                    new_paths.extend(p2)
                elif p:
                    new_paths.extend(p)
                else:
                    if include_non_existing:
                        new_paths.append(n)
                    print('could not resolve pattern in %r: %r' %
                            (local_path, n))
            else:
                n2 = njoin(local_path, n)
                if os.path.exists(n2):
                    new_paths.append(n2)
                else:
                    if os.path.exists(n):
                        new_paths.append(n)
                    elif include_non_existing:
                        new_paths.append(n)
                    if not os.path.exists(n):
                        print('non-existing path in %r: %r' %
                                (local_path, n))
        elif is_sequence(n):
            new_paths.extend(_fix_paths(n, local_path, include_non_existing))
        else:
            new_paths.append(n)
    return [minrelpath(p) for p in new_paths]
def gpaths(paths, local_path='', include_non_existing=True):
    if is_string(paths):
        paths = (paths,)
    return _fix_paths(paths, local_path, include_non_existing)
def make_temp_file(suffix='', prefix='', text=True):
    if not hasattr(_tdata, 'tempdir'):
        _tdata.tempdir = tempfile.mkdtemp()
        _tmpdirs.append(_tdata.tempdir)
    fid, name = tempfile.mkstemp(suffix=suffix,
                                 prefix=prefix,
                                 dir=_tdata.tempdir,
                                 text=text)
    fo = os.fdopen(fid, 'w')
    return fo, name
def terminal_has_colors():
    if sys.platform=='cygwin' and 'USE_COLOR' not in os.environ:
        return 0
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        try:
            import curses
            curses.setupterm()
            if (curses.tigetnum("colors") >= 0
                and curses.tigetnum("pairs") >= 0
                and ((curses.tigetstr("setf") is not None
                      and curses.tigetstr("setb") is not None)
                     or (curses.tigetstr("setaf") is not None
                         and curses.tigetstr("setab") is not None)
                     or curses.tigetstr("scp") is not None)):
                return 1
        except Exception:
            pass
    return 0
if terminal_has_colors():
    _colour_codes = dict(black=0, red=1, green=2, yellow=3,
                         blue=4, magenta=5, cyan=6, white=7, default=9)
    def colour_text(s, fg=None, bg=None, bold=False):
        seq = []
        if bold:
            seq.append('1')
        if fg:
            fgcode = 30 + _colour_codes.get(fg.lower(), 0)
            seq.append(str(fgcode))
        if bg:
            bgcode = 40 + _colour_codes.get(fg.lower(), 7)
            seq.append(str(bgcode))
        if seq:
            return '\x1b[%sm%s\x1b[0m' % (';'.join(seq), s)
        else:
            return s
else:
    def colour_text(s, fg=None, bg=None):
        return s
def default_text(s):
    return colour_text(s, 'default')
def red_text(s):
    return colour_text(s, 'red')
def green_text(s):
    return colour_text(s, 'green')
def yellow_text(s):
    return colour_text(s, 'yellow')
def cyan_text(s):
    return colour_text(s, 'cyan')
def blue_text(s):
    return colour_text(s, 'blue')
def cyg2win32(path):
    if sys.platform=='cygwin' and path.startswith('/cygdrive'):
        path = path[10] + ':' + os.path.normcase(path[11:])
    return path
def mingw32():
    if sys.platform=='win32':
        if os.environ.get('OSTYPE', '')=='msys':
            return True
        if os.environ.get('MSYSTEM', '')=='MINGW32':
            return True
    return False
def msvc_runtime_version():
    msc_pos = sys.version.find('MSC v.')
    if msc_pos != -1:
        msc_ver = int(sys.version[msc_pos+6:msc_pos+10])
    else:
        msc_ver = None
    return msc_ver
def msvc_runtime_library():
    ver = msvc_runtime_major ()
    if ver:
        if ver < 140:
            return "msvcr%i" % ver
        else:
            return "vcruntime%i" % ver
    else:
        return None
def msvc_runtime_major():
    major = {1300:  70,
             1310:  71,
             1400:  80,
             1500:  90,
             1600: 100,
             1900: 140,
    }.get(msvc_runtime_version(), None)
    return major
cxx_ext_match = re.compile(r'.*[.](cpp|cxx|cc)\Z', re.I).match
fortran_ext_match = re.compile(r'.*[.](f90|f95|f77|for|ftn|f)\Z', re.I).match
f90_ext_match = re.compile(r'.*[.](f90|f95)\Z', re.I).match
f90_module_name_match = re.compile(r'\s*module\s*(?P<name>[\w_]+)', re.I).match
def _get_f90_modules(source):
    if not f90_ext_match(source):
        return []
    modules = []
    with open(source, 'r') as f:
        for line in f:
            m = f90_module_name_match(line)
            if m:
                name = m.group('name')
                modules.append(name)
    return modules
def is_string(s):
    return isinstance(s, str)
def all_strings(lst):
    for item in lst:
        if not is_string(item):
            return False
    return True
def is_sequence(seq):
    if is_string(seq):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True
def is_glob_pattern(s):
    return is_string(s) and ('*' in s or '?' in s)
def as_list(seq):
    if is_sequence(seq):
        return list(seq)
    else:
        return [seq]
def get_language(sources):
    language = None
    for source in sources:
        if isinstance(source, str):
            if f90_ext_match(source):
                language = 'f90'
                break
            elif fortran_ext_match(source):
                language = 'f77'
    return language
def has_f_sources(sources):
    for source in sources:
        if fortran_ext_match(source):
            return True
    return False
def has_cxx_sources(sources):
    for source in sources:
        if cxx_ext_match(source):
            return True
    return False
def filter_sources(sources):
    c_sources = []
    cxx_sources = []
    f_sources = []
    fmodule_sources = []
    for source in sources:
        if fortran_ext_match(source):
            modules = _get_f90_modules(source)
            if modules:
                fmodule_sources.append(source)
            else:
                f_sources.append(source)
        elif cxx_ext_match(source):
            cxx_sources.append(source)
        else:
            c_sources.append(source)
    return c_sources, cxx_sources, f_sources, fmodule_sources
def _get_headers(directory_list):
    headers = []
    for d in directory_list:
        head = sorted_glob(os.path.join(d, "*.h"))
        headers.extend(head)
    return headers
def _get_directories(list_of_sources):
    direcs = []
    for f in list_of_sources:
        d = os.path.split(f)
        if d[0] != '' and not d[0] in direcs:
            direcs.append(d[0])
    return direcs
def _commandline_dep_string(cc_args, extra_postargs, pp_opts):
    cmdline = 'commandline: '
    cmdline += ' '.join(cc_args)
    cmdline += ' '.join(extra_postargs)
    cmdline += ' '.join(pp_opts) + '\n'
    return cmdline
def get_dependencies(sources):
    return _get_headers(_get_directories(sources))
def is_local_src_dir(directory):
    if not is_string(directory):
        return False
    abs_dir = os.path.abspath(directory)
    c = os.path.commonprefix([os.getcwd(), abs_dir])
    new_dir = abs_dir[len(c):].split(os.sep)
    if new_dir and not new_dir[0]:
        new_dir = new_dir[1:]
    if new_dir and new_dir[0]=='build':
        return False
    new_dir = os.sep.join(new_dir)
    return os.path.isdir(new_dir)
def general_source_files(top_path):
    pruned_directories = {'CVS':1, '.svn':1, 'build':1}
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        pruned = [ d for d in dirnames if d not in pruned_directories ]
        dirnames[:] = pruned
        for f in filenames:
            if not prune_file_pat.search(f):
                yield os.path.join(dirpath, f)
def general_source_directories_files(top_path):
    pruned_directories = ['CVS', '.svn', 'build']
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        pruned = [ d for d in dirnames if d not in pruned_directories ]
        dirnames[:] = pruned
        for d in dirnames:
            dpath = os.path.join(dirpath, d)
            rpath = rel_path(dpath, top_path)
            files = []
            for f in os.listdir(dpath):
                fn = os.path.join(dpath, f)
                if os.path.isfile(fn) and not prune_file_pat.search(fn):
                    files.append(fn)
            yield rpath, files
    dpath = top_path
    rpath = rel_path(dpath, top_path)
    filenames = [os.path.join(dpath, f) for f in os.listdir(dpath)                 if not prune_file_pat.search(f)]
    files = [f for f in filenames if os.path.isfile(f)]
    yield rpath, files
def get_ext_source_files(ext):
    filenames = []
    sources = [_m for _m in ext.sources if is_string(_m)]
    filenames.extend(sources)
    filenames.extend(get_dependencies(sources))
    for d in ext.depends:
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames
def get_script_files(scripts):
    scripts = [_m for _m in scripts if is_string(_m)]
    return scripts
def get_lib_source_files(lib):
    filenames = []
    sources = lib[1].get('sources', [])
    sources = [_m for _m in sources if is_string(_m)]
    filenames.extend(sources)
    filenames.extend(get_dependencies(sources))
    depends = lib[1].get('depends', [])
    for d in depends:
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames
def get_shared_lib_extension(is_python_ext=False):
    confvars = distutils.sysconfig.get_config_vars()
    so_ext = confvars.get('EXT_SUFFIX', None)
    if so_ext is None:
        so_ext = confvars.get('SO', '')
    if not is_python_ext:
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            if 'SOABI' in confvars:
                so_ext = so_ext.replace('.' + confvars.get('SOABI'), '', 1)
    return so_ext
def get_data_files(data):
    if is_string(data):
        return [data]
    sources = data[1]
    filenames = []
    for s in sources:
        if hasattr(s, '__call__'):
            continue
        if is_local_src_dir(s):
            filenames.extend(list(general_source_files(s)))
        elif is_string(s):
            if os.path.isfile(s):
                filenames.append(s)
            else:
                print('Not existing data file:', s)
        else:
            raise TypeError(repr(s))
    return filenames
def dot_join(*args):
    return '.'.join([a for a in args if a])
def get_frame(level=0):
    try:
        return sys._getframe(level+1)
    except AttributeError:
        frame = sys.exc_info()[2].tb_frame
        for _ in range(level+1):
            frame = frame.f_back
        return frame
class Configuration:
    _list_keys = ['packages', 'ext_modules', 'data_files', 'include_dirs',
                  'libraries', 'headers', 'scripts', 'py_modules',
                  'installed_libraries', 'define_macros']
    _dict_keys = ['package_dir', 'installed_pkg_config']
    _extra_keys = ['name', 'version']
    numpy_include_dirs = []
    def __init__(self,
                 package_name=None,
                 parent_name=None,
                 top_path=None,
                 package_path=None,
                 caller_level=1,
                 setup_name='setup.py',
                 **attrs):
        self.name = dot_join(parent_name, package_name)
        self.version = None
        caller_frame = get_frame(caller_level)
        self.local_path = get_path_from_frame(caller_frame, top_path)
        if top_path is None:
            top_path = self.local_path
            self.local_path = ''
        if package_path is None:
            package_path = self.local_path
        elif os.path.isdir(njoin(self.local_path, package_path)):
            package_path = njoin(self.local_path, package_path)
        if not os.path.isdir(package_path or '.'):
            raise ValueError("%r is not a directory" % (package_path,))
        self.top_path = top_path
        self.package_path = package_path
        self.path_in_package = os.path.join(*self.name.split('.'))
        self.list_keys = self._list_keys[:]
        self.dict_keys = self._dict_keys[:]
        for n in self.list_keys:
            v = copy.copy(attrs.get(n, []))
            setattr(self, n, as_list(v))
        for n in self.dict_keys:
            v = copy.copy(attrs.get(n, {}))
            setattr(self, n, v)
        known_keys = self.list_keys + self.dict_keys
        self.extra_keys = self._extra_keys[:]
        for n in attrs.keys():
            if n in known_keys:
                continue
            a = attrs[n]
            setattr(self, n, a)
            if isinstance(a, list):
                self.list_keys.append(n)
            elif isinstance(a, dict):
                self.dict_keys.append(n)
            else:
                self.extra_keys.append(n)
        if os.path.exists(njoin(package_path, '__init__.py')):
            self.packages.append(self.name)
            self.package_dir[self.name] = package_path
        self.options = dict(
            ignore_setup_xxx_py = False,
            assume_default_configuration = False,
            delegate_options_to_subpackages = False,
            quiet = False,
            )
        caller_instance = None
        for i in range(1, 3):
            try:
                f = get_frame(i)
            except ValueError:
                break
            try:
                caller_instance = eval('self', f.f_globals, f.f_locals)
                break
            except NameError:
                pass
        if isinstance(caller_instance, self.__class__):
            if caller_instance.options['delegate_options_to_subpackages']:
                self.set_options(**caller_instance.options)
        self.setup_name = setup_name
    def todict(self):
        self._optimize_data_files()
        d = {}
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        for n in known_keys:
            a = getattr(self, n)
            if a:
                d[n] = a
        return d
    def info(self, message):
        if not self.options['quiet']:
            print(message)
    def warn(self, message):
        sys.stderr.write('Warning: %s\n' % (message,))
    def set_options(self, **options):
        for key, value in options.items():
            if key in self.options:
                self.options[key] = value
            else:
                raise ValueError('Unknown option: '+key)
    def get_distribution(self):
        from numpy.distutils.core import get_distribution
        return get_distribution()
    def _wildcard_get_subpackage(self, subpackage_name,
                                 parent_name,
                                 caller_level = 1):
        l = subpackage_name.split('.')
        subpackage_path = njoin([self.local_path]+l)
        dirs = [_m for _m in sorted_glob(subpackage_path) if os.path.isdir(_m)]
        config_list = []
        for d in dirs:
            if not os.path.isfile(njoin(d, '__init__.py')):
                continue
            if 'build' in d.split(os.sep):
                continue
            n = '.'.join(d.split(os.sep)[-len(l):])
            c = self.get_subpackage(n,
                                    parent_name = parent_name,
                                    caller_level = caller_level+1)
            config_list.extend(c)
        return config_list
    def _get_configuration_from_setup_py(self, setup_py,
                                         subpackage_name,
                                         subpackage_path,
                                         parent_name,
                                         caller_level = 1):
        sys.path.insert(0, os.path.dirname(setup_py))
        try:
            setup_name = os.path.splitext(os.path.basename(setup_py))[0]
            n = dot_join(self.name, subpackage_name, setup_name)
            setup_module = npy_load_module('_'.join(n.split('.')),
                                           setup_py,
                                           ('.py', 'U', 1))
            if not hasattr(setup_module, 'configuration'):
                if not self.options['assume_default_configuration']:
                    self.warn('Assuming default configuration '                              '(%s does not define configuration())'                              % (setup_module))
                config = Configuration(subpackage_name, parent_name,
                                       self.top_path, subpackage_path,
                                       caller_level = caller_level + 1)
            else:
                pn = dot_join(*([parent_name] + subpackage_name.split('.')[:-1]))
                args = (pn,)
                if setup_module.configuration.__code__.co_argcount > 1:
                    args = args + (self.top_path,)
                config = setup_module.configuration(*args)
            if config.name!=dot_join(parent_name, subpackage_name):
                self.warn('Subpackage %r configuration returned as %r' %                          (dot_join(parent_name, subpackage_name), config.name))
        finally:
            del sys.path[0]
        return config
    def get_subpackage(self,subpackage_name,
                       subpackage_path=None,
                       parent_name=None,
                       caller_level = 1):
        if subpackage_name is None:
            if subpackage_path is None:
                raise ValueError(
                    "either subpackage_name or subpackage_path must be specified")
            subpackage_name = os.path.basename(subpackage_path)
        l = subpackage_name.split('.')
        if subpackage_path is None and '*' in subpackage_name:
            return self._wildcard_get_subpackage(subpackage_name,
                                                 parent_name,
                                                 caller_level = caller_level+1)
        assert '*' not in subpackage_name, repr((subpackage_name, subpackage_path, parent_name))
        if subpackage_path is None:
            subpackage_path = njoin([self.local_path] + l)
        else:
            subpackage_path = njoin([subpackage_path] + l[:-1])
            subpackage_path = self.paths([subpackage_path])[0]
        setup_py = njoin(subpackage_path, self.setup_name)
        if not self.options['ignore_setup_xxx_py']:
            if not os.path.isfile(setup_py):
                setup_py = njoin(subpackage_path,
                                 'setup_%s.py' % (subpackage_name))
        if not os.path.isfile(setup_py):
            if not self.options['assume_default_configuration']:
                self.warn('Assuming default configuration '                          '(%s/{setup_%s,setup}.py was not found)'                          % (os.path.dirname(setup_py), subpackage_name))
            config = Configuration(subpackage_name, parent_name,
                                   self.top_path, subpackage_path,
                                   caller_level = caller_level+1)
        else:
            config = self._get_configuration_from_setup_py(
                setup_py,
                subpackage_name,
                subpackage_path,
                parent_name,
                caller_level = caller_level + 1)
        if config:
            return [config]
        else:
            return []
    def add_subpackage(self,subpackage_name,
                       subpackage_path=None,
                       standalone = False):
        if standalone:
            parent_name = None
        else:
            parent_name = self.name
        config_list = self.get_subpackage(subpackage_name, subpackage_path,
                                          parent_name = parent_name,
                                          caller_level = 2)
        if not config_list:
            self.warn('No configuration returned, assuming unavailable.')
        for config in config_list:
            d = config
            if isinstance(config, Configuration):
                d = config.todict()
            assert isinstance(d, dict), repr(type(d))
            self.info('Appending %s configuration to %s'                      % (d.get('name'), self.name))
            self.dict_append(**d)
        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'                      ' it may be too late to add a subpackage '+ subpackage_name)
    def add_data_dir(self, data_path):
        if is_sequence(data_path):
            d, data_path = data_path
        else:
            d = None
        if is_sequence(data_path):
            [self.add_data_dir((d, p)) for p in data_path]
            return
        if not is_string(data_path):
            raise TypeError("not a string: %r" % (data_path,))
        if d is None:
            if os.path.isabs(data_path):
                return self.add_data_dir((os.path.basename(data_path), data_path))
            return self.add_data_dir((data_path, data_path))
        paths = self.paths(data_path, include_non_existing=False)
        if is_glob_pattern(data_path):
            if is_glob_pattern(d):
                pattern_list = allpath(d).split(os.sep)
                pattern_list.reverse()
                rl = list(range(len(pattern_list)-1)); rl.reverse()
                for i in rl:
                    if not pattern_list[i]:
                        del pattern_list[i]
                for path in paths:
                    if not os.path.isdir(path):
                        print('Not a directory, skipping', path)
                        continue
                    rpath = rel_path(path, self.local_path)
                    path_list = rpath.split(os.sep)
                    path_list.reverse()
                    target_list = []
                    i = 0
                    for s in pattern_list:
                        if is_glob_pattern(s):
                            if i>=len(path_list):
                                raise ValueError('cannot fill pattern %r with %r'                                      % (d, path))
                            target_list.append(path_list[i])
                        else:
                            assert s==path_list[i], repr((s, path_list[i], data_path, d, path, rpath))
                            target_list.append(s)
                        i += 1
                    if path_list[i:]:
                        self.warn('mismatch of pattern_list=%s and path_list=%s'                                  % (pattern_list, path_list))
                    target_list.reverse()
                    self.add_data_dir((os.sep.join(target_list), path))
            else:
                for path in paths:
                    self.add_data_dir((d, path))
            return
        assert not is_glob_pattern(d), repr(d)
        dist = self.get_distribution()
        if dist is not None and dist.data_files is not None:
            data_files = dist.data_files
        else:
            data_files = self.data_files
        for path in paths:
            for d1, f in list(general_source_directories_files(path)):
                target_path = os.path.join(self.path_in_package, d, d1)
                data_files.append((target_path, f))
    def _optimize_data_files(self):
        data_dict = {}
        for p, files in self.data_files:
            if p not in data_dict:
                data_dict[p] = set()
            for f in files:
                data_dict[p].add(f)
        self.data_files[:] = [(p, list(files)) for p, files in data_dict.items()]
    def add_data_files(self,*files):
        if len(files)>1:
            for f in files:
                self.add_data_files(f)
            return
        assert len(files)==1
        if is_sequence(files[0]):
            d, files = files[0]
        else:
            d = None
        if is_string(files):
            filepat = files
        elif is_sequence(files):
            if len(files)==1:
                filepat = files[0]
            else:
                for f in files:
                    self.add_data_files((d, f))
                return
        else:
            raise TypeError(repr(type(files)))
        if d is None:
            if hasattr(filepat, '__call__'):
                d = ''
            elif os.path.isabs(filepat):
                d = ''
            else:
                d = os.path.dirname(filepat)
            self.add_data_files((d, files))
            return
        paths = self.paths(filepat, include_non_existing=False)
        if is_glob_pattern(filepat):
            if is_glob_pattern(d):
                pattern_list = d.split(os.sep)
                pattern_list.reverse()
                for path in paths:
                    path_list = path.split(os.sep)
                    path_list.reverse()
                    path_list.pop()
                    target_list = []
                    i = 0
                    for s in pattern_list:
                        if is_glob_pattern(s):
                            target_list.append(path_list[i])
                            i += 1
                        else:
                            target_list.append(s)
                    target_list.reverse()
                    self.add_data_files((os.sep.join(target_list), path))
            else:
                self.add_data_files((d, paths))
            return
        assert not is_glob_pattern(d), repr((d, filepat))
        dist = self.get_distribution()
        if dist is not None and dist.data_files is not None:
            data_files = dist.data_files
        else:
            data_files = self.data_files
        data_files.append((os.path.join(self.path_in_package, d), paths))
    def add_define_macros(self, macros):
        dist = self.get_distribution()
        if dist is not None:
            if not hasattr(dist, 'define_macros'):
                dist.define_macros = []
            dist.define_macros.extend(macros)
        else:
            self.define_macros.extend(macros)
    def add_include_dirs(self,*paths):
        include_dirs = self.paths(paths)
        dist = self.get_distribution()
        if dist is not None:
            if dist.include_dirs is None:
                dist.include_dirs = []
            dist.include_dirs.extend(include_dirs)
        else:
            self.include_dirs.extend(include_dirs)
    def add_headers(self,*files):
        headers = []
        for path in files:
            if is_string(path):
                [headers.append((self.name, p)) for p in self.paths(path)]
            else:
                if not isinstance(path, (tuple, list)) or len(path) != 2:
                    raise TypeError(repr(path))
                [headers.append((path[0], p)) for p in self.paths(path[1])]
        dist = self.get_distribution()
        if dist is not None:
            if dist.headers is None:
                dist.headers = []
            dist.headers.extend(headers)
        else:
            self.headers.extend(headers)
    def paths(self,*paths,**kws):
        include_non_existing = kws.get('include_non_existing', True)
        return gpaths(paths,
                      local_path = self.local_path,
                      include_non_existing=include_non_existing)
    def _fix_paths_dict(self, kw):
        for k in kw.keys():
            v = kw[k]
            if k in ['sources', 'depends', 'include_dirs', 'library_dirs',
                     'module_dirs', 'extra_objects']:
                new_v = self.paths(v)
                kw[k] = new_v
    def add_extension(self,name,sources,**kw):
        ext_args = copy.copy(kw)
        ext_args['name'] = dot_join(self.name, name)
        ext_args['sources'] = sources
        if 'extra_info' in ext_args:
            extra_info = ext_args['extra_info']
            del ext_args['extra_info']
            if isinstance(extra_info, dict):
                extra_info = [extra_info]
            for info in extra_info:
                assert isinstance(info, dict), repr(info)
                dict_append(ext_args,**info)
        self._fix_paths_dict(ext_args)
        libraries = ext_args.get('libraries', [])
        libnames = []
        ext_args['libraries'] = []
        for libname in libraries:
            if isinstance(libname, tuple):
                self._fix_paths_dict(libname[1])
            if '@' in libname:
                lname, lpath = libname.split('@', 1)
                lpath = os.path.abspath(njoin(self.local_path, lpath))
                if os.path.isdir(lpath):
                    c = self.get_subpackage(None, lpath,
                                            caller_level = 2)
                    if isinstance(c, Configuration):
                        c = c.todict()
                    for l in [l[0] for l in c.get('libraries', [])]:
                        llname = l.split('__OF__', 1)[0]
                        if llname == lname:
                            c.pop('name', None)
                            dict_append(ext_args,**c)
                            break
                    continue
            libnames.append(libname)
        ext_args['libraries'] = libnames + ext_args['libraries']
        ext_args['define_macros'] =            self.define_macros + ext_args.get('define_macros', [])
        from numpy.distutils.core import Extension
        ext = Extension(**ext_args)
        self.ext_modules.append(ext)
        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'                      ' it may be too late to add an extension '+name)
        return ext
    def add_library(self,name,sources,**build_info):
        self._add_library(name, sources, None, build_info)
        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'                      ' it may be too late to add a library '+ name)
    def _add_library(self, name, sources, install_dir, build_info):
        build_info = copy.copy(build_info)
        build_info['sources'] = sources
        if not 'depends' in build_info:
            build_info['depends'] = []
        self._fix_paths_dict(build_info)
        self.libraries.append((name, build_info))
    def add_installed_library(self, name, sources, install_dir, build_info=None):
        if not build_info:
            build_info = {}
        install_dir = os.path.join(self.package_path, install_dir)
        self._add_library(name, sources, install_dir, build_info)
        self.installed_libraries.append(InstallableLib(name, build_info, install_dir))
    def add_npy_pkg_config(self, template, install_dir, subst_dict=None):
        if subst_dict is None:
            subst_dict = {}
        template = os.path.join(self.package_path, template)
        if self.name in self.installed_pkg_config:
            self.installed_pkg_config[self.name].append((template, install_dir,
                subst_dict))
        else:
            self.installed_pkg_config[self.name] = [(template, install_dir,
                subst_dict)]
    def add_scripts(self,*files):
        scripts = self.paths(files)
        dist = self.get_distribution()
        if dist is not None:
            if dist.scripts is None:
                dist.scripts = []
            dist.scripts.extend(scripts)
        else:
            self.scripts.extend(scripts)
    def dict_append(self,**dict):
        for key in self.list_keys:
            a = getattr(self, key)
            a.extend(dict.get(key, []))
        for key in self.dict_keys:
            a = getattr(self, key)
            a.update(dict.get(key, {}))
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        for key in dict.keys():
            if key not in known_keys:
                a = getattr(self, key, None)
                if a and a==dict[key]: continue
                self.warn('Inheriting attribute %r=%r from %r'                          % (key, dict[key], dict.get('name', '?')))
                setattr(self, key, dict[key])
                self.extra_keys.append(key)
            elif key in self.extra_keys:
                self.info('Ignoring attempt to set %r (from %r to %r)'                          % (key, getattr(self, key), dict[key]))
            elif key in known_keys:
                pass
            else:
                raise ValueError("Don't know about key=%r" % (key))
    def __str__(self):
        from pprint import pformat
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        s = '<'+5*'-' + '\n'
        s += 'Configuration of '+self.name+':\n'
        known_keys.sort()
        for k in known_keys:
            a = getattr(self, k, None)
            if a:
                s += '%s = %s\n' % (k, pformat(a))
        s += 5*'-' + '>'
        return s
    def get_config_cmd(self):
        cmd = get_cmd('config')
        cmd.ensure_finalized()
        cmd.dump_source = 0
        cmd.noisy = 0
        old_path = os.environ.get('PATH')
        if old_path:
            path = os.pathsep.join(['.', old_path])
            os.environ['PATH'] = path
        return cmd
    def get_build_temp_dir(self):
        cmd = get_cmd('build')
        cmd.ensure_finalized()
        return cmd.build_temp
    def have_f77c(self):
        simple_fortran_subroutine = '''
        subroutine simple
        end
        '''
        config_cmd = self.get_config_cmd()
        flag = config_cmd.try_compile(simple_fortran_subroutine, lang='f77')
        return flag
    def have_f90c(self):
        simple_fortran_subroutine = '''
        subroutine simple
        end
        '''
        config_cmd = self.get_config_cmd()
        flag = config_cmd.try_compile(simple_fortran_subroutine, lang='f90')
        return flag
    def append_to(self, extlib):
        if is_sequence(extlib):
            lib_name, build_info = extlib
            dict_append(build_info,
                        libraries=self.libraries,
                        include_dirs=self.include_dirs)
        else:
            from numpy.distutils.core import Extension
            assert isinstance(extlib, Extension), repr(extlib)
            extlib.libraries.extend(self.libraries)
            extlib.include_dirs.extend(self.include_dirs)
    def _get_svn_revision(self, path):
        try:
            output = subprocess.check_output(['svnversion'], cwd=path)
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            m = re.match(rb'(?P<revision>\d+)', output)
            if m:
                return int(m.group('revision'))
        if sys.platform=='win32' and os.environ.get('SVN_ASP_DOT_NET_HACK', None):
            entries = njoin(path, '_svn', 'entries')
        else:
            entries = njoin(path, '.svn', 'entries')
        if os.path.isfile(entries):
            with open(entries) as f:
                fstr = f.read()
            if fstr[:5] == '<?xml':
                m = re.search(r'revision="(?P<revision>\d+)"', fstr)
                if m:
                    return int(m.group('revision'))
            else:
                m = re.search(r'dir[\n\r]+(?P<revision>\d+)', fstr)
                if m:
                    return int(m.group('revision'))
        return None
    def _get_hg_revision(self, path):
        try:
            output = subprocess.check_output(
                ['hg', 'identify', '--num'], cwd=path)
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            m = re.match(rb'(?P<revision>\d+)', output)
            if m:
                return int(m.group('revision'))
        branch_fn = njoin(path, '.hg', 'branch')
        branch_cache_fn = njoin(path, '.hg', 'branch.cache')
        if os.path.isfile(branch_fn):
            branch0 = None
            with open(branch_fn) as f:
                revision0 = f.read().strip()
            branch_map = {}
            with open(branch_cache_fn, 'r') as f:
                for line in f:
                    branch1, revision1  = line.split()[:2]
                    if revision1==revision0:
                        branch0 = branch1
                    try:
                        revision1 = int(revision1)
                    except ValueError:
                        continue
                    branch_map[branch1] = revision1
            return branch_map.get(branch0)
        return None
    def get_version(self, version_file=None, version_variable=None):
        version = getattr(self, 'version', None)
        if version is not None:
            return version
        if version_file is None:
            files = ['__version__.py',
                     self.name.split('.')[-1]+'_version.py',
                     'version.py',
                     '__svn_version__.py',
                     '__hg_version__.py']
        else:
            files = [version_file]
        if version_variable is None:
            version_vars = ['version',
                            '__version__',
                            self.name.split('.')[-1]+'_version']
        else:
            version_vars = [version_variable]
        for f in files:
            fn = njoin(self.local_path, f)
            if os.path.isfile(fn):
                info = ('.py', 'U', 1)
                name = os.path.splitext(os.path.basename(fn))[0]
                n = dot_join(self.name, name)
                try:
                    version_module = npy_load_module('_'.join(n.split('.')),
                                                     fn, info)
                except ImportError as e:
                    self.warn(str(e))
                    version_module = None
                if version_module is None:
                    continue
                for a in version_vars:
                    version = getattr(version_module, a, None)
                    if version is not None:
                        break
                if version is not None:
                    break
        if version is not None:
            self.version = version
            return version
        revision = self._get_svn_revision(self.local_path)
        if revision is None:
            revision = self._get_hg_revision(self.local_path)
        if revision is not None:
            version = str(revision)
            self.version = version
        return version
    def make_svn_version_py(self, delete=True):
        target = njoin(self.local_path, '__svn_version__.py')
        revision = self._get_svn_revision(self.local_path)
        if os.path.isfile(target) or revision is None:
            return
        else:
            def generate_svn_version_py():
                if not os.path.isfile(target):
                    version = str(revision)
                    self.info('Creating %s (version=%r)' % (target, version))
                    with open(target, 'w') as f:
                        f.write('version = %r\n' % (version))
                def rm_file(f=target,p=self.info):
                    if delete:
                        try: os.remove(f); p('removed '+f)
                        except OSError: pass
                        try: os.remove(f+'c'); p('removed '+f+'c')
                        except OSError: pass
                atexit.register(rm_file)
                return target
            self.add_data_files(('', generate_svn_version_py()))
    def make_hg_version_py(self, delete=True):
        target = njoin(self.local_path, '__hg_version__.py')
        revision = self._get_hg_revision(self.local_path)
        if os.path.isfile(target) or revision is None:
            return
        else:
            def generate_hg_version_py():
                if not os.path.isfile(target):
                    version = str(revision)
                    self.info('Creating %s (version=%r)' % (target, version))
                    with open(target, 'w') as f:
                        f.write('version = %r\n' % (version))
                def rm_file(f=target,p=self.info):
                    if delete:
                        try: os.remove(f); p('removed '+f)
                        except OSError: pass
                        try: os.remove(f+'c'); p('removed '+f+'c')
                        except OSError: pass
                atexit.register(rm_file)
                return target
            self.add_data_files(('', generate_hg_version_py()))
    def make_config_py(self,name='__config__'):
        self.py_modules.append((self.name, name, generate_config_py))
    def get_info(self,*names):
        from .system_info import get_info, dict_append
        info_dict = {}
        for a in names:
            dict_append(info_dict,**get_info(a))
        return info_dict
def get_cmd(cmdname, _cache={}):
    if cmdname not in _cache:
        import distutils.core
        dist = distutils.core._setup_distribution
        if dist is None:
            from distutils.errors import DistutilsInternalError
            raise DistutilsInternalError(
                  'setup distribution instance not initialized')
        cmd = dist.get_command_obj(cmdname)
        _cache[cmdname] = cmd
    return _cache[cmdname]
def get_numpy_include_dirs():
    include_dirs = Configuration.numpy_include_dirs[:]
    if not include_dirs:
        import numpy
        include_dirs = [ numpy.get_include() ]
    return include_dirs
def get_npy_pkg_dir():
    d = os.environ.get('NPY_PKG_CONFIG_PATH')
    if d is not None:
        return d
    spec = importlib.util.find_spec('numpy')
    d = os.path.join(os.path.dirname(spec.origin),
            'core', 'lib', 'npy-pkg-config')
    return d
def get_pkg_info(pkgname, dirs=None):
    from numpy.distutils.npy_pkg_config import read_config
    if dirs:
        dirs.append(get_npy_pkg_dir())
    else:
        dirs = [get_npy_pkg_dir()]
    return read_config(pkgname, dirs)
def get_info(pkgname, dirs=None):
    from numpy.distutils.npy_pkg_config import parse_flags
    pkg_info = get_pkg_info(pkgname, dirs)
    info = parse_flags(pkg_info.cflags())
    for k, v in parse_flags(pkg_info.libs()).items():
        info[k].extend(v)
    info['define_macros'] = info['macros']
    del info['macros']
    del info['ignored']
    return info
def is_bootstrapping():
    import builtins
    try:
        builtins.__NUMPY_SETUP__
        return True
    except AttributeError:
        return False
def default_config_dict(name = None, parent_name = None, local_path=None):
    import warnings
    warnings.warn('Use Configuration(%r,%r,top_path=%r) instead of '                  'deprecated default_config_dict(%r,%r,%r)'
                  % (name, parent_name, local_path,
                     name, parent_name, local_path,
                     ), stacklevel=2)
    c = Configuration(name, parent_name, local_path)
    return c.todict()
def dict_append(d, **kws):
    for k, v in kws.items():
        if k in d:
            ov = d[k]
            if isinstance(ov, str):
                d[k] = v
            else:
                d[k].extend(v)
        else:
            d[k] = v
def appendpath(prefix, path):
    if os.path.sep != '/':
        prefix = prefix.replace('/', os.path.sep)
        path = path.replace('/', os.path.sep)
    drive = ''
    if os.path.isabs(path):
        drive = os.path.splitdrive(prefix)[0]
        absprefix = os.path.splitdrive(os.path.abspath(prefix))[1]
        pathdrive, path = os.path.splitdrive(path)
        d = os.path.commonprefix([absprefix, path])
        if os.path.join(absprefix[:len(d)], absprefix[len(d):]) != absprefix           or os.path.join(path[:len(d)], path[len(d):]) != path:
            d = os.path.dirname(d)
        subpath = path[len(d):]
        if os.path.isabs(subpath):
            subpath = subpath[1:]
    else:
        subpath = path
    return os.path.normpath(njoin(drive + prefix, subpath))
def generate_config_py(target):
    from numpy.distutils.system_info import system_info
    from distutils.dir_util import mkpath
    mkpath(os.path.dirname(target))
    with open(target, 'w') as f:
        f.write('__all__ = ["get_info","show"]\n\n')
        f.write(textwrap.dedent("""
            import os
            import sys
            extra_dll_dir = os.path.join(os.path.dirname(__file__), '.libs')
            if sys.platform == 'win32' and os.path.isdir(extra_dll_dir):
                if sys.version_info >= (3, 8):
                    os.add_dll_directory(extra_dll_dir)
                else:
                    os.environ.setdefault('PATH', '')
                    os.environ['PATH'] += os.pathsep + extra_dll_dir
            """))
        for k, i in system_info.saved_results.items():
            f.write('%s=%r\n' % (k, i))
        f.write(textwrap.dedent(r'''
            def get_info(name):
                g = globals()
                return g.get(name, g.get(name + "_info", {}))
            def show():
                """
                Show libraries in the system on which NumPy was built.
                Print information about various resources (libraries, library
                directories, include directories, etc.) in the system on which
                NumPy was built.
                See Also
                --------
                get_include : Returns the directory containing NumPy C
                              header files.
                Notes
                -----
                Classes specifying the information to be printed are defined
                in the `numpy.distutils.system_info` module.
                Information may include:
                * ``language``: language used to write the libraries (mostly
                  C or f77)
                * ``libraries``: names of libraries found in the system
                * ``library_dirs``: directories containing the libraries
                * ``include_dirs``: directories containing library header files
                * ``src_dirs``: directories containing library source files
                * ``define_macros``: preprocessor macros used by
                  ``distutils.setup``
                Examples
                --------
                >>> import numpy as np
                >>> np.show_config()
                blas_opt_info:
                    language = c
                    define_macros = [('HAVE_CBLAS', None)]
                    libraries = ['openblas', 'openblas']
                    library_dirs = ['/usr/local/lib']
                """
                for name,info_dict in globals().items():
                    if name[0] == "_" or type(info_dict) is not type({}): continue
                    print(name + ":")
                    if not info_dict:
                        print("  NOT AVAILABLE")
                    for k,v in info_dict.items():
                        v = str(v)
                        if k == "sources" and len(v) > 200:
                            v = v[:60] + " ...\n... " + v[-60:]
                        print("    %s = %s" % (k,v))
                    '''))
    return target
def msvc_version(compiler):
    if not compiler.compiler_type == "msvc":
        raise ValueError("Compiler instance is not msvc (%s)"                         % compiler.compiler_type)
    return compiler._MSVCCompiler__version
def get_build_architecture():
    from distutils.msvccompiler import get_build_architecture
    return get_build_architecture()
