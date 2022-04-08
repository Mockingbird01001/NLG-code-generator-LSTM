
import codecs
import os
import re
import sys
from os.path import pardir, realpath
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
__all__ = [
    'get_config_h_filename',
    'get_config_var',
    'get_config_vars',
    'get_makefile_filename',
    'get_path',
    'get_path_names',
    'get_paths',
    'get_platform',
    'get_python_version',
    'get_scheme_names',
    'parse_config_h',
]
def _safe_realpath(path):
    try:
        return realpath(path)
    except OSError:
        return path
if sys.executable:
    _PROJECT_BASE = os.path.dirname(_safe_realpath(sys.executable))
else:
    _PROJECT_BASE = _safe_realpath(os.getcwd())
if os.name == "nt" and "pcbuild" in _PROJECT_BASE[-8:].lower():
    _PROJECT_BASE = _safe_realpath(os.path.join(_PROJECT_BASE, pardir))
if os.name == "nt" and "\\pc\\v" in _PROJECT_BASE[-10:].lower():
    _PROJECT_BASE = _safe_realpath(os.path.join(_PROJECT_BASE, pardir, pardir))
if os.name == "nt" and "\\pcbuild\\amd64" in _PROJECT_BASE[-14:].lower():
    _PROJECT_BASE = _safe_realpath(os.path.join(_PROJECT_BASE, pardir, pardir))
def is_python_build():
    for fn in ("Setup.dist", "Setup.local"):
        if os.path.isfile(os.path.join(_PROJECT_BASE, "Modules", fn)):
            return True
    return False
_PYTHON_BUILD = is_python_build()
_cfg_read = False
def _ensure_cfg_read():
    global _cfg_read
    if not _cfg_read:
        from ..resources import finder
        backport_package = __name__.rsplit('.', 1)[0]
        _finder = finder(backport_package)
        _cfgfile = _finder.find('sysconfig.cfg')
        assert _cfgfile, 'sysconfig.cfg exists'
        with _cfgfile.as_stream() as s:
            _SCHEMES.readfp(s)
        if _PYTHON_BUILD:
            for scheme in ('posix_prefix', 'posix_home'):
                _SCHEMES.set(scheme, 'include', '{srcdir}/Include')
                _SCHEMES.set(scheme, 'platinclude', '{projectbase}/.')
        _cfg_read = True
_SCHEMES = configparser.RawConfigParser()
_VAR_REPL = re.compile(r'\{([^{]*?)\}')
def _expand_globals(config):
    _ensure_cfg_read()
    if config.has_section('globals'):
        globals = config.items('globals')
    else:
        globals = tuple()
    sections = config.sections()
    for section in sections:
        if section == 'globals':
            continue
        for option, value in globals:
            if config.has_option(section, option):
                continue
            config.set(section, option, value)
    config.remove_section('globals')
    for section in config.sections():
        variables = dict(config.items(section))
        def _replacer(matchobj):
            name = matchobj.group(1)
            if name in variables:
                return variables[name]
            return matchobj.group(0)
        for option, value in config.items(section):
            config.set(section, option, _VAR_REPL.sub(_replacer, value))
_PY_VERSION = '%s.%s.%s' % sys.version_info[:3]
_PY_VERSION_SHORT = '%s.%s' % sys.version_info[:2]
_PY_VERSION_SHORT_NO_DOT = '%s%s' % sys.version_info[:2]
_PREFIX = os.path.normpath(sys.prefix)
_EXEC_PREFIX = os.path.normpath(sys.exec_prefix)
_CONFIG_VARS = None
_USER_BASE = None
def _subst_vars(path, local_vars):
    def _replacer(matchobj):
        name = matchobj.group(1)
        if name in local_vars:
            return local_vars[name]
        elif name in os.environ:
            return os.environ[name]
        return matchobj.group(0)
    return _VAR_REPL.sub(_replacer, path)
def _extend_dict(target_dict, other_dict):
    target_keys = target_dict.keys()
    for key, value in other_dict.items():
        if key in target_keys:
            continue
        target_dict[key] = value
def _expand_vars(scheme, vars):
    res = {}
    if vars is None:
        vars = {}
    _extend_dict(vars, get_config_vars())
    for key, value in _SCHEMES.items(scheme):
        if os.name in ('posix', 'nt'):
            value = os.path.expanduser(value)
        res[key] = os.path.normpath(_subst_vars(value, vars))
    return res
def format_value(value, vars):
    def _replacer(matchobj):
        name = matchobj.group(1)
        if name in vars:
            return vars[name]
        return matchobj.group(0)
    return _VAR_REPL.sub(_replacer, value)
def _get_default_scheme():
    if os.name == 'posix':
        return 'posix_prefix'
    return os.name
def _getuserbase():
    env_base = os.environ.get("PYTHONUSERBASE", None)
    def joinuser(*args):
        return os.path.expanduser(os.path.join(*args))
    if os.name == "nt":
        base = os.environ.get("APPDATA") or "~"
        if env_base:
            return env_base
        else:
            return joinuser(base, "Python")
    if sys.platform == "darwin":
        framework = get_config_var("PYTHONFRAMEWORK")
        if framework:
            if env_base:
                return env_base
            else:
                return joinuser("~", "Library", framework, "%d.%d" %
                                sys.version_info[:2])
    if env_base:
        return env_base
    else:
        return joinuser("~", ".local")
def _parse_makefile(filename, vars=None):
    _variable_rx = re.compile(r"([a-zA-Z][a-zA-Z0-9_]+)\s*=\s*(.*)")
    _findvar1_rx = re.compile(r"\$\(([A-Za-z][A-Za-z0-9_]*)\)")
    _findvar2_rx = re.compile(r"\${([A-Za-z][A-Za-z0-9_]*)}")
    if vars is None:
        vars = {}
    done = {}
    notdone = {}
    with codecs.open(filename, encoding='utf-8', errors="surrogateescape") as f:
        lines = f.readlines()
    for line in lines:
            continue
        m = _variable_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            v = v.strip()
            tmpv = v.replace('$$', '')
            if "$" in tmpv:
                notdone[n] = v
            else:
                try:
                    v = int(v)
                except ValueError:
                    done[n] = v.replace('$$', '$')
                else:
                    done[n] = v
    variables = list(notdone.keys())
    renamed_variables = ('CFLAGS', 'LDFLAGS', 'CPPFLAGS')
    while len(variables) > 0:
        for name in tuple(variables):
            value = notdone[name]
            m = _findvar1_rx.search(value) or _findvar2_rx.search(value)
            if m is not None:
                n = m.group(1)
                found = True
                if n in done:
                    item = str(done[n])
                elif n in notdone:
                    found = False
                elif n in os.environ:
                    item = os.environ[n]
                elif n in renamed_variables:
                    if (name.startswith('PY_') and
                        name[3:] in renamed_variables):
                        item = ""
                    elif 'PY_' + n in notdone:
                        found = False
                    else:
                        item = str(done['PY_' + n])
                else:
                    done[n] = item = ""
                if found:
                    after = value[m.end():]
                    value = value[:m.start()] + item + after
                    if "$" in after:
                        notdone[name] = value
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            done[name] = value.strip()
                        else:
                            done[name] = value
                        variables.remove(name)
                        if (name.startswith('PY_') and
                            name[3:] in renamed_variables):
                            name = name[3:]
                            if name not in done:
                                done[name] = value
            else:
                done[name] = value
                variables.remove(name)
    for k, v in done.items():
        if isinstance(v, str):
            done[k] = v.strip()
    vars.update(done)
    return vars
def get_makefile_filename():
    if _PYTHON_BUILD:
        return os.path.join(_PROJECT_BASE, "Makefile")
    if hasattr(sys, 'abiflags'):
        config_dir_name = 'config-%s%s' % (_PY_VERSION_SHORT, sys.abiflags)
    else:
        config_dir_name = 'config'
    return os.path.join(get_path('stdlib'), config_dir_name, 'Makefile')
def _init_posix(vars):
    makefile = get_makefile_filename()
    try:
        _parse_makefile(makefile, vars)
    except IOError as e:
        msg = "invalid Python installation: unable to open %s" % makefile
        if hasattr(e, "strerror"):
            msg = msg + " (%s)" % e.strerror
        raise IOError(msg)
    config_h = get_config_h_filename()
    try:
        with open(config_h) as f:
            parse_config_h(f, vars)
    except IOError as e:
        msg = "invalid Python installation: unable to open %s" % config_h
        if hasattr(e, "strerror"):
            msg = msg + " (%s)" % e.strerror
        raise IOError(msg)
    if _PYTHON_BUILD:
        vars['LDSHARED'] = vars['BLDSHARED']
def _init_non_posix(vars):
    vars['LIBDEST'] = get_path('stdlib')
    vars['BINLIBDEST'] = get_path('platstdlib')
    vars['INCLUDEPY'] = get_path('include')
    vars['SO'] = '.pyd'
    vars['EXE'] = '.exe'
    vars['VERSION'] = _PY_VERSION_SHORT_NO_DOT
    vars['BINDIR'] = os.path.dirname(_safe_realpath(sys.executable))
def parse_config_h(fp, vars=None):
    if vars is None:
        vars = {}
    while True:
        line = fp.readline()
        if not line:
            break
        m = define_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            try:
                v = int(v)
            except ValueError:
                pass
            vars[n] = v
        else:
            m = undef_rx.match(line)
            if m:
                vars[m.group(1)] = 0
    return vars
def get_config_h_filename():
    if _PYTHON_BUILD:
        if os.name == "nt":
            inc_dir = os.path.join(_PROJECT_BASE, "PC")
        else:
            inc_dir = _PROJECT_BASE
    else:
        inc_dir = get_path('platinclude')
    return os.path.join(inc_dir, 'pyconfig.h')
def get_scheme_names():
    return tuple(sorted(_SCHEMES.sections()))
def get_path_names():
    return _SCHEMES.options('posix_prefix')
def get_paths(scheme=_get_default_scheme(), vars=None, expand=True):
    _ensure_cfg_read()
    if expand:
        return _expand_vars(scheme, vars)
    else:
        return dict(_SCHEMES.items(scheme))
def get_path(name, scheme=_get_default_scheme(), vars=None, expand=True):
    return get_paths(scheme, vars, expand)[name]
def get_config_vars(*args):
    global _CONFIG_VARS
    if _CONFIG_VARS is None:
        _CONFIG_VARS = {}
        _CONFIG_VARS['prefix'] = _PREFIX
        _CONFIG_VARS['exec_prefix'] = _EXEC_PREFIX
        _CONFIG_VARS['py_version'] = _PY_VERSION
        _CONFIG_VARS['py_version_short'] = _PY_VERSION_SHORT
        _CONFIG_VARS['py_version_nodot'] = _PY_VERSION[0] + _PY_VERSION[2]
        _CONFIG_VARS['base'] = _PREFIX
        _CONFIG_VARS['platbase'] = _EXEC_PREFIX
        _CONFIG_VARS['projectbase'] = _PROJECT_BASE
        try:
            _CONFIG_VARS['abiflags'] = sys.abiflags
        except AttributeError:
            _CONFIG_VARS['abiflags'] = ''
        if os.name in ('nt', 'os2'):
            _init_non_posix(_CONFIG_VARS)
        if os.name == 'posix':
            _init_posix(_CONFIG_VARS)
        if sys.version >= '2.6':
            _CONFIG_VARS['userbase'] = _getuserbase()
        if 'srcdir' not in _CONFIG_VARS:
            _CONFIG_VARS['srcdir'] = _PROJECT_BASE
        else:
            _CONFIG_VARS['srcdir'] = _safe_realpath(_CONFIG_VARS['srcdir'])
        if _PYTHON_BUILD and os.name == "posix":
            base = _PROJECT_BASE
            try:
                cwd = os.getcwd()
            except OSError:
                cwd = None
            if (not os.path.isabs(_CONFIG_VARS['srcdir']) and
                base != cwd):
                srcdir = os.path.join(base, _CONFIG_VARS['srcdir'])
                _CONFIG_VARS['srcdir'] = os.path.normpath(srcdir)
        if sys.platform == 'darwin':
            kernel_version = os.uname()[2]
            major_version = int(kernel_version.split('.')[0])
            if major_version < 8:
                for key in ('LDFLAGS', 'BASECFLAGS',
                        'CFLAGS', 'PY_CFLAGS', 'BLDSHARED'):
                    flags = _CONFIG_VARS[key]
                    flags = re.sub(r'-arch\s+\w+\s', ' ', flags)
                    flags = re.sub('-isysroot [^ \t]*', ' ', flags)
                    _CONFIG_VARS[key] = flags
            else:
                if 'ARCHFLAGS' in os.environ:
                    arch = os.environ['ARCHFLAGS']
                    for key in ('LDFLAGS', 'BASECFLAGS',
                        'CFLAGS', 'PY_CFLAGS', 'BLDSHARED'):
                        flags = _CONFIG_VARS[key]
                        flags = re.sub(r'-arch\s+\w+\s', ' ', flags)
                        flags = flags + ' ' + arch
                        _CONFIG_VARS[key] = flags
                CFLAGS = _CONFIG_VARS.get('CFLAGS', '')
                m = re.search(r'-isysroot\s+(\S+)', CFLAGS)
                if m is not None:
                    sdk = m.group(1)
                    if not os.path.exists(sdk):
                        for key in ('LDFLAGS', 'BASECFLAGS',
                            'CFLAGS', 'PY_CFLAGS', 'BLDSHARED'):
                            flags = _CONFIG_VARS[key]
                            flags = re.sub(r'-isysroot\s+\S+(\s|$)', ' ', flags)
                            _CONFIG_VARS[key] = flags
    if args:
        vals = []
        for name in args:
            vals.append(_CONFIG_VARS.get(name))
        return vals
    else:
        return _CONFIG_VARS
def get_config_var(name):
    return get_config_vars().get(name)
def get_platform():
    if os.name == 'nt':
        prefix = " bit ("
        i = sys.version.find(prefix)
        if i == -1:
            return sys.platform
        j = sys.version.find(")", i)
        look = sys.version[i+len(prefix):j].lower()
        if look == 'amd64':
            return 'win-amd64'
        if look == 'itanium':
            return 'win-ia64'
        return sys.platform
    if os.name != "posix" or not hasattr(os, 'uname'):
        return sys.platform
    osname, host, release, version, machine = os.uname()
    osname = osname.lower().replace('/', '')
    machine = machine.replace(' ', '_')
    machine = machine.replace('/', '-')
    if osname[:5] == "linux":
        return  "%s-%s" % (osname, machine)
    elif osname[:5] == "sunos":
        if release[0] >= "5":
            osname = "solaris"
            release = "%d.%s" % (int(release[0]) - 3, release[2:])
    elif osname[:4] == "irix":
        return "%s-%s" % (osname, release)
    elif osname[:3] == "aix":
        return "%s-%s.%s" % (osname, version, release)
    elif osname[:6] == "cygwin":
        osname = "cygwin"
        rel_re = re.compile(r'[\d.]+')
        m = rel_re.match(release)
        if m:
            release = m.group()
    elif osname[:6] == "darwin":
        cfgvars = get_config_vars()
        macver = cfgvars.get('MACOSX_DEPLOYMENT_TARGET')
        if True:
            macrelease = macver
            try:
                f = open('/System/Library/CoreServices/SystemVersion.plist')
            except IOError:
                pass
            else:
                try:
                    m = re.search(r'<key>ProductUserVisibleVersion</key>\s*'
                                  r'<string>(.*?)</string>', f.read())
                finally:
                    f.close()
                if m is not None:
                    macrelease = '.'.join(m.group(1).split('.')[:2])
        if not macver:
            macver = macrelease
        if macver:
            release = macver
            osname = "macosx"
            if ((macrelease + '.') >= '10.4.' and
                '-arch' in get_config_vars().get('CFLAGS', '').strip()):
                machine = 'fat'
                cflags = get_config_vars().get('CFLAGS')
                archs = re.findall(r'-arch\s+(\S+)', cflags)
                archs = tuple(sorted(set(archs)))
                if len(archs) == 1:
                    machine = archs[0]
                elif archs == ('i386', 'ppc'):
                    machine = 'fat'
                elif archs == ('i386', 'x86_64'):
                    machine = 'intel'
                elif archs == ('i386', 'ppc', 'x86_64'):
                    machine = 'fat3'
                elif archs == ('ppc64', 'x86_64'):
                    machine = 'fat64'
                elif archs == ('i386', 'ppc', 'ppc64', 'x86_64'):
                    machine = 'universal'
                else:
                    raise ValueError(
                       "Don't know machine value for archs=%r" % (archs,))
            elif machine == 'i386':
                if sys.maxsize >= 2**32:
                    machine = 'x86_64'
            elif machine in ('PowerPC', 'Power_Macintosh'):
                if sys.maxsize >= 2**32:
                    machine = 'ppc64'
                else:
                    machine = 'ppc'
    return "%s-%s-%s" % (osname, release, machine)
def get_python_version():
    return _PY_VERSION_SHORT
def _print_dict(title, data):
    for index, (key, value) in enumerate(sorted(data.items())):
        if index == 0:
            print('%s: ' % (title))
        print('\t%s = "%s"' % (key, value))
def _main():
    print('Platform: "%s"' % get_platform())
    print('Python version: "%s"' % get_python_version())
    print('Current installation scheme: "%s"' % _get_default_scheme())
    print()
    _print_dict('Paths', get_paths())
    print()
    _print_dict('Variables', get_config_vars())
if __name__ == '__main__':
    _main()
