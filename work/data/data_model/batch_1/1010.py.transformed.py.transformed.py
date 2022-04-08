
import sys
import os
try:
    import __builtin__ as builtins
except ImportError:
    import builtins
try:
    set
except NameError:
    from sets import Set as set
PREFIXES = [sys.prefix, sys.exec_prefix]
ENABLE_USER_SITE = None
USER_SITE = None
USER_BASE = None
_is_64bit = (getattr(sys, 'maxsize', None) or getattr(sys, 'maxint')) > 2**32
_is_pypy = hasattr(sys, 'pypy_version_info')
_is_jython = sys.platform[:4] == 'java'
if _is_jython:
    ModuleType = type(os)
def makepath(*paths):
    dir = os.path.join(*paths)
    if _is_jython and (dir == '__classpath__' or
                       dir.startswith('__pyclasspath__')):
        return dir, dir
    dir = os.path.abspath(dir)
    return dir, os.path.normcase(dir)
def abs__file__():
    for m in sys.modules.values():
        if ((_is_jython and not isinstance(m, ModuleType)) or
            hasattr(m, '__loader__')):
            continue
        f = getattr(m, '__file__', None)
        if f is None:
            continue
        m.__file__ = os.path.abspath(f)
def removeduppaths():
    L = []
    known_paths = set()
    for dir in sys.path:
        dir, dircase = makepath(dir)
        if not dircase in known_paths:
            L.append(dir)
            known_paths.add(dircase)
    sys.path[:] = L
    return known_paths
def addbuilddir():
    from distutils.util import get_platform
    s = "build/lib.%s-%.3s" % (get_platform(), sys.version)
    if hasattr(sys, 'gettotalrefcount'):
        s += '-pydebug'
    s = os.path.join(os.path.dirname(sys.path[-1]), s)
    sys.path.append(s)
def _init_pathinfo():
    d = set()
    for dir in sys.path:
        try:
            if os.path.isdir(dir):
                dir, dircase = makepath(dir)
                d.add(dircase)
        except TypeError:
            continue
    return d
def addpackage(sitedir, name, known_paths):
    if known_paths is None:
        _init_pathinfo()
        reset = 1
    else:
        reset = 0
    fullname = os.path.join(sitedir, name)
    try:
        f = open(fullname, "rU")
    except IOError:
        return
    try:
        for line in f:
                continue
            if line.startswith("import"):
                exec(line)
                continue
            line = line.rstrip()
            dir, dircase = makepath(sitedir, line)
            if not dircase in known_paths and os.path.exists(dir):
                sys.path.append(dir)
                known_paths.add(dircase)
    finally:
        f.close()
    if reset:
        known_paths = None
    return known_paths
def addsitedir(sitedir, known_paths=None):
    if known_paths is None:
        known_paths = _init_pathinfo()
        reset = 1
    else:
        reset = 0
    sitedir, sitedircase = makepath(sitedir)
    if not sitedircase in known_paths:
        sys.path.append(sitedir)
    try:
        names = os.listdir(sitedir)
    except os.error:
        return
    names.sort()
    for name in names:
        if name.endswith(os.extsep + "pth"):
            addpackage(sitedir, name, known_paths)
    if reset:
        known_paths = None
    return known_paths
def addsitepackages(known_paths, sys_prefix=sys.prefix, exec_prefix=sys.exec_prefix):
    prefixes = [os.path.join(sys_prefix, "local"), sys_prefix]
    if exec_prefix != sys_prefix:
        prefixes.append(os.path.join(exec_prefix, "local"))
    for prefix in prefixes:
        if prefix:
            if sys.platform in ('os2emx', 'riscos') or _is_jython:
                sitedirs = [os.path.join(prefix, "Lib", "site-packages")]
            elif _is_pypy:
                sitedirs = [os.path.join(prefix, 'site-packages')]
            elif sys.platform == 'darwin' and prefix == sys_prefix:
                if prefix.startswith("/System/Library/Frameworks/"):
                    sitedirs = [os.path.join("/Library/Python", sys.version[:3], "site-packages"),
                                os.path.join(prefix, "Extras", "lib", "python")]
                else:
                    sitedirs = [os.path.join(prefix, "lib",
                                             "python" + sys.version[:3], "site-packages")]
            elif os.sep == '/':
                sitedirs = [os.path.join(prefix,
                                         "lib",
                                         "python" + sys.version[:3],
                                         "site-packages"),
                            os.path.join(prefix, "lib", "site-python"),
                            os.path.join(prefix, "python" + sys.version[:3], "lib-dynload")]
                lib64_dir = os.path.join(prefix, "lib64", "python" + sys.version[:3], "site-packages")
                if (os.path.exists(lib64_dir) and
                    os.path.realpath(lib64_dir) not in [os.path.realpath(p) for p in sitedirs]):
                    if _is_64bit:
                        sitedirs.insert(0, lib64_dir)
                    else:
                        sitedirs.append(lib64_dir)
                try:
                    sys.getobjects
                    sitedirs.insert(0, os.path.join(sitedirs[0], 'debug'))
                except AttributeError:
                    pass
                sitedirs.append(os.path.join(prefix, "local/lib",
                                             "python" + sys.version[:3],
                                             "dist-packages"))
                if sys.version[0] == '2':
                    sitedirs.append(os.path.join(prefix, "lib",
                                                 "python" + sys.version[:3],
                                                 "dist-packages"))
                else:
                    sitedirs.append(os.path.join(prefix, "lib",
                                                 "python" + sys.version[0],
                                                 "dist-packages"))
                sitedirs.append(os.path.join(prefix, "lib", "dist-python"))
            else:
                sitedirs = [prefix, os.path.join(prefix, "lib", "site-packages")]
            if sys.platform == 'darwin':
                if 'Python.framework' in prefix:
                    home = os.environ.get('HOME')
                    if home:
                        sitedirs.append(
                            os.path.join(home,
                                         'Library',
                                         'Python',
                                         sys.version[:3],
                                         'site-packages'))
            for sitedir in sitedirs:
                if os.path.isdir(sitedir):
                    addsitedir(sitedir, known_paths)
    return None
def check_enableusersite():
    if hasattr(sys, 'flags') and getattr(sys.flags, 'no_user_site', False):
        return False
    if hasattr(os, "getuid") and hasattr(os, "geteuid"):
        if os.geteuid() != os.getuid():
            return None
    if hasattr(os, "getgid") and hasattr(os, "getegid"):
        if os.getegid() != os.getgid():
            return None
    return True
def addusersitepackages(known_paths):
    global USER_BASE, USER_SITE, ENABLE_USER_SITE
    env_base = os.environ.get("PYTHONUSERBASE", None)
    def joinuser(*args):
        return os.path.expanduser(os.path.join(*args))
    if os.name == "nt":
        base = os.environ.get("APPDATA") or "~"
        if env_base:
            USER_BASE = env_base
        else:
            USER_BASE = joinuser(base, "Python")
        USER_SITE = os.path.join(USER_BASE,
                                 "Python" + sys.version[0] + sys.version[2],
                                 "site-packages")
    else:
        if env_base:
            USER_BASE = env_base
        else:
            USER_BASE = joinuser("~", ".local")
        USER_SITE = os.path.join(USER_BASE, "lib",
                                 "python" + sys.version[:3],
                                 "site-packages")
    if ENABLE_USER_SITE and os.path.isdir(USER_SITE):
        addsitedir(USER_SITE, known_paths)
    if ENABLE_USER_SITE:
        for dist_libdir in ("lib", "local/lib"):
            user_site = os.path.join(USER_BASE, dist_libdir,
                                     "python" + sys.version[:3],
                                     "dist-packages")
            if os.path.isdir(user_site):
                addsitedir(user_site, known_paths)
    return known_paths
def setBEGINLIBPATH():
    dllpath = os.path.join(sys.prefix, "Lib", "lib-dynload")
    libpath = os.environ['BEGINLIBPATH'].split(';')
    if libpath[-1]:
        libpath.append(dllpath)
    else:
        libpath[-1] = dllpath
    os.environ['BEGINLIBPATH'] = ';'.join(libpath)
def setquit():
    if os.sep == ':':
        eof = 'Cmd-Q'
    elif os.sep == '\\':
        eof = 'Ctrl-Z plus Return'
    else:
        eof = 'Ctrl-D (i.e. EOF)'
    class Quitter(object):
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return 'Use %s() or %s to exit' % (self.name, eof)
        def __call__(self, code=None):
            try:
                sys.stdin.close()
            except:
                pass
            raise SystemExit(code)
    builtins.quit = Quitter('quit')
    builtins.exit = Quitter('exit')
class _Printer(object):
    MAXLINES = 23
    def __init__(self, name, data, files=(), dirs=()):
        self.__name = name
        self.__data = data
        self.__files = files
        self.__dirs = dirs
        self.__lines = None
    def __setup(self):
        if self.__lines:
            return
        data = None
        for dir in self.__dirs:
            for filename in self.__files:
                filename = os.path.join(dir, filename)
                try:
                    fp = open(filename, "rU")
                    data = fp.read()
                    fp.close()
                    break
                except IOError:
                    pass
            if data:
                break
        if not data:
            data = self.__data
        self.__lines = data.split('\n')
        self.__linecnt = len(self.__lines)
    def __repr__(self):
        self.__setup()
        if len(self.__lines) <= self.MAXLINES:
            return "\n".join(self.__lines)
        else:
            return "Type %s() to see the full %s text" % ((self.__name,)*2)
    def __call__(self):
        self.__setup()
        prompt = 'Hit Return for more, or q (and Return) to quit: '
        lineno = 0
        while 1:
            try:
                for i in range(lineno, lineno + self.MAXLINES):
                    print(self.__lines[i])
            except IndexError:
                break
            else:
                lineno += self.MAXLINES
                key = None
                while key is None:
                    try:
                        key = raw_input(prompt)
                    except NameError:
                        key = input(prompt)
                    if key not in ('', 'q'):
                        key = None
                if key == 'q':
                    break
def setcopyright():
    builtins.copyright = _Printer("copyright", sys.copyright)
    if _is_jython:
        builtins.credits = _Printer(
            "credits",
            "Jython is maintained by the Jython developers (www.jython.org).")
    elif _is_pypy:
        builtins.credits = _Printer(
            "credits",
            "PyPy is maintained by the PyPy developers: http://pypy.org/")
    else:
        builtins.credits = _Printer("credits",)
    here = os.path.dirname(os.__file__)
    builtins.license = _Printer(
        "license", "See http://www.python.org/%.3s/license.html" % sys.version,
        ["LICENSE.txt", "LICENSE"],
        [os.path.join(here, os.pardir), here, os.curdir])
class _Helper(object):
    def __repr__(self):
        return "Type help() for interactive help, "               "or help(object) for help about object."
    def __call__(self, *args, **kwds):
        import pydoc
        return pydoc.help(*args, **kwds)
def sethelper():
    builtins.help = _Helper()
def aliasmbcs():
    if sys.platform == 'win32':
        import locale, codecs
        enc = locale.getdefaultlocale()[1]
        if enc.startswith('cp'):
            try:
                codecs.lookup(enc)
            except LookupError:
                import encodings
                encodings._cache[enc] = encodings._unknown
                encodings.aliases.aliases[enc] = 'mbcs'
def setencoding():
    encoding = "ascii"
    if 0:
        import locale
        loc = locale.getdefaultlocale()
        if loc[1]:
            encoding = loc[1]
    if 0:
        encoding = "undefined"
    if encoding != "ascii":
        sys.setdefaultencoding(encoding)
def execsitecustomize():
    try:
        import sitecustomize
    except ImportError:
        pass
def virtual_install_main_packages():
    f = open(os.path.join(os.path.dirname(__file__), 'orig-prefix.txt'))
    sys.real_prefix = f.read().strip()
    f.close()
    pos = 2
    hardcoded_relative_dirs = []
    if sys.path[0] == '':
        pos += 1
    if _is_jython:
        paths = [os.path.join(sys.real_prefix, 'Lib')]
    elif _is_pypy:
        if sys.version_info > (3, 2):
            cpyver = '%d' % sys.version_info[0]
        elif sys.pypy_version_info >= (1, 5):
            cpyver = '%d.%d' % sys.version_info[:2]
        else:
            cpyver = '%d.%d.%d' % sys.version_info[:3]
        paths = [os.path.join(sys.real_prefix, 'lib_pypy'),
                 os.path.join(sys.real_prefix, 'lib-python', cpyver)]
        if sys.pypy_version_info < (1, 9):
            paths.insert(1, os.path.join(sys.real_prefix,
                                         'lib-python', 'modified-%s' % cpyver))
        hardcoded_relative_dirs = paths[:]
        for path in paths[:]:
            plat_path = os.path.join(path, 'plat-%s' % sys.platform)
            if os.path.exists(plat_path):
                paths.append(plat_path)
    elif sys.platform == 'win32':
        paths = [os.path.join(sys.real_prefix, 'Lib'), os.path.join(sys.real_prefix, 'DLLs')]
    else:
        paths = [os.path.join(sys.real_prefix, 'lib', 'python'+sys.version[:3])]
        hardcoded_relative_dirs = paths[:]
        lib64_path = os.path.join(sys.real_prefix, 'lib64', 'python'+sys.version[:3])
        if os.path.exists(lib64_path):
            if _is_64bit:
                paths.insert(0, lib64_path)
            else:
                paths.append(lib64_path)
        try:
            arch = getattr(sys, 'implementation', sys)._multiarch
        except AttributeError:
            arch = sys.platform
        plat_path = os.path.join(sys.real_prefix, 'lib',
                                 'python'+sys.version[:3],
                                 'plat-%s' % arch)
        if os.path.exists(plat_path):
            paths.append(plat_path)
    for path in list(paths):
        tk_dir = os.path.join(path, 'lib-tk')
        if os.path.exists(tk_dir):
            paths.append(tk_dir)
    if sys.platform == 'darwin':
        hardcoded_paths = [os.path.join(relative_dir, module)
                           for relative_dir in hardcoded_relative_dirs
                           for module in ('plat-darwin', 'plat-mac', 'plat-mac/lib-scriptpackages')]
        for path in hardcoded_paths:
            if os.path.exists(path):
                paths.append(path)
    sys.path.extend(paths)
def force_global_eggs_after_local_site_packages():
    egginsert = getattr(sys, '__egginsert', 0)
    for i, path in enumerate(sys.path):
        if i > egginsert and path.startswith(sys.prefix):
            egginsert = i
    sys.__egginsert = egginsert + 1
def virtual_addsitepackages(known_paths):
    force_global_eggs_after_local_site_packages()
    return addsitepackages(known_paths, sys_prefix=sys.real_prefix)
def fixclasspath():
    paths = []
    classpaths = []
    for path in sys.path:
        if path == '__classpath__' or path.startswith('__pyclasspath__'):
            classpaths.append(path)
        else:
            paths.append(path)
    sys.path = paths
    sys.path.extend(classpaths)
def execusercustomize():
    try:
        import usercustomize
    except ImportError:
        pass
def main():
    global ENABLE_USER_SITE
    virtual_install_main_packages()
    abs__file__()
    paths_in_sys = removeduppaths()
    if (os.name == "posix" and sys.path and
        os.path.basename(sys.path[-1]) == "Modules"):
        addbuilddir()
    if _is_jython:
        fixclasspath()
    GLOBAL_SITE_PACKAGES = not os.path.exists(os.path.join(os.path.dirname(__file__), 'no-global-site-packages.txt'))
    if not GLOBAL_SITE_PACKAGES:
        ENABLE_USER_SITE = False
    if ENABLE_USER_SITE is None:
        ENABLE_USER_SITE = check_enableusersite()
    paths_in_sys = addsitepackages(paths_in_sys)
    paths_in_sys = addusersitepackages(paths_in_sys)
    if GLOBAL_SITE_PACKAGES:
        paths_in_sys = virtual_addsitepackages(paths_in_sys)
    if sys.platform == 'os2emx':
        setBEGINLIBPATH()
    setquit()
    setcopyright()
    sethelper()
    aliasmbcs()
    setencoding()
    execsitecustomize()
    if ENABLE_USER_SITE:
        execusercustomize()
    if hasattr(sys, "setdefaultencoding"):
        del sys.setdefaultencoding
main()
def _script():
    help =
    args = sys.argv[1:]
    if not args:
        print("sys.path = [")
        for dir in sys.path:
            print("    %r," % (dir,))
        print("]")
        def exists(path):
            if os.path.isdir(path):
                return "exists"
            else:
                return "doesn't exist"
        print("USER_BASE: %r (%s)" % (USER_BASE, exists(USER_BASE)))
        print("USER_SITE: %r (%s)" % (USER_SITE, exists(USER_BASE)))
        print("ENABLE_USER_SITE: %r" %  ENABLE_USER_SITE)
        sys.exit(0)
    buffer = []
    if '--user-base' in args:
        buffer.append(USER_BASE)
    if '--user-site' in args:
        buffer.append(USER_SITE)
    if buffer:
        print(os.pathsep.join(buffer))
        if ENABLE_USER_SITE:
            sys.exit(0)
        elif ENABLE_USER_SITE is False:
            sys.exit(1)
        elif ENABLE_USER_SITE is None:
            sys.exit(2)
        else:
            sys.exit(3)
    else:
        import textwrap
        print(textwrap.dedent(help % (sys.argv[0], os.pathsep)))
        sys.exit(10)
if __name__ == '__main__':
    _script()
