
__all__ = ['exec_command', 'find_executable']
import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log
def filepath_from_subprocess_output(output):
    mylocale = locale.getpreferredencoding(False)
    if mylocale is None:
        mylocale = 'ascii'
    output = output.decode(mylocale, errors='replace')
    output = output.replace('\r\n', '\n')
    if output[-1:] == '\n':
        output = output[:-1]
    return output
def forward_bytes_to_stdout(val):
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout.buffer.write(val)
    elif hasattr(sys.stdout, 'encoding'):
        sys.stdout.write(val.decode(sys.stdout.encoding))
    else:
        sys.stdout.write(val.decode('utf8', errors='replace'))
def temp_file_name():
    warnings.warn('temp_file_name is deprecated since NumPy v1.17, use '
                  'tempfile.mkstemp instead', DeprecationWarning, stacklevel=1)
    fo, name = make_temp_file()
    fo.close()
    return name
def get_pythonexe():
    pythonexe = sys.executable
    if os.name in ['nt', 'dos']:
        fdir, fn = os.path.split(pythonexe)
        fn = fn.upper().replace('PYTHONW', 'PYTHON')
        pythonexe = os.path.join(fdir, fn)
        assert os.path.isfile(pythonexe), '%r is not a file' % (pythonexe,)
    return pythonexe
def find_executable(exe, path=None, _cache={}):
    key = exe, path
    try:
        return _cache[key]
    except KeyError:
        pass
    log.debug('find_executable(%r)' % exe)
    orig_exe = exe
    if path is None:
        path = os.environ.get('PATH', os.defpath)
    if os.name=='posix':
        realpath = os.path.realpath
    else:
        realpath = lambda a:a
    if exe.startswith('"'):
        exe = exe[1:-1]
    suffixes = ['']
    if os.name in ['nt', 'dos', 'os2']:
        fn, ext = os.path.splitext(exe)
        extra_suffixes = ['.exe', '.com', '.bat']
        if ext.lower() not in extra_suffixes:
            suffixes = extra_suffixes
    if os.path.isabs(exe):
        paths = ['']
    else:
        paths = [ os.path.abspath(p) for p in path.split(os.pathsep) ]
    for path in paths:
        fn = os.path.join(path, exe)
        for s in suffixes:
            f_ext = fn+s
            if not os.path.islink(f_ext):
                f_ext = realpath(f_ext)
            if os.path.isfile(f_ext) and os.access(f_ext, os.X_OK):
                log.info('Found executable %s' % f_ext)
                _cache[key] = f_ext
                return f_ext
    log.warn('Could not locate executable %s' % orig_exe)
    return None
def _preserve_environment( names ):
    log.debug('_preserve_environment(%r)' % (names))
    env = {name: os.environ.get(name) for name in names}
    return env
def _update_environment( **env ):
    log.debug('_update_environment(...)')
    for name, value in env.items():
        os.environ[name] = value or ''
def exec_command(command, execute_in='', use_shell=None, use_tee=None,
                 _with_python = 1, **env ):
    warnings.warn('exec_command is deprecated since NumPy v1.17, use '
                  'subprocess.Popen instead', DeprecationWarning, stacklevel=1)
    log.debug('exec_command(%r,%s)' % (command,
         ','.join(['%s=%r'%kv for kv in env.items()])))
    if use_tee is None:
        use_tee = os.name=='posix'
    if use_shell is None:
        use_shell = os.name=='posix'
    execute_in = os.path.abspath(execute_in)
    oldcwd = os.path.abspath(os.getcwd())
    if __name__[-12:] == 'exec_command':
        exec_dir = os.path.dirname(os.path.abspath(__file__))
    elif os.path.isfile('exec_command.py'):
        exec_dir = os.path.abspath('.')
    else:
        exec_dir = os.path.abspath(sys.argv[0])
        if os.path.isfile(exec_dir):
            exec_dir = os.path.dirname(exec_dir)
    if oldcwd!=execute_in:
        os.chdir(execute_in)
        log.debug('New cwd: %s' % execute_in)
    else:
        log.debug('Retaining cwd: %s' % oldcwd)
    oldenv = _preserve_environment( list(env.keys()) )
    _update_environment( **env )
    try:
        st = _exec_command(command,
                           use_shell=use_shell,
                           use_tee=use_tee,
                           **env)
    finally:
        if oldcwd!=execute_in:
            os.chdir(oldcwd)
            log.debug('Restored cwd to %s' % oldcwd)
        _update_environment(**oldenv)
    return st
def _exec_command(command, use_shell=None, use_tee = None, **env):
    if use_shell is None:
        use_shell = os.name=='posix'
    if use_tee is None:
        use_tee = os.name=='posix'
    if os.name == 'posix' and use_shell:
        sh = os.environ.get('SHELL', '/bin/sh')
        if is_sequence(command):
            command = [sh, '-c', ' '.join(command)]
        else:
            command = [sh, '-c', command]
        use_shell = False
    elif os.name == 'nt' and is_sequence(command):
        command = ' '.join(_quote_arg(arg) for arg in command)
    env = env or None
    try:
        proc = subprocess.Popen(command, shell=use_shell, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=False)
    except EnvironmentError:
        return 127, ''
    text, err = proc.communicate()
    mylocale = locale.getpreferredencoding(False)
    if mylocale is None:
        mylocale = 'ascii'
    text = text.decode(mylocale, errors='replace')
    text = text.replace('\r\n', '\n')
    if text[-1:] == '\n':
        text = text[:-1]
    if use_tee and text:
        print(text)
    return proc.returncode, text
def _quote_arg(arg):
    if '"' not in arg and ' ' in arg:
        return '"%s"' % arg
    return arg
