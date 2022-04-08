import threading
from contextlib import contextmanager
import os
from os.path import abspath, join as pjoin
import shutil
from subprocess import check_call, check_output, STDOUT
import sys
from tempfile import mkdtemp
from . import compat
from .in_process import _in_proc_script_path
__all__ = [
    'BackendUnavailable',
    'BackendInvalid',
    'HookMissing',
    'UnsupportedOperation',
    'default_subprocess_runner',
    'quiet_subprocess_runner',
    'Pep517HookCaller',
]
@contextmanager
def tempdir():
    td = mkdtemp()
    try:
        yield td
    finally:
        shutil.rmtree(td)
class BackendUnavailable(Exception):
    def __init__(self, traceback):
        self.traceback = traceback
class BackendInvalid(Exception):
    def __init__(self, backend_name, backend_path, message):
        self.backend_name = backend_name
        self.backend_path = backend_path
        self.message = message
class HookMissing(Exception):
    def __init__(self, hook_name):
        super(HookMissing, self).__init__(hook_name)
        self.hook_name = hook_name
class UnsupportedOperation(Exception):
    def __init__(self, traceback):
        self.traceback = traceback
def default_subprocess_runner(cmd, cwd=None, extra_environ=None):
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)
    check_call(cmd, cwd=cwd, env=env)
def quiet_subprocess_runner(cmd, cwd=None, extra_environ=None):
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)
    check_output(cmd, cwd=cwd, env=env, stderr=STDOUT)
def norm_and_check(source_tree, requested):
    if os.path.isabs(requested):
        raise ValueError("paths must be relative")
    abs_source = os.path.abspath(source_tree)
    abs_requested = os.path.normpath(os.path.join(abs_source, requested))
    norm_source = os.path.normcase(abs_source)
    norm_requested = os.path.normcase(abs_requested)
    if os.path.commonprefix([norm_source, norm_requested]) != norm_source:
        raise ValueError("paths must be inside source tree")
    return abs_requested
class Pep517HookCaller(object):
    def __init__(
            self,
            source_dir,
            build_backend,
            backend_path=None,
            runner=None,
            python_executable=None,
    ):
        if runner is None:
            runner = default_subprocess_runner
        self.source_dir = abspath(source_dir)
        self.build_backend = build_backend
        if backend_path:
            backend_path = [
                norm_and_check(self.source_dir, p) for p in backend_path
            ]
        self.backend_path = backend_path
        self._subprocess_runner = runner
        if not python_executable:
            python_executable = sys.executable
        self.python_executable = python_executable
    @contextmanager
    def subprocess_runner(self, runner):
        prev = self._subprocess_runner
        self._subprocess_runner = runner
        try:
            yield
        finally:
            self._subprocess_runner = prev
    def get_requires_for_build_wheel(self, config_settings=None):
        return self._call_hook('get_requires_for_build_wheel', {
            'config_settings': config_settings
        })
    def prepare_metadata_for_build_wheel(
            self, metadata_directory, config_settings=None,
            _allow_fallback=True):
        return self._call_hook('prepare_metadata_for_build_wheel', {
            'metadata_directory': abspath(metadata_directory),
            'config_settings': config_settings,
            '_allow_fallback': _allow_fallback,
        })
    def build_wheel(
            self, wheel_directory, config_settings=None,
            metadata_directory=None):
        if metadata_directory is not None:
            metadata_directory = abspath(metadata_directory)
        return self._call_hook('build_wheel', {
            'wheel_directory': abspath(wheel_directory),
            'config_settings': config_settings,
            'metadata_directory': metadata_directory,
        })
    def get_requires_for_build_sdist(self, config_settings=None):
        return self._call_hook('get_requires_for_build_sdist', {
            'config_settings': config_settings
        })
    def build_sdist(self, sdist_directory, config_settings=None):
        return self._call_hook('build_sdist', {
            'sdist_directory': abspath(sdist_directory),
            'config_settings': config_settings,
        })
    def _call_hook(self, hook_name, kwargs):
        if sys.version_info[0] == 2:
            build_backend = self.build_backend.encode('ASCII')
        else:
            build_backend = self.build_backend
        extra_environ = {'PEP517_BUILD_BACKEND': build_backend}
        if self.backend_path:
            backend_path = os.pathsep.join(self.backend_path)
            if sys.version_info[0] == 2:
                backend_path = backend_path.encode(sys.getfilesystemencoding())
            extra_environ['PEP517_BACKEND_PATH'] = backend_path
        with tempdir() as td:
            hook_input = {'kwargs': kwargs}
            compat.write_json(hook_input, pjoin(td, 'input.json'),
                              indent=2)
            with _in_proc_script_path() as script:
                python = self.python_executable
                self._subprocess_runner(
                    [python, abspath(str(script)), hook_name, td],
                    cwd=self.source_dir,
                    extra_environ=extra_environ
                )
            data = compat.read_json(pjoin(td, 'output.json'))
            if data.get('unsupported'):
                raise UnsupportedOperation(data.get('traceback', ''))
            if data.get('no_backend'):
                raise BackendUnavailable(data.get('traceback', ''))
            if data.get('backend_invalid'):
                raise BackendInvalid(
                    backend_name=self.build_backend,
                    backend_path=self.backend_path,
                    message=data.get('backend_error', '')
                )
            if data.get('hook_missing'):
                raise HookMissing(hook_name)
            return data['return_val']
class LoggerWrapper(threading.Thread):
    def __init__(self, logger, level):
        threading.Thread.__init__(self)
        self.daemon = True
        self.logger = logger
        self.level = level
        self.fd_read, self.fd_write = os.pipe()
        self.reader = os.fdopen(self.fd_read)
        self.start()
    def fileno(self):
        return self.fd_write
    @staticmethod
    def remove_newline(msg):
        return msg[:-1] if msg.endswith(os.linesep) else msg
    def run(self):
        for line in self.reader:
            self._write(self.remove_newline(line))
    def _write(self, message):
        self.logger.log(self.level, message)
