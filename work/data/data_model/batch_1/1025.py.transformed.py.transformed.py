
import sys
import os
import subprocess
from distutils.errors import DistutilsPlatformError, DistutilsExecError
from distutils.debug import DEBUG
from distutils import log
if sys.platform == 'darwin':
    _cfg_target = None
    _cfg_target_split = None
def spawn(cmd, search_path=1, verbose=0, dry_run=0, env=None):
    cmd = list(cmd)
    log.info(subprocess.list2cmdline(cmd))
    if dry_run:
        return
    if search_path:
        executable = find_executable(cmd[0])
        if executable is not None:
            cmd[0] = executable
    env = env if env is not None else dict(os.environ)
    if sys.platform == 'darwin':
        global _cfg_target, _cfg_target_split
        if _cfg_target is None:
            from distutils import sysconfig
            _cfg_target = sysconfig.get_config_var(
                                  'MACOSX_DEPLOYMENT_TARGET') or ''
            if _cfg_target:
                _cfg_target_split = [int(x) for x in _cfg_target.split('.')]
        if _cfg_target:
            cur_target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', _cfg_target)
            cur_target_split = [int(x) for x in cur_target.split('.')]
            if _cfg_target_split[:2] >= [10, 3] and cur_target_split[:2] < [10, 3]:
                my_msg = ('$MACOSX_DEPLOYMENT_TARGET mismatch: '
                          'now "%s" but "%s" during configure;'
                          'must use 10.3 or later'
                                % (cur_target, _cfg_target))
                raise DistutilsPlatformError(my_msg)
            env.update(MACOSX_DEPLOYMENT_TARGET=cur_target)
    try:
        proc = subprocess.Popen(cmd, env=env)
        proc.wait()
        exitcode = proc.returncode
    except OSError as exc:
        if not DEBUG:
            cmd = cmd[0]
        raise DistutilsExecError(
            "command %r failed: %s" % (cmd, exc.args[-1])) from exc
    if exitcode:
        if not DEBUG:
            cmd = cmd[0]
        raise DistutilsExecError(
              "command %r failed with exit code %s" % (cmd, exitcode))
def find_executable(executable, path=None):
    _, ext = os.path.splitext(executable)
    if (sys.platform == 'win32') and (ext != '.exe'):
        executable = executable + '.exe'
    if os.path.isfile(executable):
        return executable
    if path is None:
        path = os.environ.get('PATH', None)
        if path is None:
            try:
                path = os.confstr("CS_PATH")
            except (AttributeError, ValueError):
                path = os.defpath
    if not path:
        return None
    paths = path.split(os.pathsep)
    for p in paths:
        f = os.path.join(p, executable)
        if os.path.isfile(f):
            return f
    return None
