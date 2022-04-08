
import io
import os
import sys
import tokenize
import shutil
import contextlib
import tempfile
import setuptools
import distutils
from pkg_resources import parse_requirements
__all__ = ['get_requires_for_build_sdist',
           'get_requires_for_build_wheel',
           'prepare_metadata_for_build_wheel',
           'build_wheel',
           'build_sdist',
           '__legacy__',
           'SetupRequirementsError']
class SetupRequirementsError(BaseException):
    def __init__(self, specifiers):
        self.specifiers = specifiers
class Distribution(setuptools.dist.Distribution):
    def fetch_build_eggs(self, specifiers):
        specifier_list = list(map(str, parse_requirements(specifiers)))
        raise SetupRequirementsError(specifier_list)
    @classmethod
    @contextlib.contextmanager
    def patch(cls):
        orig = distutils.core.Distribution
        distutils.core.Distribution = cls
        try:
            yield
        finally:
            distutils.core.Distribution = orig
@contextlib.contextmanager
def no_install_setup_requires():
    orig = setuptools._install_setup_requires
    setuptools._install_setup_requires = lambda attrs: None
    try:
        yield
    finally:
        setuptools._install_setup_requires = orig
def _get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
def _file_with_extension(directory, extension):
    matching = (
        f for f in os.listdir(directory)
        if f.endswith(extension)
    )
    try:
        file, = matching
    except ValueError:
        raise ValueError(
            'No distribution was found. Ensure that `setup.py` '
            'is not empty and that it calls `setup()`.')
    return file
def _open_setup_script(setup_script):
    if not os.path.exists(setup_script):
        return io.StringIO(u"from setuptools import setup; setup()")
    return getattr(tokenize, 'open', open)(setup_script)
class _BuildMetaBackend(object):
    def _fix_config(self, config_settings):
        config_settings = config_settings or {}
        config_settings.setdefault('--global-option', [])
        return config_settings
    def _get_build_requires(self, config_settings, requirements):
        config_settings = self._fix_config(config_settings)
        sys.argv = sys.argv[:1] + ['egg_info'] +            config_settings["--global-option"]
        try:
            with Distribution.patch():
                self.run_setup()
        except SetupRequirementsError as e:
            requirements += e.specifiers
        return requirements
    def run_setup(self, setup_script='setup.py'):
        __file__ = setup_script
        __name__ = '__main__'
        with _open_setup_script(__file__) as f:
            code = f.read().replace(r'\r\n', r'\n')
        exec(compile(code, __file__, 'exec'), locals())
    def get_requires_for_build_wheel(self, config_settings=None):
        config_settings = self._fix_config(config_settings)
        return self._get_build_requires(
            config_settings, requirements=['wheel'])
    def get_requires_for_build_sdist(self, config_settings=None):
        config_settings = self._fix_config(config_settings)
        return self._get_build_requires(config_settings, requirements=[])
    def prepare_metadata_for_build_wheel(self, metadata_directory,
                                         config_settings=None):
        sys.argv = sys.argv[:1] + [
            'dist_info', '--egg-base', metadata_directory]
        with no_install_setup_requires():
            self.run_setup()
        dist_info_directory = metadata_directory
        while True:
            dist_infos = [f for f in os.listdir(dist_info_directory)
                          if f.endswith('.dist-info')]
            if (
                len(dist_infos) == 0 and
                len(_get_immediate_subdirectories(dist_info_directory)) == 1
            ):
                dist_info_directory = os.path.join(
                    dist_info_directory, os.listdir(dist_info_directory)[0])
                continue
            assert len(dist_infos) == 1
            break
        if dist_info_directory != metadata_directory:
            shutil.move(
                os.path.join(dist_info_directory, dist_infos[0]),
                metadata_directory)
            shutil.rmtree(dist_info_directory, ignore_errors=True)
        return dist_infos[0]
    def _build_with_temp_dir(self, setup_command, result_extension,
                             result_directory, config_settings):
        config_settings = self._fix_config(config_settings)
        result_directory = os.path.abspath(result_directory)
        os.makedirs(result_directory, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=result_directory) as tmp_dist_dir:
            sys.argv = (sys.argv[:1] + setup_command +
                        ['--dist-dir', tmp_dist_dir] +
                        config_settings["--global-option"])
            with no_install_setup_requires():
                self.run_setup()
            result_basename = _file_with_extension(
                tmp_dist_dir, result_extension)
            result_path = os.path.join(result_directory, result_basename)
            if os.path.exists(result_path):
                os.remove(result_path)
            os.rename(os.path.join(tmp_dist_dir, result_basename), result_path)
        return result_basename
    def build_wheel(self, wheel_directory, config_settings=None,
                    metadata_directory=None):
        return self._build_with_temp_dir(['bdist_wheel'], '.whl',
                                         wheel_directory, config_settings)
    def build_sdist(self, sdist_directory, config_settings=None):
        return self._build_with_temp_dir(['sdist', '--formats', 'gztar'],
                                         '.tar.gz', sdist_directory,
                                         config_settings)
class _BuildMetaLegacyBackend(_BuildMetaBackend):
    def run_setup(self, setup_script='setup.py'):
        sys_path = list(sys.path)
        script_dir = os.path.dirname(os.path.abspath(setup_script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        sys_argv_0 = sys.argv[0]
        sys.argv[0] = setup_script
        try:
            super(_BuildMetaLegacyBackend,
                  self).run_setup(setup_script=setup_script)
        finally:
            sys.path[:] = sys_path
            sys.argv[0] = sys_argv_0
_BACKEND = _BuildMetaBackend()
get_requires_for_build_wheel = _BACKEND.get_requires_for_build_wheel
get_requires_for_build_sdist = _BACKEND.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _BACKEND.prepare_metadata_for_build_wheel
build_wheel = _BACKEND.build_wheel
build_sdist = _BACKEND.build_sdist
__legacy__ = _BuildMetaLegacyBackend()
