import distutils
import os
import shutil
import stat
import sys
import re
import warnings
from collections import OrderedDict
from distutils.core import Command
from distutils import log as logger
from io import BytesIO
from glob import iglob
from shutil import rmtree
from sysconfig import get_config_var
from zipfile import ZIP_DEFLATED, ZIP_STORED
import pkg_resources
from .pkginfo import write_pkg_info
from .macosx_libfile import calculate_macosx_platform_tag
from .metadata import pkginfo_to_metadata
from .vendored.packaging import tags
from .wheelfile import WheelFile
from . import __version__ as wheel_version
if sys.version_info < (3,):
    from email.generator import Generator as BytesGenerator
else:
    from email.generator import BytesGenerator
safe_name = pkg_resources.safe_name
safe_version = pkg_resources.safe_version
PY_LIMITED_API_PATTERN = r'cp3\d'
def python_tag():
    return 'py{}'.format(sys.version_info[0])
def get_platform(archive_root):
    result = distutils.util.get_platform()
    if result.startswith("macosx") and archive_root is not None:
        result = calculate_macosx_platform_tag(archive_root, result)
    if result == "linux_x86_64" and sys.maxsize == 2147483647:
        result = "linux_i686"
    return result
def get_flag(var, fallback, expected=True, warn=True):
    val = get_config_var(var)
    if val is None:
        if warn:
            warnings.warn("Config variable '{0}' is unset, Python ABI tag may "
                          "be incorrect".format(var), RuntimeWarning, 2)
        return fallback
    return val == expected
def get_abi_tag():
    soabi = get_config_var('SOABI')
    impl = tags.interpreter_name()
    if not soabi and impl in ('cp', 'pp') and hasattr(sys, 'maxunicode'):
        d = ''
        m = ''
        u = ''
        if get_flag('Py_DEBUG',
                    hasattr(sys, 'gettotalrefcount'),
                    warn=(impl == 'cp')):
            d = 'd'
        if get_flag('WITH_PYMALLOC',
                    impl == 'cp',
                    warn=(impl == 'cp' and
                          sys.version_info < (3, 8)))                and sys.version_info < (3, 8):
            m = 'm'
        if get_flag('Py_UNICODE_SIZE',
                    sys.maxunicode == 0x10ffff,
                    expected=4,
                    warn=(impl == 'cp' and
                          sys.version_info < (3, 3)))                and sys.version_info < (3, 3):
            u = 'u'
        abi = '%s%s%s%s%s' % (impl, tags.interpreter_version(), d, m, u)
    elif soabi and soabi.startswith('cpython-'):
        abi = 'cp' + soabi.split('-')[1]
    elif soabi and soabi.startswith('pypy-'):
        abi = '-'.join(soabi.split('-')[:2])
        abi = abi.replace('.', '_').replace('-', '_')
    elif soabi:
        abi = soabi.replace('.', '_').replace('-', '_')
    else:
        abi = None
    return abi
def safer_name(name):
    return safe_name(name).replace('-', '_')
def safer_version(version):
    return safe_version(version).replace('-', '_')
def remove_readonly(func, path, excinfo):
    print(str(excinfo[1]))
    os.chmod(path, stat.S_IWRITE)
    func(path)
class bdist_wheel(Command):
    description = 'create a wheel distribution'
    supported_compressions = OrderedDict([
        ('stored', ZIP_STORED),
        ('deflated', ZIP_DEFLATED)
    ])
    user_options = [('bdist-dir=', 'b',
                     "temporary directory for creating the distribution"),
                    ('plat-name=', 'p',
                     "platform name to embed in generated filenames "
                     "(default: %s)" % get_platform(None)),
                    ('keep-temp', 'k',
                     "keep the pseudo-installation tree around after " +
                     "creating the distribution archive"),
                    ('dist-dir=', 'd',
                     "directory to put final built distributions in"),
                    ('skip-build', None,
                     "skip rebuilding everything (for testing/debugging)"),
                    ('relative', None,
                     "build the archive using relative paths "
                     "(default: false)"),
                    ('owner=', 'u',
                     "Owner name used when creating a tar file"
                     " [default: current user]"),
                    ('group=', 'g',
                     "Group name used when creating a tar file"
                     " [default: current group]"),
                    ('universal', None,
                     "make a universal wheel"
                     " (default: false)"),
                    ('compression=', None,
                     "zipfile compression (one of: {})"
                     " (default: 'deflated')"
                     .format(', '.join(supported_compressions))),
                    ('python-tag=', None,
                     "Python implementation compatibility tag"
                     " (default: '%s')" % (python_tag())),
                    ('build-number=', None,
                     "Build number for this particular version. "
                     "As specified in PEP-0427, this must start with a digit. "
                     "[default: None]"),
                    ('py-limited-api=', None,
                     "Python tag (cp32|cp33|cpNN) for abi3 wheel tag"
                     " (default: false)"),
                    ]
    boolean_options = ['keep-temp', 'skip-build', 'relative', 'universal']
    def initialize_options(self):
        self.bdist_dir = None
        self.data_dir = None
        self.plat_name = None
        self.plat_tag = None
        self.format = 'zip'
        self.keep_temp = False
        self.dist_dir = None
        self.egginfo_dir = None
        self.root_is_pure = None
        self.skip_build = None
        self.relative = False
        self.owner = None
        self.group = None
        self.universal = False
        self.compression = 'deflated'
        self.python_tag = python_tag()
        self.build_number = None
        self.py_limited_api = False
        self.plat_name_supplied = False
    def finalize_options(self):
        if self.bdist_dir is None:
            bdist_base = self.get_finalized_command('bdist').bdist_base
            self.bdist_dir = os.path.join(bdist_base, 'wheel')
        self.data_dir = self.wheel_dist_name + '.data'
        self.plat_name_supplied = self.plat_name is not None
        try:
            self.compression = self.supported_compressions[self.compression]
        except KeyError:
            raise ValueError('Unsupported compression: {}'.format(self.compression))
        need_options = ('dist_dir', 'plat_name', 'skip_build')
        self.set_undefined_options('bdist',
                                   *zip(need_options, need_options))
        self.root_is_pure = not (self.distribution.has_ext_modules()
                                 or self.distribution.has_c_libraries())
        if self.py_limited_api and not re.match(PY_LIMITED_API_PATTERN, self.py_limited_api):
            raise ValueError("py-limited-api must match '%s'" % PY_LIMITED_API_PATTERN)
        wheel = self.distribution.get_option_dict('wheel')
        if 'universal' in wheel:
            logger.warn('The [wheel] section is deprecated. Use [bdist_wheel] instead.')
            val = wheel['universal'][1].strip()
            if val.lower() in ('1', 'true', 'yes'):
                self.universal = True
        if self.build_number is not None and not self.build_number[:1].isdigit():
            raise ValueError("Build tag (build-number) must start with a digit.")
    @property
    def wheel_dist_name(self):
        components = (safer_name(self.distribution.get_name()),
                      safer_version(self.distribution.get_version()))
        if self.build_number:
            components += (self.build_number,)
        return '-'.join(components)
    def get_tag(self):
        if self.plat_name_supplied:
            plat_name = self.plat_name
        elif self.root_is_pure:
            plat_name = 'any'
        else:
            if self.plat_name and not self.plat_name.startswith("macosx"):
                plat_name = self.plat_name
            else:
                plat_name = get_platform(self.bdist_dir)
            if plat_name in ('linux-x86_64', 'linux_x86_64') and sys.maxsize == 2147483647:
                plat_name = 'linux_i686'
        plat_name = plat_name.lower().replace('-', '_').replace('.', '_')
        if self.root_is_pure:
            if self.universal:
                impl = 'py2.py3'
            else:
                impl = self.python_tag
            tag = (impl, 'none', plat_name)
        else:
            impl_name = tags.interpreter_name()
            impl_ver = tags.interpreter_version()
            impl = impl_name + impl_ver
            if self.py_limited_api and (impl_name + impl_ver).startswith('cp3'):
                impl = self.py_limited_api
                abi_tag = 'abi3'
            else:
                abi_tag = str(get_abi_tag()).lower()
            tag = (impl, abi_tag, plat_name)
            supported_tags = [(t.interpreter, t.abi, plat_name)
                              for t in tags.sys_tags()]
            assert tag in supported_tags, "would build wheel with unsupported tag {}".format(tag)
        return tag
    def run(self):
        build_scripts = self.reinitialize_command('build_scripts')
        build_scripts.executable = 'python'
        build_scripts.force = True
        build_ext = self.reinitialize_command('build_ext')
        build_ext.inplace = False
        if not self.skip_build:
            self.run_command('build')
        install = self.reinitialize_command('install',
                                            reinit_subcommands=True)
        install.root = self.bdist_dir
        install.compile = False
        install.skip_build = self.skip_build
        install.warn_dir = False
        install_scripts = self.reinitialize_command('install_scripts')
        install_scripts.no_ep = True
        for key in ('headers', 'scripts', 'data', 'purelib', 'platlib'):
            setattr(install,
                    'install_' + key,
                    os.path.join(self.data_dir, key))
        basedir_observed = ''
        if os.name == 'nt':
            basedir_observed = os.path.normpath(os.path.join(self.data_dir, '..'))
            self.install_libbase = self.install_lib = basedir_observed
        setattr(install,
                'install_purelib' if self.root_is_pure else 'install_platlib',
                basedir_observed)
        logger.info("installing to %s", self.bdist_dir)
        self.run_command('install')
        impl_tag, abi_tag, plat_tag = self.get_tag()
        archive_basename = "{}-{}-{}-{}".format(self.wheel_dist_name, impl_tag, abi_tag, plat_tag)
        if not self.relative:
            archive_root = self.bdist_dir
        else:
            archive_root = os.path.join(
                self.bdist_dir,
                self._ensure_relative(install.install_base))
        self.set_undefined_options('install_egg_info', ('target', 'egginfo_dir'))
        distinfo_dirname = '{}-{}.dist-info'.format(
            safer_name(self.distribution.get_name()),
            safer_version(self.distribution.get_version()))
        distinfo_dir = os.path.join(self.bdist_dir, distinfo_dirname)
        self.egg2dist(self.egginfo_dir, distinfo_dir)
        self.write_wheelfile(distinfo_dir)
        if not os.path.exists(self.dist_dir):
            os.makedirs(self.dist_dir)
        wheel_path = os.path.join(self.dist_dir, archive_basename + '.whl')
        with WheelFile(wheel_path, 'w', self.compression) as wf:
            wf.write_files(archive_root)
        getattr(self.distribution, 'dist_files', []).append(
            ('bdist_wheel',
             '{}.{}'.format(*sys.version_info[:2]),
             wheel_path))
        if not self.keep_temp:
            logger.info('removing %s', self.bdist_dir)
            if not self.dry_run:
                rmtree(self.bdist_dir, onerror=remove_readonly)
    def write_wheelfile(self, wheelfile_base, generator='bdist_wheel (' + wheel_version + ')'):
        from email.message import Message
        if sys.version_info < (3,) and not isinstance(generator, str):
            generator = generator.encode('utf-8')
        msg = Message()
        msg['Wheel-Version'] = '1.0'
        msg['Generator'] = generator
        msg['Root-Is-Purelib'] = str(self.root_is_pure).lower()
        if self.build_number is not None:
            msg['Build'] = self.build_number
        impl_tag, abi_tag, plat_tag = self.get_tag()
        for impl in impl_tag.split('.'):
            for abi in abi_tag.split('.'):
                for plat in plat_tag.split('.'):
                    msg['Tag'] = '-'.join((impl, abi, plat))
        wheelfile_path = os.path.join(wheelfile_base, 'WHEEL')
        logger.info('creating %s', wheelfile_path)
        buffer = BytesIO()
        BytesGenerator(buffer, maxheaderlen=0).flatten(msg)
        with open(wheelfile_path, 'wb') as f:
            f.write(buffer.getvalue().replace(b'\r\n', b'\r'))
    def _ensure_relative(self, path):
        drive, path = os.path.splitdrive(path)
        if path[0:1] == os.sep:
            path = drive + path[1:]
        return path
    @property
    def license_paths(self):
        metadata = self.distribution.get_option_dict('metadata')
        files = set()
        patterns = sorted({
            option for option in metadata.get('license_files', ('', ''))[1].split()
        })
        if 'license_file' in metadata:
            warnings.warn('The "license_file" option is deprecated. Use '
                          '"license_files" instead.', DeprecationWarning)
            files.add(metadata['license_file'][1])
        if 'license_file' not in metadata and 'license_files' not in metadata:
            patterns = ('LICEN[CS]E*', 'COPYING*', 'NOTICE*', 'AUTHORS*')
        for pattern in patterns:
            for path in iglob(pattern):
                if path.endswith('~'):
                    logger.debug('ignoring license file "%s" as it looks like a backup', path)
                    continue
                if path not in files and os.path.isfile(path):
                    logger.info('adding license file "%s" (matched pattern "%s")', path, pattern)
                    files.add(path)
        return files
    def egg2dist(self, egginfo_path, distinfo_path):
        def adios(p):
            if os.path.exists(p) and not os.path.islink(p) and os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.unlink(p)
        adios(distinfo_path)
        if not os.path.exists(egginfo_path):
            import glob
            pat = os.path.join(os.path.dirname(egginfo_path), '*.egg-info')
            possible = glob.glob(pat)
            err = "Egg metadata expected at %s but not found" % (egginfo_path,)
            if possible:
                alt = os.path.basename(possible[0])
                err += " (%s found - possible misnamed archive file?)" % (alt,)
            raise ValueError(err)
        if os.path.isfile(egginfo_path):
            pkginfo_path = egginfo_path
            pkg_info = pkginfo_to_metadata(egginfo_path, egginfo_path)
            os.mkdir(distinfo_path)
        else:
            pkginfo_path = os.path.join(egginfo_path, 'PKG-INFO')
            pkg_info = pkginfo_to_metadata(egginfo_path, pkginfo_path)
            shutil.copytree(egginfo_path, distinfo_path,
                            ignore=lambda x, y: {'PKG-INFO', 'requires.txt', 'SOURCES.txt',
                                                 'not-zip-safe'}
                            )
            dependency_links_path = os.path.join(distinfo_path, 'dependency_links.txt')
            with open(dependency_links_path, 'r') as dependency_links_file:
                dependency_links = dependency_links_file.read().strip()
            if not dependency_links:
                adios(dependency_links_path)
        write_pkg_info(os.path.join(distinfo_path, 'METADATA'), pkg_info)
        for license_path in self.license_paths:
            filename = os.path.basename(license_path)
            shutil.copy(license_path, os.path.join(distinfo_path, filename))
        adios(egginfo_path)