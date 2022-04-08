
import logging
import os
import shutil
import sys
import uuid
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from pip._vendor import pkg_resources, six
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.pep517.wrappers import Pep517HookCaller
from pip._vendor.pkg_resources import Distribution
from pip._internal.build_env import BuildEnvironment, NoOpBuildEnvironment
from pip._internal.exceptions import InstallationError
from pip._internal.locations import get_scheme
from pip._internal.models.link import Link
from pip._internal.operations.build.metadata import generate_metadata
from pip._internal.operations.build.metadata_legacy import (
    generate_metadata as generate_metadata_legacy,
)
from pip._internal.operations.install.editable_legacy import (
    install_editable as install_editable_legacy,
)
from pip._internal.operations.install.legacy import LegacyInstallFailure
from pip._internal.operations.install.legacy import install as install_legacy
from pip._internal.operations.install.wheel import install_wheel
from pip._internal.pyproject import load_pyproject_toml, make_pyproject_path
from pip._internal.req.req_uninstall import UninstallPathSet
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import (
    ask_path_exists,
    backup_dir,
    display_path,
    dist_in_site_packages,
    dist_in_usersite,
    get_distribution,
    hide_url,
    redact_auth_from_url,
)
from pip._internal.utils.packaging import get_metadata
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.virtualenv import running_under_virtualenv
from pip._internal.vcs import vcs
logger = logging.getLogger(__name__)
def _get_dist(metadata_directory):
    dist_dir = metadata_directory.rstrip(os.sep)
    base_dir, dist_dir_name = os.path.split(dist_dir)
    metadata = pkg_resources.PathMetadata(base_dir, dist_dir)
    if dist_dir.endswith(".egg-info"):
        dist_cls = pkg_resources.Distribution
        dist_name = os.path.splitext(dist_dir_name)[0]
    else:
        assert dist_dir.endswith(".dist-info")
        dist_cls = pkg_resources.DistInfoDistribution
        dist_name = os.path.splitext(dist_dir_name)[0].split("-")[0]
    return dist_cls(
        base_dir,
        project_name=dist_name,
        metadata=metadata,
    )
class InstallRequirement:
    def __init__(
        self,
        req,
        comes_from,
        editable=False,
        link=None,
        markers=None,
        use_pep517=None,
        isolated=False,
        install_options=None,
        global_options=None,
        hash_options=None,
        constraint=False,
        extras=(),
        user_supplied=False,
    ):
        assert req is None or isinstance(req, Requirement), req
        self.req = req
        self.comes_from = comes_from
        self.constraint = constraint
        self.editable = editable
        self.legacy_install_reason = None
        self.source_dir = None
        if self.editable:
            assert link
            if link.is_file:
                self.source_dir = os.path.normpath(
                    os.path.abspath(link.file_path)
                )
        if link is None and req and req.url:
            link = Link(req.url)
        self.link = self.original_link = link
        self.original_link_is_in_wheel_cache = False
        self.local_file_path = None
        if self.link and self.link.is_file:
            self.local_file_path = self.link.file_path
        if extras:
            self.extras = extras
        elif req:
            self.extras = {
                pkg_resources.safe_extra(extra) for extra in req.extras
            }
        else:
            self.extras = set()
        if markers is None and req:
            markers = req.marker
        self.markers = markers
        self.satisfied_by = None
        self.should_reinstall = False
        self._temp_build_dir = None
        self.install_succeeded = None
        self.install_options = install_options if install_options else []
        self.global_options = global_options if global_options else []
        self.hash_options = hash_options if hash_options else {}
        self.prepared = False
        self.user_supplied = user_supplied
        self.isolated = isolated
        self.build_env = NoOpBuildEnvironment()
        self.metadata_directory = None
        self.pyproject_requires = None
        self.requirements_to_check = []
        self.pep517_backend = None
        self.use_pep517 = use_pep517
        self.needs_more_preparation = False
    def __str__(self):
        if self.req:
            s = str(self.req)
            if self.link:
                s += ' from {}'.format(redact_auth_from_url(self.link.url))
        elif self.link:
            s = redact_auth_from_url(self.link.url)
        else:
            s = '<InstallRequirement>'
        if self.satisfied_by is not None:
            s += ' in {}'.format(display_path(self.satisfied_by.location))
        if self.comes_from:
            if isinstance(self.comes_from, str):
                comes_from = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            if comes_from:
                s += f' (from {comes_from})'
        return s
    def __repr__(self):
        return '<{} object: {} editable={!r}>'.format(
            self.__class__.__name__, str(self), self.editable)
    def format_debug(self):
        attributes = vars(self)
        names = sorted(attributes)
        state = (
            "{}={!r}".format(attr, attributes[attr]) for attr in sorted(names)
        )
        return '<{name} object: {{{state}}}>'.format(
            name=self.__class__.__name__,
            state=", ".join(state),
        )
    @property
    def name(self):
        if self.req is None:
            return None
        return pkg_resources.safe_name(self.req.name)
    @property
    def specifier(self):
        return self.req.specifier
    @property
    def is_pinned(self):
        specifiers = self.specifier
        return (len(specifiers) == 1 and
                next(iter(specifiers)).operator in {'==', '==='})
    def match_markers(self, extras_requested=None):
        if not extras_requested:
            extras_requested = ('',)
        if self.markers is not None:
            return any(
                self.markers.evaluate({'extra': extra})
                for extra in extras_requested)
        else:
            return True
    @property
    def has_hash_options(self):
        return bool(self.hash_options)
    def hashes(self, trust_internet=True):
        good_hashes = self.hash_options.copy()
        link = self.link if trust_internet else self.original_link
        if link and link.hash:
            good_hashes.setdefault(link.hash_name, []).append(link.hash)
        return Hashes(good_hashes)
    def from_path(self):
        if self.req is None:
            return None
        s = str(self.req)
        if self.comes_from:
            if isinstance(self.comes_from, str):
                comes_from = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            if comes_from:
                s += '->' + comes_from
        return s
    def ensure_build_location(self, build_dir, autodelete, parallel_builds):
        assert build_dir is not None
        if self._temp_build_dir is not None:
            assert self._temp_build_dir.path
            return self._temp_build_dir.path
        if self.req is None:
            self._temp_build_dir = TempDirectory(
                kind=tempdir_kinds.REQ_BUILD, globally_managed=True
            )
            return self._temp_build_dir.path
        dir_name = canonicalize_name(self.name)
        if parallel_builds:
            dir_name = f"{dir_name}_{uuid.uuid4().hex}"
        if not os.path.exists(build_dir):
            logger.debug('Creating directory %s', build_dir)
            os.makedirs(build_dir)
        actual_build_dir = os.path.join(build_dir, dir_name)
        delete_arg = None if autodelete else False
        return TempDirectory(
            path=actual_build_dir,
            delete=delete_arg,
            kind=tempdir_kinds.REQ_BUILD,
            globally_managed=True,
        ).path
    def _set_requirement(self):
        assert self.req is None
        assert self.metadata is not None
        assert self.source_dir is not None
        if isinstance(parse_version(self.metadata["Version"]), Version):
            op = "=="
        else:
            op = "==="
        self.req = Requirement(
            "".join([
                self.metadata["Name"],
                op,
                self.metadata["Version"],
            ])
        )
    def warn_on_mismatching_name(self):
        metadata_name = canonicalize_name(self.metadata["Name"])
        if canonicalize_name(self.req.name) == metadata_name:
            return
        logger.warning(
            'Generating metadata for package %s '
            'produced metadata for project name %s. Fix your '
            self.name, metadata_name, self.name
        )
        self.req = Requirement(metadata_name)
    def check_if_exists(self, use_user_site):
        if self.req is None:
            return
        existing_dist = get_distribution(self.req.name)
        if not existing_dist:
            return
        existing_version = existing_dist.version
        version_compatible = (
            existing_version is not None and
            self.req.specifier.contains(existing_version, prereleases=True)
        )
        if not version_compatible:
            self.satisfied_by = None
            if use_user_site:
                if dist_in_usersite(existing_dist):
                    self.should_reinstall = True
                elif (running_under_virtualenv() and
                        dist_in_site_packages(existing_dist)):
                    raise InstallationError(
                        "Will not install to the user site because it will "
                        "lack sys.path precedence to {} in {}".format(
                            existing_dist.project_name, existing_dist.location)
                    )
            else:
                self.should_reinstall = True
        else:
            if self.editable:
                self.should_reinstall = True
                self.satisfied_by = None
            else:
                self.satisfied_by = existing_dist
    @property
    def is_wheel(self):
        if not self.link:
            return False
        return self.link.is_wheel
    @property
    def unpacked_source_directory(self):
        return os.path.join(
            self.source_dir,
            self.link and self.link.subdirectory_fragment or '')
    @property
    def setup_py_path(self):
        assert self.source_dir, f"No source dir for {self}"
        setup_py = os.path.join(self.unpacked_source_directory, 'setup.py')
        return setup_py
    @property
    def pyproject_toml_path(self):
        assert self.source_dir, f"No source dir for {self}"
        return make_pyproject_path(self.unpacked_source_directory)
    def load_pyproject_toml(self):
        pyproject_toml_data = load_pyproject_toml(
            self.use_pep517,
            self.pyproject_toml_path,
            self.setup_py_path,
            str(self)
        )
        if pyproject_toml_data is None:
            self.use_pep517 = False
            return
        self.use_pep517 = True
        requires, backend, check, backend_path = pyproject_toml_data
        self.requirements_to_check = check
        self.pyproject_requires = requires
        self.pep517_backend = Pep517HookCaller(
            self.unpacked_source_directory, backend, backend_path=backend_path,
        )
    def _check_setup_py_or_cfg_exists(self) -> bool:
        if os.path.exists(self.setup_py_path):
            return True
        stem, ext = os.path.splitext(self.setup_py_path)
        if ext == ".py" and os.path.exists(f"{stem}.cfg"):
            return True
        return False
    def _generate_metadata(self):
        if not self.use_pep517:
            assert self.unpacked_source_directory
            if not self._check_setup_py_or_cfg_exists():
                raise InstallationError(
                    f'File "setup.py" or "setup.cfg" not found for legacy '
                    f'project {self}.'
                )
            return generate_metadata_legacy(
                build_env=self.build_env,
                setup_py_path=self.setup_py_path,
                source_dir=self.unpacked_source_directory,
                isolated=self.isolated,
                details=self.name or f"from {self.link}"
            )
        assert self.pep517_backend is not None
        return generate_metadata(
            build_env=self.build_env,
            backend=self.pep517_backend,
        )
    def prepare_metadata(self):
        assert self.source_dir
        with indent_log():
            self.metadata_directory = self._generate_metadata()
        if not self.name:
            self._set_requirement()
        else:
            self.warn_on_mismatching_name()
        self.assert_source_matches_version()
    @property
    def metadata(self):
        if not hasattr(self, '_metadata'):
            self._metadata = get_metadata(self.get_dist())
        return self._metadata
    def get_dist(self):
        return _get_dist(self.metadata_directory)
    def assert_source_matches_version(self):
        assert self.source_dir
        version = self.metadata['version']
        if self.req.specifier and version not in self.req.specifier:
            logger.warning(
                'Requested %s, but installing version %s',
                self,
                version,
            )
        else:
            logger.debug(
                'Source in %s has version %s, which satisfies requirement %s',
                display_path(self.source_dir),
                version,
                self,
            )
    def ensure_has_source_dir(
        self,
        parent_dir,
        autodelete=False,
        parallel_builds=False,
    ):
        if self.source_dir is None:
            self.source_dir = self.ensure_build_location(
                parent_dir,
                autodelete=autodelete,
                parallel_builds=parallel_builds,
            )
    def update_editable(self):
        if not self.link:
            logger.debug(
                "Cannot update repository at %s; repository location is "
                "unknown",
                self.source_dir,
            )
            return
        assert self.editable
        assert self.source_dir
        if self.link.scheme == 'file':
            return
        vcs_backend = vcs.get_backend_for_scheme(self.link.scheme)
        assert vcs_backend, f"Unsupported VCS URL {self.link.url}"
        hidden_url = hide_url(self.link.url)
        vcs_backend.obtain(self.source_dir, url=hidden_url)
    def uninstall(self, auto_confirm=False, verbose=False):
        assert self.req
        dist = get_distribution(self.req.name)
        if not dist:
            logger.warning("Skipping %s as it is not installed.", self.name)
            return None
        logger.info('Found existing installation: %s', dist)
        uninstalled_pathset = UninstallPathSet.from_dist(dist)
        uninstalled_pathset.remove(auto_confirm, verbose)
        return uninstalled_pathset
    def _get_archive_name(self, path, parentdir, rootdir):
        def _clean_zip_name(name, prefix):
            assert name.startswith(prefix + os.path.sep), (
                f"name {name!r} doesn't start with prefix {prefix!r}"
            )
            name = name[len(prefix) + 1:]
            name = name.replace(os.path.sep, '/')
            return name
        path = os.path.join(parentdir, path)
        name = _clean_zip_name(path, rootdir)
        return self.name + '/' + name
    def archive(self, build_dir):
        assert self.source_dir
        if build_dir is None:
            return
        create_archive = True
        archive_name = '{}-{}.zip'.format(self.name, self.metadata["version"])
        archive_path = os.path.join(build_dir, archive_name)
        if os.path.exists(archive_path):
            response = ask_path_exists(
                'The file {} exists. (i)gnore, (w)ipe, '
                '(b)ackup, (a)bort '.format(
                    display_path(archive_path)),
                ('i', 'w', 'b', 'a'))
            if response == 'i':
                create_archive = False
            elif response == 'w':
                logger.warning('Deleting %s', display_path(archive_path))
                os.remove(archive_path)
            elif response == 'b':
                dest_file = backup_dir(archive_path)
                logger.warning(
                    'Backing up %s to %s',
                    display_path(archive_path),
                    display_path(dest_file),
                )
                shutil.move(archive_path, dest_file)
            elif response == 'a':
                sys.exit(-1)
        if not create_archive:
            return
        zip_output = zipfile.ZipFile(
            archive_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True,
        )
        with zip_output:
            dir = os.path.normcase(
                os.path.abspath(self.unpacked_source_directory)
            )
            for dirpath, dirnames, filenames in os.walk(dir):
                for dirname in dirnames:
                    dir_arcname = self._get_archive_name(
                        dirname, parentdir=dirpath, rootdir=dir,
                    )
                    zipdir = zipfile.ZipInfo(dir_arcname + '/')
                    zipdir.external_attr = 0x1ED << 16
                    zip_output.writestr(zipdir, '')
                for filename in filenames:
                    file_arcname = self._get_archive_name(
                        filename, parentdir=dirpath, rootdir=dir,
                    )
                    filename = os.path.join(dirpath, filename)
                    zip_output.write(filename, file_arcname)
        logger.info('Saved %s', display_path(archive_path))
    def install(
        self,
        install_options,
        global_options=None,
        root=None,
        home=None,
        prefix=None,
        warn_script_location=True,
        use_user_site=False,
        pycompile=True
    ):
        scheme = get_scheme(
            self.name,
            user=use_user_site,
            home=home,
            root=root,
            isolated=self.isolated,
            prefix=prefix,
        )
        global_options = global_options if global_options is not None else []
        if self.editable:
            install_editable_legacy(
                install_options,
                global_options,
                prefix=prefix,
                home=home,
                use_user_site=use_user_site,
                name=self.name,
                setup_py_path=self.setup_py_path,
                isolated=self.isolated,
                build_env=self.build_env,
                unpacked_source_directory=self.unpacked_source_directory,
            )
            self.install_succeeded = True
            return
        if self.is_wheel:
            assert self.local_file_path
            direct_url = None
            if self.original_link:
                direct_url = direct_url_from_link(
                    self.original_link,
                    self.source_dir,
                    self.original_link_is_in_wheel_cache,
                )
            install_wheel(
                self.name,
                self.local_file_path,
                scheme=scheme,
                req_description=str(self.req),
                pycompile=pycompile,
                warn_script_location=warn_script_location,
                direct_url=direct_url,
                requested=self.user_supplied,
            )
            self.install_succeeded = True
            return
        global_options = list(global_options) + self.global_options
        install_options = list(install_options) + self.install_options
        try:
            success = install_legacy(
                install_options=install_options,
                global_options=global_options,
                root=root,
                home=home,
                prefix=prefix,
                use_user_site=use_user_site,
                pycompile=pycompile,
                scheme=scheme,
                setup_py_path=self.setup_py_path,
                isolated=self.isolated,
                req_name=self.name,
                build_env=self.build_env,
                unpacked_source_directory=self.unpacked_source_directory,
                req_description=str(self.req),
            )
        except LegacyInstallFailure as exc:
            self.install_succeeded = False
            six.reraise(*exc.parent)
        except Exception:
            self.install_succeeded = True
            raise
        self.install_succeeded = success
        if success and self.legacy_install_reason == 8368:
            deprecated(
                reason=(
                    "{} was installed using the legacy 'setup.py install' "
                    "method, because a wheel could not be built for it.".
                    format(self.name)
                ),
                replacement="to fix the wheel build issue reported above",
                gone_in=None,
                issue=8368,
            )
def check_invalid_constraint_type(req):
    problem = ""
    if not req.name:
        problem = "Unnamed requirements are not allowed as constraints"
    elif req.editable:
        problem = "Editable requirements are not allowed as constraints"
    elif req.extras:
        problem = "Constraints cannot have extras"
    if problem:
        deprecated(
            reason=(
                "Constraints are only allowed to take the form of a package "
                "name and a version specifier. Other forms were originally "
                "permitted as an accident of the implementation, but were "
                "undocumented. The new implementation of the resolver no "
                "longer supports these forms."
            ),
            replacement=(
                "replacing the constraint with a requirement."
            ),
            gone_in=None,
            issue=8210
        )
    return problem
