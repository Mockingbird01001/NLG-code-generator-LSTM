import logging
import os.path
import re
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple
from pip._vendor.packaging.version import _BaseVersion
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path, hide_url
from pip._internal.utils.subprocess import make_command
from pip._internal.vcs.versioncontrol import (
    AuthInfo,
    RemoteNotFoundError,
    RevOptions,
    VersionControl,
    find_path_to_setup_from_repo_root,
    vcs,
)
urlsplit = urllib.parse.urlsplit
urlunsplit = urllib.parse.urlunsplit
logger = logging.getLogger(__name__)
HASH_REGEX = re.compile('^[a-fA-F0-9]{40}$')
def looks_like_hash(sha):
    return bool(HASH_REGEX.match(sha))
class Git(VersionControl):
    name = 'git'
    dirname = '.git'
    repo_name = 'clone'
    schemes = (
        'git+http', 'git+https', 'git+ssh', 'git+git', 'git+file',
    )
    unset_environ = ('GIT_DIR', 'GIT_WORK_TREE')
    default_arg_rev = 'HEAD'
    @staticmethod
    def get_base_rev_args(rev):
        return [rev]
    def is_immutable_rev_checkout(self, url, dest):
        _, rev_options = self.get_url_rev_options(hide_url(url))
        if not rev_options.rev:
            return False
        if not self.is_commit_id_equal(dest, rev_options.rev):
            return False
        is_tag_or_branch = bool(
            self.get_revision_sha(dest, rev_options.rev)[0]
        )
        return not is_tag_or_branch
    def get_git_version(self):
        VERSION_PFX = 'git version '
        version = self.run_command(
            ['version'], show_stdout=False, stdout_only=True
        )
        if version.startswith(VERSION_PFX):
            version = version[len(VERSION_PFX):].split()[0]
        else:
            version = ''
        version = '.'.join(version.split('.')[:3])
        return parse_version(version)
    @classmethod
    def get_current_branch(cls, location):
        args = ['symbolic-ref', '-q', 'HEAD']
        output = cls.run_command(
            args,
            extra_ok_returncodes=(1, ),
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        )
        ref = output.strip()
        if ref.startswith('refs/heads/'):
            return ref[len('refs/heads/'):]
        return None
    @classmethod
    def get_revision_sha(cls, dest, rev):
        output = cls.run_command(
            ['show-ref', rev],
            cwd=dest,
            show_stdout=False,
            stdout_only=True,
            on_returncode='ignore',
        )
        refs = {}
        for line in output.strip().split("\n"):
            line = line.rstrip("\r")
            if not line:
                continue
            try:
                ref_sha, ref_name = line.split(" ", maxsplit=2)
            except ValueError:
                raise ValueError(f'unexpected show-ref line: {line!r}')
            refs[ref_name] = ref_sha
        branch_ref = f'refs/remotes/origin/{rev}'
        tag_ref = f'refs/tags/{rev}'
        sha = refs.get(branch_ref)
        if sha is not None:
            return (sha, True)
        sha = refs.get(tag_ref)
        return (sha, False)
    @classmethod
    def _should_fetch(cls, dest, rev):
        if rev.startswith("refs/"):
            return True
        if not looks_like_hash(rev):
            return False
        if cls.has_commit(dest, rev):
            return False
        return True
    @classmethod
    def resolve_revision(cls, dest, url, rev_options):
        rev = rev_options.arg_rev
        assert rev is not None
        sha, is_branch = cls.get_revision_sha(dest, rev)
        if sha is not None:
            rev_options = rev_options.make_new(sha)
            rev_options.branch_name = rev if is_branch else None
            return rev_options
        if not looks_like_hash(rev):
            logger.warning(
                "Did not find branch or tag '%s', assuming revision or ref.",
                rev,
            )
        if not cls._should_fetch(dest, rev):
            return rev_options
        cls.run_command(
            make_command('fetch', '-q', url, rev_options.to_args()),
            cwd=dest,
        )
        sha = cls.get_revision(dest, rev='FETCH_HEAD')
        rev_options = rev_options.make_new(sha)
        return rev_options
    @classmethod
    def is_commit_id_equal(cls, dest, name):
        if not name:
            return False
        return cls.get_revision(dest) == name
    def fetch_new(self, dest, url, rev_options):
        rev_display = rev_options.to_display()
        logger.info('Cloning %s%s to %s', url, rev_display, display_path(dest))
        self.run_command(make_command('clone', '-q', url, dest))
        if rev_options.rev:
            rev_options = self.resolve_revision(dest, url, rev_options)
            branch_name = getattr(rev_options, 'branch_name', None)
            if branch_name is None:
                if not self.is_commit_id_equal(dest, rev_options.rev):
                    cmd_args = make_command(
                        'checkout', '-q', rev_options.to_args(),
                    )
                    self.run_command(cmd_args, cwd=dest)
            elif self.get_current_branch(dest) != branch_name:
                track_branch = f'origin/{branch_name}'
                cmd_args = [
                    'checkout', '-b', branch_name, '--track', track_branch,
                ]
                self.run_command(cmd_args, cwd=dest)
        self.update_submodules(dest)
    def switch(self, dest, url, rev_options):
        self.run_command(
            make_command('config', 'remote.origin.url', url),
            cwd=dest,
        )
        cmd_args = make_command('checkout', '-q', rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)
        self.update_submodules(dest)
    def update(self, dest, url, rev_options):
        if self.get_git_version() >= parse_version('1.9.0'):
            self.run_command(['fetch', '-q', '--tags'], cwd=dest)
        else:
            self.run_command(['fetch', '-q'], cwd=dest)
        rev_options = self.resolve_revision(dest, url, rev_options)
        cmd_args = make_command('reset', '--hard', '-q', rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)
        self.update_submodules(dest)
    @classmethod
    def get_remote_url(cls, location):
        stdout = cls.run_command(
            ['config', '--get-regexp', r'remote\..*\.url'],
            extra_ok_returncodes=(1, ),
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        )
        remotes = stdout.splitlines()
        try:
            found_remote = remotes[0]
        except IndexError:
            raise RemoteNotFoundError
        for remote in remotes:
            if remote.startswith('remote.origin.url '):
                found_remote = remote
                break
        url = found_remote.split(' ')[1]
        return url.strip()
    @classmethod
    def has_commit(cls, location, rev):
        try:
            cls.run_command(
                ['rev-parse', '-q', '--verify', "sha^" + rev],
                cwd=location,
                log_failed_cmd=False,
            )
        except InstallationError:
            return False
        else:
            return True
    @classmethod
    def get_revision(cls, location, rev=None):
        if rev is None:
            rev = 'HEAD'
        current_rev = cls.run_command(
            ['rev-parse', rev],
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        )
        return current_rev.strip()
    @classmethod
    def get_subdirectory(cls, location):
        git_dir = cls.run_command(
            ['rev-parse', '--git-dir'],
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        ).strip()
        if not os.path.isabs(git_dir):
            git_dir = os.path.join(location, git_dir)
        repo_root = os.path.abspath(os.path.join(git_dir, '..'))
        return find_path_to_setup_from_repo_root(location, repo_root)
    @classmethod
    def get_url_rev_and_auth(cls, url):
        scheme, netloc, path, query, fragment = urlsplit(url)
        if scheme.endswith('file'):
            initial_slashes = path[:-len(path.lstrip('/'))]
            newpath = (
                initial_slashes +
                urllib.request.url2pathname(path)
                .replace('\\', '/').lstrip('/')
            )
            after_plus = scheme.find('+') + 1
            url = scheme[:after_plus] + urlunsplit(
                (scheme[after_plus:], netloc, newpath, query, fragment),
            )
        if '://' not in url:
            assert 'file:' not in url
            url = url.replace('git+', 'git+ssh://')
            url, rev, user_pass = super().get_url_rev_and_auth(url)
            url = url.replace('ssh://', '')
        else:
            url, rev, user_pass = super().get_url_rev_and_auth(url)
        return url, rev, user_pass
    @classmethod
    def update_submodules(cls, location):
        if not os.path.exists(os.path.join(location, '.gitmodules')):
            return
        cls.run_command(
            ['submodule', 'update', '--init', '--recursive', '-q'],
            cwd=location,
        )
    @classmethod
    def get_repository_root(cls, location):
        loc = super().get_repository_root(location)
        if loc:
            return loc
        try:
            r = cls.run_command(
                ['rev-parse', '--show-toplevel'],
                cwd=location,
                show_stdout=False,
                stdout_only=True,
                on_returncode='raise',
                log_failed_cmd=False,
            )
        except BadCommand:
            logger.debug("could not determine if %s is under git control "
                         "because git is not available", location)
            return None
        except InstallationError:
            return None
        return os.path.normpath(r.rstrip('\r\n'))
vcs.register(Git)
