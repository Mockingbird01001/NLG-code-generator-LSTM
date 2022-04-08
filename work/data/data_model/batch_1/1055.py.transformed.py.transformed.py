import logging
import os
import re
from typing import List, Optional, Tuple
from pip._internal.utils.misc import (
    HiddenText,
    display_path,
    is_console_interactive,
    split_auth_from_netloc,
)
from pip._internal.utils.subprocess import CommandArgs, make_command
from pip._internal.vcs.versioncontrol import (
    AuthInfo,
    RemoteNotFoundError,
    RevOptions,
    VersionControl,
    vcs,
)
logger = logging.getLogger(__name__)
_svn_xml_url_re = re.compile('url="([^"]+)"')
_svn_rev_re = re.compile(r'committed-rev="(\d+)"')
_svn_info_xml_rev_re = re.compile(r'\s*revision="(\d+)"')
_svn_info_xml_url_re = re.compile(r'<url>(.*)</url>')
class Subversion(VersionControl):
    name = 'svn'
    dirname = '.svn'
    repo_name = 'checkout'
    schemes = (
        'svn+ssh', 'svn+http', 'svn+https', 'svn+svn', 'svn+file'
    )
    @classmethod
    def should_add_vcs_url_prefix(cls, remote_url):
        return True
    @staticmethod
    def get_base_rev_args(rev):
        return ['-r', rev]
    @classmethod
    def get_revision(cls, location):
        revision = 0
        for base, dirs, _ in os.walk(location):
            if cls.dirname not in dirs:
                dirs[:] = []
                continue
            dirs.remove(cls.dirname)
            entries_fn = os.path.join(base, cls.dirname, 'entries')
            if not os.path.exists(entries_fn):
                continue
            dirurl, localrev = cls._get_svn_url_rev(base)
            if base == location:
                assert dirurl is not None
                base = dirurl + '/'
            elif not dirurl or not dirurl.startswith(base):
                dirs[:] = []
                continue
            revision = max(revision, localrev)
        return str(revision)
    @classmethod
    def get_netloc_and_auth(cls, netloc, scheme):
        if scheme == 'ssh':
            return super().get_netloc_and_auth(netloc, scheme)
        return split_auth_from_netloc(netloc)
    @classmethod
    def get_url_rev_and_auth(cls, url):
        url, rev, user_pass = super().get_url_rev_and_auth(url)
        if url.startswith('ssh://'):
            url = 'svn+' + url
        return url, rev, user_pass
    @staticmethod
    def make_rev_args(username, password):
        extra_args = []
        if username:
            extra_args += ['--username', username]
        if password:
            extra_args += ['--password', password]
        return extra_args
    @classmethod
    def get_remote_url(cls, location):
        orig_location = location
        while not os.path.exists(os.path.join(location, 'setup.py')):
            last_location = location
            location = os.path.dirname(location)
            if location == last_location:
                logger.warning(
                    "Could not find setup.py for directory %s (tried all "
                    "parent directories)",
                    orig_location,
                )
                raise RemoteNotFoundError
        url, _rev = cls._get_svn_url_rev(location)
        if url is None:
            raise RemoteNotFoundError
        return url
    @classmethod
    def _get_svn_url_rev(cls, location):
        from pip._internal.exceptions import InstallationError
        entries_path = os.path.join(location, cls.dirname, 'entries')
        if os.path.exists(entries_path):
            with open(entries_path) as f:
                data = f.read()
        else:
            data = ''
        url = None
        if (data.startswith('8') or
                data.startswith('9') or
                data.startswith('10')):
            entries = list(map(str.splitlines, data.split('\n\x0c\n')))
            del entries[0][0]
            url = entries[0][3]
            revs = [int(d[9]) for d in entries if len(d) > 9 and d[9]] + [0]
        elif data.startswith('<?xml'):
            match = _svn_xml_url_re.search(data)
            if not match:
                raise ValueError(f'Badly formatted data: {data!r}')
            url = match.group(1)
            revs = [int(m.group(1)) for m in _svn_rev_re.finditer(data)] + [0]
        else:
            try:
                xml = cls.run_command(
                    ['info', '--xml', location],
                    show_stdout=False,
                    stdout_only=True,
                )
                match = _svn_info_xml_url_re.search(xml)
                assert match is not None
                url = match.group(1)
                revs = [
                    int(m.group(1)) for m in _svn_info_xml_rev_re.finditer(xml)
                ]
            except InstallationError:
                url, revs = None, []
        if revs:
            rev = max(revs)
        else:
            rev = 0
        return url, rev
    @classmethod
    def is_commit_id_equal(cls, dest, name):
        return False
    def __init__(self, use_interactive=None):
        if use_interactive is None:
            use_interactive = is_console_interactive()
        self.use_interactive = use_interactive
        self._vcs_version = None
        super().__init__()
    def call_vcs_version(self):
        version_prefix = 'svn, version '
        version = self.run_command(
            ['--version'], show_stdout=False, stdout_only=True
        )
        if not version.startswith(version_prefix):
            return ()
        version = version[len(version_prefix):].split()[0]
        version_list = version.partition('-')[0].split('.')
        try:
            parsed_version = tuple(map(int, version_list))
        except ValueError:
            return ()
        return parsed_version
    def get_vcs_version(self):
        if self._vcs_version is not None:
            return self._vcs_version
        vcs_version = self.call_vcs_version()
        self._vcs_version = vcs_version
        return vcs_version
    def get_remote_call_options(self):
        if not self.use_interactive:
            return ['--non-interactive']
        svn_version = self.get_vcs_version()
        if svn_version >= (1, 8):
            return ['--force-interactive']
        return []
    def fetch_new(self, dest, url, rev_options):
        rev_display = rev_options.to_display()
        logger.info(
            'Checking out %s%s to %s',
            url,
            rev_display,
            display_path(dest),
        )
        cmd_args = make_command(
            'checkout', '-q', self.get_remote_call_options(),
            rev_options.to_args(), url, dest,
        )
        self.run_command(cmd_args)
    def switch(self, dest, url, rev_options):
        cmd_args = make_command(
            'switch', self.get_remote_call_options(), rev_options.to_args(),
            url, dest,
        )
        self.run_command(cmd_args)
    def update(self, dest, url, rev_options):
        cmd_args = make_command(
            'update', self.get_remote_call_options(), rev_options.to_args(),
            dest,
        )
        self.run_command(cmd_args)
vcs.register(Subversion)
