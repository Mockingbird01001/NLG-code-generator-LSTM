
"""
The ``distro`` package (``distro`` stands for Linux Distribution) provides
information about the Linux distribution it runs on, such as a reliable
machine-readable distro ID, or version information.
It is the recommended replacement for Python's original
:py:func:`platform.linux_distribution` function, but it provides much more
functionality. An alternative implementation became necessary because Python
3.5 deprecated this function, and Python 3.8 will remove it altogether.
Its predecessor function :py:func:`platform.dist` was already
deprecated since Python 2.6 and will also be removed in Python 3.8.
Still, there are many cases in which access to OS distribution information
is needed. See `Python issue 1322 <https://bugs.python.org/issue1322>`_ for
more information.
"""
import os
import re
import sys
import json
import shlex
import logging
import argparse
import subprocess
_UNIXCONFDIR = os.environ.get('UNIXCONFDIR', '/etc')
_OS_RELEASE_BASENAME = 'os-release'
NORMALIZED_OS_ID = {
    'ol': 'oracle',
}
NORMALIZED_LSB_ID = {
    'enterpriseenterpriseas': 'oracle',
    'enterpriseenterpriseserver': 'oracle',
    'redhatenterpriseworkstation': 'rhel',
    'redhatenterpriseserver': 'rhel',
    'redhatenterprisecomputenode': 'rhel',
}
NORMALIZED_DISTRO_ID = {
    'redhat': 'rhel',
}
_DISTRO_RELEASE_CONTENT_REVERSED_PATTERN = re.compile(
    r'(?:[^)]*\)(.*)\()? *(?:STL )?([\d.+\-a-z]*\d) *(?:esaeler *)?(.+)')
_DISTRO_RELEASE_BASENAME_PATTERN = re.compile(
    r'(\w+)[-_](release|version)$')
_DISTRO_RELEASE_IGNORE_BASENAMES = (
    'debian_version',
    'lsb-release',
    'oem-release',
    _OS_RELEASE_BASENAME,
    'system-release',
    'plesk-release',
)
def linux_distribution(full_distribution_name=True):
    return _distro.linux_distribution(full_distribution_name)
def id():
    return _distro.id()
def name(pretty=False):
    return _distro.name(pretty)
def version(pretty=False, best=False):
    return _distro.version(pretty, best)
def version_parts(best=False):
    return _distro.version_parts(best)
def major_version(best=False):
    return _distro.major_version(best)
def minor_version(best=False):
    return _distro.minor_version(best)
def build_number(best=False):
    return _distro.build_number(best)
def like():
    return _distro.like()
def codename():
    return _distro.codename()
def info(pretty=False, best=False):
    return _distro.info(pretty, best)
def os_release_info():
    return _distro.os_release_info()
def lsb_release_info():
    return _distro.lsb_release_info()
def distro_release_info():
    return _distro.distro_release_info()
def uname_info():
    return _distro.uname_info()
def os_release_attr(attribute):
    return _distro.os_release_attr(attribute)
def lsb_release_attr(attribute):
    return _distro.lsb_release_attr(attribute)
def distro_release_attr(attribute):
    return _distro.distro_release_attr(attribute)
def uname_attr(attribute):
    return _distro.uname_attr(attribute)
class cached_property(object):
    def __init__(self, f):
        self._fname = f.__name__
        self._f = f
    def __get__(self, obj, owner):
        assert obj is not None, 'call {} on an instance'.format(self._fname)
        ret = obj.__dict__[self._fname] = self._f(obj)
        return ret
class LinuxDistribution(object):
    def __init__(self,
                 include_lsb=True,
                 os_release_file='',
                 distro_release_file='',
                 include_uname=True):
        self.os_release_file = os_release_file or            os.path.join(_UNIXCONFDIR, _OS_RELEASE_BASENAME)
        self.distro_release_file = distro_release_file or ''
        self.include_lsb = include_lsb
        self.include_uname = include_uname
    def __repr__(self):
        return            "LinuxDistribution("            "os_release_file={self.os_release_file!r}, "            "distro_release_file={self.distro_release_file!r}, "            "include_lsb={self.include_lsb!r}, "            "include_uname={self.include_uname!r}, "            "_os_release_info={self._os_release_info!r}, "            "_lsb_release_info={self._lsb_release_info!r}, "            "_distro_release_info={self._distro_release_info!r}, "            "_uname_info={self._uname_info!r})".format(
                self=self)
    def linux_distribution(self, full_distribution_name=True):
        return (
            self.name() if full_distribution_name else self.id(),
            self.version(),
            self.codename()
        )
    def id(self):
        def normalize(distro_id, table):
            distro_id = distro_id.lower().replace(' ', '_')
            return table.get(distro_id, distro_id)
        distro_id = self.os_release_attr('id')
        if distro_id:
            return normalize(distro_id, NORMALIZED_OS_ID)
        distro_id = self.lsb_release_attr('distributor_id')
        if distro_id:
            return normalize(distro_id, NORMALIZED_LSB_ID)
        distro_id = self.distro_release_attr('id')
        if distro_id:
            return normalize(distro_id, NORMALIZED_DISTRO_ID)
        distro_id = self.uname_attr('id')
        if distro_id:
            return normalize(distro_id, NORMALIZED_DISTRO_ID)
        return ''
    def name(self, pretty=False):
        name = self.os_release_attr('name')            or self.lsb_release_attr('distributor_id')            or self.distro_release_attr('name')            or self.uname_attr('name')
        if pretty:
            name = self.os_release_attr('pretty_name')                or self.lsb_release_attr('description')
            if not name:
                name = self.distro_release_attr('name')                       or self.uname_attr('name')
                version = self.version(pretty=True)
                if version:
                    name = name + ' ' + version
        return name or ''
    def version(self, pretty=False, best=False):
        versions = [
            self.os_release_attr('version_id'),
            self.lsb_release_attr('release'),
            self.distro_release_attr('version_id'),
            self._parse_distro_release_content(
                self.os_release_attr('pretty_name')).get('version_id', ''),
            self._parse_distro_release_content(
                self.lsb_release_attr('description')).get('version_id', ''),
            self.uname_attr('release')
        ]
        version = ''
        if best:
            for v in versions:
                if v.count(".") > version.count(".") or version == '':
                    version = v
        else:
            for v in versions:
                if v != '':
                    version = v
                    break
        if pretty and version and self.codename():
            version = '{0} ({1})'.format(version, self.codename())
        return version
    def version_parts(self, best=False):
        version_str = self.version(best=best)
        if version_str:
            version_regex = re.compile(r'(\d+)\.?(\d+)?\.?(\d+)?')
            matches = version_regex.match(version_str)
            if matches:
                major, minor, build_number = matches.groups()
                return major, minor or '', build_number or ''
        return '', '', ''
    def major_version(self, best=False):
        return self.version_parts(best)[0]
    def minor_version(self, best=False):
        return self.version_parts(best)[1]
    def build_number(self, best=False):
        return self.version_parts(best)[2]
    def like(self):
        return self.os_release_attr('id_like') or ''
    def codename(self):
        try:
            return self._os_release_info['codename']
        except KeyError:
            return self.lsb_release_attr('codename')                or self.distro_release_attr('codename')                or ''
    def info(self, pretty=False, best=False):
        return dict(
            id=self.id(),
            version=self.version(pretty, best),
            version_parts=dict(
                major=self.major_version(best),
                minor=self.minor_version(best),
                build_number=self.build_number(best)
            ),
            like=self.like(),
            codename=self.codename(),
        )
    def os_release_info(self):
        return self._os_release_info
    def lsb_release_info(self):
        return self._lsb_release_info
    def distro_release_info(self):
        return self._distro_release_info
    def uname_info(self):
        return self._uname_info
    def os_release_attr(self, attribute):
        return self._os_release_info.get(attribute, '')
    def lsb_release_attr(self, attribute):
        return self._lsb_release_info.get(attribute, '')
    def distro_release_attr(self, attribute):
        return self._distro_release_info.get(attribute, '')
    def uname_attr(self, attribute):
        return self._uname_info.get(attribute, '')
    @cached_property
    def _os_release_info(self):
        if os.path.isfile(self.os_release_file):
            with open(self.os_release_file) as release_file:
                return self._parse_os_release_content(release_file)
        return {}
    @staticmethod
    def _parse_os_release_content(lines):
        props = {}
        lexer = shlex.shlex(lines, posix=True)
        lexer.whitespace_split = True
        if sys.version_info[0] == 2 and isinstance(lexer.wordchars, bytes):
            lexer.wordchars = lexer.wordchars.decode('iso-8859-1')
        tokens = list(lexer)
        for token in tokens:
            if '=' in token:
                k, v = token.split('=', 1)
                props[k.lower()] = v
            else:
                pass
        if 'version_codename' in props:
            props['codename'] = props['version_codename']
        elif 'ubuntu_codename' in props:
            props['codename'] = props['ubuntu_codename']
        elif 'version' in props:
            codename = re.search(r'(\(\D+\))|,(\s+)?\D+', props['version'])
            if codename:
                codename = codename.group()
                codename = codename.strip('()')
                codename = codename.strip(',')
                codename = codename.strip()
                props['codename'] = codename
        return props
    @cached_property
    def _lsb_release_info(self):
        if not self.include_lsb:
            return {}
        with open(os.devnull, 'w') as devnull:
            try:
                cmd = ('lsb_release', '-a')
                stdout = subprocess.check_output(cmd, stderr=devnull)
            except OSError:
                return {}
        content = self._to_str(stdout).splitlines()
        return self._parse_lsb_release_content(content)
    @staticmethod
    def _parse_lsb_release_content(lines):
        props = {}
        for line in lines:
            kv = line.strip('\n').split(':', 1)
            if len(kv) != 2:
                continue
            k, v = kv
            props.update({k.replace(' ', '_').lower(): v.strip()})
        return props
    @cached_property
    def _uname_info(self):
        with open(os.devnull, 'w') as devnull:
            try:
                cmd = ('uname', '-rs')
                stdout = subprocess.check_output(cmd, stderr=devnull)
            except OSError:
                return {}
        content = self._to_str(stdout).splitlines()
        return self._parse_uname_content(content)
    @staticmethod
    def _parse_uname_content(lines):
        props = {}
        match = re.search(r'^([^\s]+)\s+([\d\.]+)', lines[0].strip())
        if match:
            name, version = match.groups()
            if name == 'Linux':
                return {}
            props['id'] = name.lower()
            props['name'] = name
            props['release'] = version
        return props
    @staticmethod
    def _to_str(text):
        encoding = sys.getfilesystemencoding()
        encoding = 'utf-8' if encoding == 'ascii' else encoding
        if sys.version_info[0] >= 3:
            if isinstance(text, bytes):
                return text.decode(encoding)
        else:
            if isinstance(text, unicode):
                return text.encode(encoding)
        return text
    @cached_property
    def _distro_release_info(self):
        if self.distro_release_file:
            distro_info = self._parse_distro_release_file(
                self.distro_release_file)
            basename = os.path.basename(self.distro_release_file)
            match = _DISTRO_RELEASE_BASENAME_PATTERN.match(basename)
            if 'name' in distro_info               and 'cloudlinux' in distro_info['name'].lower():
                distro_info['id'] = 'cloudlinux'
            elif match:
                distro_info['id'] = match.group(1)
            return distro_info
        else:
            try:
                basenames = os.listdir(_UNIXCONFDIR)
                basenames.sort()
            except OSError:
                basenames = ['SuSE-release',
                             'arch-release',
                             'base-release',
                             'centos-release',
                             'fedora-release',
                             'gentoo-release',
                             'mageia-release',
                             'mandrake-release',
                             'mandriva-release',
                             'mandrivalinux-release',
                             'manjaro-release',
                             'oracle-release',
                             'redhat-release',
                             'sl-release',
                             'slackware-version']
            for basename in basenames:
                if basename in _DISTRO_RELEASE_IGNORE_BASENAMES:
                    continue
                match = _DISTRO_RELEASE_BASENAME_PATTERN.match(basename)
                if match:
                    filepath = os.path.join(_UNIXCONFDIR, basename)
                    distro_info = self._parse_distro_release_file(filepath)
                    if 'name' in distro_info:
                        self.distro_release_file = filepath
                        distro_info['id'] = match.group(1)
                        if 'cloudlinux' in distro_info['name'].lower():
                            distro_info['id'] = 'cloudlinux'
                        return distro_info
            return {}
    def _parse_distro_release_file(self, filepath):
        try:
            with open(filepath) as fp:
                return self._parse_distro_release_content(fp.readline())
        except (OSError, IOError):
            return {}
    @staticmethod
    def _parse_distro_release_content(line):
        matches = _DISTRO_RELEASE_CONTENT_REVERSED_PATTERN.match(
            line.strip()[::-1])
        distro_info = {}
        if matches:
            distro_info['name'] = matches.group(3)[::-1]
            if matches.group(2):
                distro_info['version_id'] = matches.group(2)[::-1]
            if matches.group(1):
                distro_info['codename'] = matches.group(1)[::-1]
        elif line:
            distro_info['name'] = line.strip()
        return distro_info
_distro = LinuxDistribution()
def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    parser = argparse.ArgumentParser(description="OS distro info tool")
    parser.add_argument(
        '--json',
        '-j',
        help="Output in machine readable format",
        action="store_true")
    args = parser.parse_args()
    if args.json:
        logger.info(json.dumps(info(), indent=4, sort_keys=True))
    else:
        logger.info('Name: %s', name(pretty=True))
        distribution_version = version(pretty=True)
        logger.info('Version: %s', distribution_version)
        distribution_codename = codename()
        logger.info('Codename: %s', distribution_codename)
if __name__ == '__main__':
    main()
