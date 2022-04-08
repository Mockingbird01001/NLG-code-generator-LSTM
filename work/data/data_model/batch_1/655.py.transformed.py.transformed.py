
import fnmatch
import logging
import os
import re
import sys
from . import DistlibException
from .compat import fsdecode
from .util import convert_path
__all__ = ['Manifest']
logger = logging.getLogger(__name__)
_COLLAPSE_PATTERN = re.compile('\\\\w*\n', re.M)
_PYTHON_VERSION = sys.version_info[:2]
class Manifest(object):
    def __init__(self, base=None):
        self.base = os.path.abspath(os.path.normpath(base or os.getcwd()))
        self.prefix = self.base + os.sep
        self.allfiles = None
        self.files = set()
    def findall(self):
        from stat import S_ISREG, S_ISDIR, S_ISLNK
        self.allfiles = allfiles = []
        root = self.base
        stack = [root]
        pop = stack.pop
        push = stack.append
        while stack:
            root = pop()
            names = os.listdir(root)
            for name in names:
                fullname = os.path.join(root, name)
                stat = os.stat(fullname)
                mode = stat.st_mode
                if S_ISREG(mode):
                    allfiles.append(fsdecode(fullname))
                elif S_ISDIR(mode) and not S_ISLNK(mode):
                    push(fullname)
    def add(self, item):
        if not item.startswith(self.prefix):
            item = os.path.join(self.base, item)
        self.files.add(os.path.normpath(item))
    def add_many(self, items):
        for item in items:
            self.add(item)
    def sorted(self, wantdirs=False):
        def add_dir(dirs, d):
            dirs.add(d)
            logger.debug('add_dir added %s', d)
            if d != self.base:
                parent, _ = os.path.split(d)
                assert parent not in ('', '/')
                add_dir(dirs, parent)
        result = set(self.files)
        if wantdirs:
            dirs = set()
            for f in result:
                add_dir(dirs, os.path.dirname(f))
            result |= dirs
        return [os.path.join(*path_tuple) for path_tuple in
                sorted(os.path.split(path) for path in result)]
    def clear(self):
        self.files = set()
        self.allfiles = []
    def process_directive(self, directive):
        action, patterns, thedir, dirpattern = self._parse_directive(directive)
        if action == 'include':
            for pattern in patterns:
                if not self._include_pattern(pattern, anchor=True):
                    logger.warning('no files found matching %r', pattern)
        elif action == 'exclude':
            for pattern in patterns:
                found = self._exclude_pattern(pattern, anchor=True)
        elif action == 'global-include':
            for pattern in patterns:
                if not self._include_pattern(pattern, anchor=False):
                    logger.warning('no files found matching %r '
                                   'anywhere in distribution', pattern)
        elif action == 'global-exclude':
            for pattern in patterns:
                found = self._exclude_pattern(pattern, anchor=False)
        elif action == 'recursive-include':
            for pattern in patterns:
                if not self._include_pattern(pattern, prefix=thedir):
                    logger.warning('no files found matching %r '
                                   'under directory %r', pattern, thedir)
        elif action == 'recursive-exclude':
            for pattern in patterns:
                found = self._exclude_pattern(pattern, prefix=thedir)
        elif action == 'graft':
            if not self._include_pattern(None, prefix=dirpattern):
                logger.warning('no directories found matching %r',
                               dirpattern)
        elif action == 'prune':
            if not self._exclude_pattern(None, prefix=dirpattern):
                logger.warning('no previously-included directories found '
                               'matching %r', dirpattern)
        else:
            raise DistlibException(
                'invalid action %r' % action)
    def _parse_directive(self, directive):
        words = directive.split()
        if len(words) == 1 and words[0] not in ('include', 'exclude',
                                                'global-include',
                                                'global-exclude',
                                                'recursive-include',
                                                'recursive-exclude',
                                                'graft', 'prune'):
            words.insert(0, 'include')
        action = words[0]
        patterns = thedir = dir_pattern = None
        if action in ('include', 'exclude',
                      'global-include', 'global-exclude'):
            if len(words) < 2:
                raise DistlibException(
                    '%r expects <pattern1> <pattern2> ...' % action)
            patterns = [convert_path(word) for word in words[1:]]
        elif action in ('recursive-include', 'recursive-exclude'):
            if len(words) < 3:
                raise DistlibException(
                    '%r expects <dir> <pattern1> <pattern2> ...' % action)
            thedir = convert_path(words[1])
            patterns = [convert_path(word) for word in words[2:]]
        elif action in ('graft', 'prune'):
            if len(words) != 2:
                raise DistlibException(
                    '%r expects a single <dir_pattern>' % action)
            dir_pattern = convert_path(words[1])
        else:
            raise DistlibException('unknown action %r' % action)
        return action, patterns, thedir, dir_pattern
    def _include_pattern(self, pattern, anchor=True, prefix=None,
                         is_regex=False):
        found = False
        pattern_re = self._translate_pattern(pattern, anchor, prefix, is_regex)
        if self.allfiles is None:
            self.findall()
        for name in self.allfiles:
            if pattern_re.search(name):
                self.files.add(name)
                found = True
        return found
    def _exclude_pattern(self, pattern, anchor=True, prefix=None,
                         is_regex=False):
        found = False
        pattern_re = self._translate_pattern(pattern, anchor, prefix, is_regex)
        for f in list(self.files):
            if pattern_re.search(f):
                self.files.remove(f)
                found = True
        return found
    def _translate_pattern(self, pattern, anchor=True, prefix=None,
                           is_regex=False):
        if is_regex:
            if isinstance(pattern, str):
                return re.compile(pattern)
            else:
                return pattern
        if _PYTHON_VERSION > (3, 2):
            start, _, end = self._glob_to_re('_').partition('_')
        if pattern:
            pattern_re = self._glob_to_re(pattern)
            if _PYTHON_VERSION > (3, 2):
                assert pattern_re.startswith(start) and pattern_re.endswith(end)
        else:
            pattern_re = ''
        base = re.escape(os.path.join(self.base, ''))
        if prefix is not None:
            if _PYTHON_VERSION <= (3, 2):
                empty_pattern = self._glob_to_re('')
                prefix_re = self._glob_to_re(prefix)[:-len(empty_pattern)]
            else:
                prefix_re = self._glob_to_re(prefix)
                assert prefix_re.startswith(start) and prefix_re.endswith(end)
                prefix_re = prefix_re[len(start): len(prefix_re) - len(end)]
            sep = os.sep
            if os.sep == '\\':
                sep = r'\\'
            if _PYTHON_VERSION <= (3, 2):
                pattern_re = '^' + base + sep.join((prefix_re,
                                                    '.*' + pattern_re))
            else:
                pattern_re = pattern_re[len(start): len(pattern_re) - len(end)]
                pattern_re = r'%s%s%s%s.*%s%s' % (start, base, prefix_re, sep,
                                                  pattern_re, end)
        else:
            if anchor:
                if _PYTHON_VERSION <= (3, 2):
                    pattern_re = '^' + base + pattern_re
                else:
                    pattern_re = r'%s%s%s' % (start, base, pattern_re[len(start):])
        return re.compile(pattern_re)
    def _glob_to_re(self, pattern):
        pattern_re = fnmatch.translate(pattern)
        sep = os.sep
        if os.sep == '\\':
            sep = r'\\\\'
        escaped = r'\1[^%s]' % sep
        pattern_re = re.sub(r'((?<!\\)(\\\\)*)\.', escaped, pattern_re)
        return pattern_re
