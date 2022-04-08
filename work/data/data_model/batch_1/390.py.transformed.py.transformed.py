
import os, re
import fnmatch
import functools
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsInternalError
from distutils import log
class FileList:
    def __init__(self, warn=None, debug_print=None):
        self.allfiles = None
        self.files = []
    def set_allfiles(self, allfiles):
        self.allfiles = allfiles
    def findall(self, dir=os.curdir):
        self.allfiles = findall(dir)
    def debug_print(self, msg):
        from distutils.debug import DEBUG
        if DEBUG:
            print(msg)
    def append(self, item):
        self.files.append(item)
    def extend(self, items):
        self.files.extend(items)
    def sort(self):
        sortable_files = sorted(map(os.path.split, self.files))
        self.files = []
        for sort_tuple in sortable_files:
            self.files.append(os.path.join(*sort_tuple))
    def remove_duplicates(self):
        for i in range(len(self.files) - 1, 0, -1):
            if self.files[i] == self.files[i - 1]:
                del self.files[i]
    def _parse_template_line(self, line):
        words = line.split()
        action = words[0]
        patterns = dir = dir_pattern = None
        if action in ('include', 'exclude',
                      'global-include', 'global-exclude'):
            if len(words) < 2:
                raise DistutilsTemplateError(
                      "'%s' expects <pattern1> <pattern2> ..." % action)
            patterns = [convert_path(w) for w in words[1:]]
        elif action in ('recursive-include', 'recursive-exclude'):
            if len(words) < 3:
                raise DistutilsTemplateError(
                      "'%s' expects <dir> <pattern1> <pattern2> ..." % action)
            dir = convert_path(words[1])
            patterns = [convert_path(w) for w in words[2:]]
        elif action in ('graft', 'prune'):
            if len(words) != 2:
                raise DistutilsTemplateError(
                      "'%s' expects a single <dir_pattern>" % action)
            dir_pattern = convert_path(words[1])
        else:
            raise DistutilsTemplateError("unknown action '%s'" % action)
        return (action, patterns, dir, dir_pattern)
    def process_template_line(self, line):
        (action, patterns, dir, dir_pattern) = self._parse_template_line(line)
        if action == 'include':
            self.debug_print("include " + ' '.join(patterns))
            for pattern in patterns:
                if not self.include_pattern(pattern, anchor=1):
                    log.warn("warning: no files found matching '%s'",
                             pattern)
        elif action == 'exclude':
            self.debug_print("exclude " + ' '.join(patterns))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, anchor=1):
                    log.warn(("warning: no previously-included files "
                              "found matching '%s'"), pattern)
        elif action == 'global-include':
            self.debug_print("global-include " + ' '.join(patterns))
            for pattern in patterns:
                if not self.include_pattern(pattern, anchor=0):
                    log.warn(("warning: no files found matching '%s' "
                              "anywhere in distribution"), pattern)
        elif action == 'global-exclude':
            self.debug_print("global-exclude " + ' '.join(patterns))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, anchor=0):
                    log.warn(("warning: no previously-included files matching "
                              "'%s' found anywhere in distribution"),
                             pattern)
        elif action == 'recursive-include':
            self.debug_print("recursive-include %s %s" %
                             (dir, ' '.join(patterns)))
            for pattern in patterns:
                if not self.include_pattern(pattern, prefix=dir):
                    log.warn(("warning: no files found matching '%s' "
                                "under directory '%s'"),
                             pattern, dir)
        elif action == 'recursive-exclude':
            self.debug_print("recursive-exclude %s %s" %
                             (dir, ' '.join(patterns)))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, prefix=dir):
                    log.warn(("warning: no previously-included files matching "
                              "'%s' found under directory '%s'"),
                             pattern, dir)
        elif action == 'graft':
            self.debug_print("graft " + dir_pattern)
            if not self.include_pattern(None, prefix=dir_pattern):
                log.warn("warning: no directories found matching '%s'",
                         dir_pattern)
        elif action == 'prune':
            self.debug_print("prune " + dir_pattern)
            if not self.exclude_pattern(None, prefix=dir_pattern):
                log.warn(("no previously-included directories found "
                          "matching '%s'"), dir_pattern)
        else:
            raise DistutilsInternalError(
                  "this cannot happen: invalid action '%s'" % action)
    def include_pattern(self, pattern, anchor=1, prefix=None, is_regex=0):
        files_found = False
        pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
        self.debug_print("include_pattern: applying regex r'%s'" %
                         pattern_re.pattern)
        if self.allfiles is None:
            self.findall()
        for name in self.allfiles:
            if pattern_re.search(name):
                self.debug_print(" adding " + name)
                self.files.append(name)
                files_found = True
        return files_found
    def exclude_pattern (self, pattern,
                         anchor=1, prefix=None, is_regex=0):
        files_found = False
        pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
        self.debug_print("exclude_pattern: applying regex r'%s'" %
                         pattern_re.pattern)
        for i in range(len(self.files)-1, -1, -1):
            if pattern_re.search(self.files[i]):
                self.debug_print(" removing " + self.files[i])
                del self.files[i]
                files_found = True
        return files_found
def _find_all_simple(path):
    results = (
        os.path.join(base, file)
        for base, dirs, files in os.walk(path, followlinks=True)
        for file in files
    )
    return filter(os.path.isfile, results)
def findall(dir=os.curdir):
    files = _find_all_simple(dir)
    if dir == os.curdir:
        make_rel = functools.partial(os.path.relpath, start=dir)
        files = map(make_rel, files)
    return list(files)
def glob_to_re(pattern):
    pattern_re = fnmatch.translate(pattern)
    sep = os.sep
    if os.sep == '\\':
        sep = r'\\\\'
    escaped = r'\1[^%s]' % sep
    pattern_re = re.sub(r'((?<!\\)(\\\\)*)\.', escaped, pattern_re)
    return pattern_re
def translate_pattern(pattern, anchor=1, prefix=None, is_regex=0):
    if is_regex:
        if isinstance(pattern, str):
            return re.compile(pattern)
        else:
            return pattern
    start, _, end = glob_to_re('_').partition('_')
    if pattern:
        pattern_re = glob_to_re(pattern)
        assert pattern_re.startswith(start) and pattern_re.endswith(end)
    else:
        pattern_re = ''
    if prefix is not None:
        prefix_re = glob_to_re(prefix)
        assert prefix_re.startswith(start) and prefix_re.endswith(end)
        prefix_re = prefix_re[len(start): len(prefix_re) - len(end)]
        sep = os.sep
        if os.sep == '\\':
            sep = r'\\'
        pattern_re = pattern_re[len(start): len(pattern_re) - len(end)]
        pattern_re = r'%s\A%s%s.*%s%s' % (start, prefix_re, sep, pattern_re, end)
    else:
        if anchor:
            pattern_re = r'%s\A%s' % (start, pattern_re[len(start):])
    return re.compile(pattern_re)
