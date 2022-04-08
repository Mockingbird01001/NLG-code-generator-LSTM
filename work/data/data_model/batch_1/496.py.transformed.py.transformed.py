
import os
import re
import fnmatch
__all__ = ["glob", "iglob", "escape"]
def glob(pathname, recursive=False):
    return list(iglob(pathname, recursive=recursive))
def iglob(pathname, recursive=False):
    it = _iglob(pathname, recursive)
    if recursive and _isrecursive(pathname):
        s = next(it)
        assert not s
    return it
def _iglob(pathname, recursive):
    dirname, basename = os.path.split(pathname)
    glob_in_dir = glob2 if recursive and _isrecursive(basename) else glob1
    if not has_magic(pathname):
        if basename:
            if os.path.lexists(pathname):
                yield pathname
        else:
            if os.path.isdir(dirname):
                yield pathname
        return
    if not dirname:
        yield from glob_in_dir(dirname, basename)
        return
    if dirname != pathname and has_magic(dirname):
        dirs = _iglob(dirname, recursive)
    else:
        dirs = [dirname]
    if not has_magic(basename):
        glob_in_dir = glob0
    for dirname in dirs:
        for name in glob_in_dir(dirname, basename):
            yield os.path.join(dirname, name)
def glob1(dirname, pattern):
    if not dirname:
        if isinstance(pattern, bytes):
            dirname = os.curdir.encode('ASCII')
        else:
            dirname = os.curdir
    try:
        names = os.listdir(dirname)
    except OSError:
        return []
    return fnmatch.filter(names, pattern)
def glob0(dirname, basename):
    if not basename:
        if os.path.isdir(dirname):
            return [basename]
    else:
        if os.path.lexists(os.path.join(dirname, basename)):
            return [basename]
    return []
def glob2(dirname, pattern):
    assert _isrecursive(pattern)
    yield pattern[:0]
    for x in _rlistdir(dirname):
        yield x
def _rlistdir(dirname):
    if not dirname:
        if isinstance(dirname, bytes):
            dirname = os.curdir.encode('ASCII')
        else:
            dirname = os.curdir
    try:
        names = os.listdir(dirname)
    except os.error:
        return
    for x in names:
        yield x
        path = os.path.join(dirname, x) if dirname else x
        for y in _rlistdir(path):
            yield os.path.join(x, y)
magic_check = re.compile('([*?[])')
magic_check_bytes = re.compile(b'([*?[])')
def has_magic(s):
    if isinstance(s, bytes):
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None
def _isrecursive(pattern):
    if isinstance(pattern, bytes):
        return pattern == b'**'
    else:
        return pattern == '**'
def escape(pathname):
    drive, pathname = os.path.splitdrive(pathname)
    if isinstance(pathname, bytes):
        pathname = magic_check_bytes.sub(br'[\1]', pathname)
    else:
        pathname = magic_check.sub(r'[\1]', pathname)
    return drive + pathname
