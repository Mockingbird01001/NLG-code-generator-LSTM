
import os
from distutils.errors import DistutilsFileError
def newer (source, target):
    if not os.path.exists(source):
        raise DistutilsFileError("file '%s' does not exist" %
                                 os.path.abspath(source))
    if not os.path.exists(target):
        return 1
    from stat import ST_MTIME
    mtime1 = os.stat(source)[ST_MTIME]
    mtime2 = os.stat(target)[ST_MTIME]
    return mtime1 > mtime2
def newer_pairwise (sources, targets):
    if len(sources) != len(targets):
        raise ValueError("'sources' and 'targets' must be same length")
    n_sources = []
    n_targets = []
    for i in range(len(sources)):
        if newer(sources[i], targets[i]):
            n_sources.append(sources[i])
            n_targets.append(targets[i])
    return (n_sources, n_targets)
def newer_group (sources, target, missing='error'):
    if not os.path.exists(target):
        return 1
    from stat import ST_MTIME
    target_mtime = os.stat(target)[ST_MTIME]
    for source in sources:
        if not os.path.exists(source):
            if missing == 'error':
                pass
            elif missing == 'ignore':
                continue
            elif missing == 'newer':
                return 1
        source_mtime = os.stat(source)[ST_MTIME]
        if source_mtime > target_mtime:
            return 1
    else:
        return 0
