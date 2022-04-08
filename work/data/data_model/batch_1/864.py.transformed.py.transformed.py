import os
import errno
import sys
from pip._vendor import six
def _makedirs_31(path, exist_ok=False):
    try:
        os.makedirs(path)
    except OSError as exc:
        if not exist_ok or exc.errno != errno.EEXIST:
            raise
needs_makedirs = (
    six.PY2 or
    (3, 4) <= sys.version_info < (3, 4, 1)
)
makedirs = _makedirs_31 if needs_makedirs else os.makedirs
