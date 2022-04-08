
from typing import Tuple
from pip._internal.utils.misc import splitext
WHEEL_EXTENSION = ".whl"
BZ2_EXTENSIONS = (".tar.bz2", ".tbz")
XZ_EXTENSIONS = (
    ".tar.xz",
    ".txz",
    ".tlz",
    ".tar.lz",
    ".tar.lzma",
)
ZIP_EXTENSIONS = (".zip", WHEEL_EXTENSION)
TAR_EXTENSIONS = (".tar.gz", ".tgz", ".tar")
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + BZ2_EXTENSIONS + TAR_EXTENSIONS + XZ_EXTENSIONS
def is_archive_file(name):
    ext = splitext(name)[1].lower()
    if ext in ARCHIVE_EXTENSIONS:
        return True
    return False
