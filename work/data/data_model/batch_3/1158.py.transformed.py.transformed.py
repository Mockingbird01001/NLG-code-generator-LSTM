
import logging
import os
import shutil
import stat
import tarfile
import zipfile
from typing import Iterable, List, Optional
from zipfile import ZipInfo
from pip._internal.exceptions import InstallationError
from pip._internal.utils.filetypes import (
    BZ2_EXTENSIONS,
    TAR_EXTENSIONS,
    XZ_EXTENSIONS,
    ZIP_EXTENSIONS,
)
from pip._internal.utils.misc import ensure_dir
logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = ZIP_EXTENSIONS + TAR_EXTENSIONS
try:
    import bz2
    SUPPORTED_EXTENSIONS += BZ2_EXTENSIONS
except ImportError:
    logger.debug("bz2 module is not available")
try:
    import lzma
    SUPPORTED_EXTENSIONS += XZ_EXTENSIONS
except ImportError:
    logger.debug("lzma module is not available")
def current_umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask
def split_leading_dir(path):
    path = path.lstrip("/").lstrip("\\")
    if "/" in path and (
        ("\\" in path and path.find("/") < path.find("\\")) or "\\" not in path
    ):
        return path.split("/", 1)
    elif "\\" in path:
        return path.split("\\", 1)
    else:
        return [path, ""]
def has_leading_dir(paths):
    common_prefix = None
    for path in paths:
        prefix, rest = split_leading_dir(path)
        if not prefix:
            return False
        elif common_prefix is None:
            common_prefix = prefix
        elif prefix != common_prefix:
            return False
    return True
def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory
def set_extracted_file_to_default_mode_plus_executable(path):
    os.chmod(path, (0o777 & ~current_umask() | 0o111))
def zip_item_is_executable(info):
    mode = info.external_attr >> 16
    return bool(mode and stat.S_ISREG(mode) and mode & 0o111)
def unzip_file(filename, location, flatten=True):
    ensure_dir(location)
    zipfp = open(filename, "rb")
    try:
        zip = zipfile.ZipFile(zipfp, allowZip64=True)
        leading = has_leading_dir(zip.namelist()) and flatten
        for info in zip.infolist():
            name = info.filename
            fn = name
            if leading:
                fn = split_leading_dir(name)[1]
            fn = os.path.join(location, fn)
            dir = os.path.dirname(fn)
            if not is_within_directory(location, fn):
                message = (
                    "The zip file ({}) has a file ({}) trying to install "
                    "outside target directory ({})"
                )
                raise InstallationError(message.format(filename, fn, location))
            if fn.endswith("/") or fn.endswith("\\"):
                ensure_dir(fn)
            else:
                ensure_dir(dir)
                fp = zip.open(name)
                try:
                    with open(fn, "wb") as destfp:
                        shutil.copyfileobj(fp, destfp)
                finally:
                    fp.close()
                    if zip_item_is_executable(info):
                        set_extracted_file_to_default_mode_plus_executable(fn)
    finally:
        zipfp.close()
def untar_file(filename, location):
    ensure_dir(location)
    if filename.lower().endswith(".gz") or filename.lower().endswith(".tgz"):
        mode = "r:gz"
    elif filename.lower().endswith(BZ2_EXTENSIONS):
        mode = "r:bz2"
    elif filename.lower().endswith(XZ_EXTENSIONS):
        mode = "r:xz"
    elif filename.lower().endswith(".tar"):
        mode = "r"
    else:
        logger.warning(
            "Cannot determine compression type for file %s",
            filename,
        )
        mode = "r:*"
    tar = tarfile.open(filename, mode)
    try:
        leading = has_leading_dir([member.name for member in tar.getmembers()])
        for member in tar.getmembers():
            fn = member.name
            if leading:
                fn = split_leading_dir(fn)[1]
            path = os.path.join(location, fn)
            if not is_within_directory(location, path):
                message = (
                    "The tar file ({}) has a file ({}) trying to install "
                    "outside target directory ({})"
                )
                raise InstallationError(message.format(filename, path, location))
            if member.isdir():
                ensure_dir(path)
            elif member.issym():
                try:
                    tar._extract_member(member, path)
                except Exception as exc:
                    logger.warning(
                        "In the tar file %s the member %s is invalid: %s",
                        filename,
                        member.name,
                        exc,
                    )
                    continue
            else:
                try:
                    fp = tar.extractfile(member)
                except (KeyError, AttributeError) as exc:
                    logger.warning(
                        "In the tar file %s the member %s is invalid: %s",
                        filename,
                        member.name,
                        exc,
                    )
                    continue
                ensure_dir(os.path.dirname(path))
                assert fp is not None
                with open(path, "wb") as destfp:
                    shutil.copyfileobj(fp, destfp)
                fp.close()
                tar.utime(member, path)
                if member.mode & 0o111:
                    set_extracted_file_to_default_mode_plus_executable(path)
    finally:
        tar.close()
def unpack_file(
    filename,
    location,
    content_type=None,
):
    filename = os.path.realpath(filename)
    if (
        content_type == "application/zip"
        or filename.lower().endswith(ZIP_EXTENSIONS)
        or zipfile.is_zipfile(filename)
    ):
        unzip_file(filename, location, flatten=not filename.endswith(".whl"))
    elif (
        content_type == "application/x-gzip"
        or tarfile.is_tarfile(filename)
        or filename.lower().endswith(TAR_EXTENSIONS + BZ2_EXTENSIONS + XZ_EXTENSIONS)
    ):
        untar_file(filename, location)
    else:
        logger.critical(
            "Cannot unpack file %s (downloaded from %s, content-type: %s); "
            "cannot detect archive format",
            filename,
            location,
            content_type,
        )
        raise InstallationError(f"Cannot determine archive format of {location}")
