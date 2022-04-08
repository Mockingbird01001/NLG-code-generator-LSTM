
import os
import sys
from typing import Optional, Tuple
def glibc_version_string():
    return glibc_version_string_confstr() or glibc_version_string_ctypes()
def glibc_version_string_confstr():
    if sys.platform == "win32":
        return None
    try:
        _, version = os.confstr("CS_GNU_LIBC_VERSION").split()
    except (AttributeError, OSError, ValueError):
        return None
    return version
def glibc_version_string_ctypes():
    try:
        import ctypes
    except ImportError:
        return None
    process_namespace = ctypes.CDLL(None)
    try:
        gnu_get_libc_version = process_namespace.gnu_get_libc_version
    except AttributeError:
        return None
    gnu_get_libc_version.restype = ctypes.c_char_p
    version_str = gnu_get_libc_version()
    if not isinstance(version_str, str):
        version_str = version_str.decode("ascii")
    return version_str
def libc_ver():
    glibc_version = glibc_version_string()
    if glibc_version is None:
        return ("", "")
    else:
        return ("glibc", glibc_version)
