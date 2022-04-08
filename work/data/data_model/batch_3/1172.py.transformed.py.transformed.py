
from __future__ import absolute_import, division, print_function
import re
from ._typing import TYPE_CHECKING, cast
from .version import InvalidVersion, Version
if TYPE_CHECKING:
    from typing import NewType, Union
    NormalizedName = NewType("NormalizedName", str)
_canonicalize_regex = re.compile(r"[-_.]+")
def canonicalize_name(name):
    value = _canonicalize_regex.sub("-", name).lower()
    return cast("NormalizedName", value)
def canonicalize_version(_version):
    try:
        version = Version(_version)
    except InvalidVersion:
        return _version
    parts = []
    if version.epoch != 0:
        parts.append("{0}!".format(version.epoch))
    parts.append(re.sub(r"(\.0)+$", "", ".".join(str(x) for x in version.release)))
    if version.pre is not None:
        parts.append("".join(str(x) for x in version.pre))
    if version.post is not None:
        parts.append(".post{0}".format(version.post))
    if version.dev is not None:
        parts.append(".dev{0}".format(version.dev))
    if version.local is not None:
        parts.append("+{0}".format(version.local))
    return "".join(parts)
