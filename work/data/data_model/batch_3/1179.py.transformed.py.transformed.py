
from __future__ import absolute_import, division, print_function
import collections
import itertools
import re
from ._structures import Infinity, NegativeInfinity
from ._typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Iterator, List, Optional, SupportsInt, Tuple, Union
    from ._structures import InfinityType, NegativeInfinityType
    InfiniteTypes = Union[InfinityType, NegativeInfinityType]
    PrePostDevType = Union[InfiniteTypes, Tuple[str, int]]
    SubLocalType = Union[InfiniteTypes, int, str]
    LocalType = Union[
        NegativeInfinityType,
        Tuple[
            Union[
                SubLocalType,
                Tuple[SubLocalType, str],
                Tuple[NegativeInfinityType, SubLocalType],
            ],
            ...,
        ],
    ]
    CmpKey = Tuple[
        int, Tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType
    ]
    LegacyCmpKey = Tuple[int, Tuple[str, ...]]
    VersionComparisonMethod = Callable[
        [Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool
    ]
__all__ = ["parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN"]
_Version = collections.namedtuple(
    "_Version", ["epoch", "release", "dev", "pre", "post", "local"]
)
def parse(version):
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)
class InvalidVersion(ValueError):
class _BaseVersion(object):
    _key = None
    def __hash__(self):
        return hash(self._key)
    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)
    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)
    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)
    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)
    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)
    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)
    def _compare(self, other, method):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return method(self._key, other._key)
class LegacyVersion(_BaseVersion):
    def __init__(self, version):
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)
    def __str__(self):
        return self._version
    def __repr__(self):
        return "<LegacyVersion({0})>".format(repr(str(self)))
    @property
    def public(self):
        return self._version
    @property
    def base_version(self):
        return self._version
    @property
    def epoch(self):
        return -1
    @property
    def release(self):
        return None
    @property
    def pre(self):
        return None
    @property
    def post(self):
        return None
    @property
    def dev(self):
        return None
    @property
    def local(self):
        return None
    @property
    def is_prerelease(self):
        return False
    @property
    def is_postrelease(self):
        return False
    @property
    def is_devrelease(self):
        return False
_legacy_version_component_re = re.compile(r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE)
_legacy_version_replacement_map = {
    "pre": "c",
    "preview": "c",
    "-": "final-",
    "rc": "c",
    "dev": "@",
}
def _parse_version_parts(s):
    for part in _legacy_version_component_re.split(s):
        part = _legacy_version_replacement_map.get(part, part)
        if not part or part == ".":
            continue
        if part[:1] in "0123456789":
            yield part.zfill(8)
        else:
            yield "*" + part
    yield "*final"
def _legacy_cmpkey(version):
    epoch = -1
    parts = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()
            while parts and parts[-1] == "00000000":
                parts.pop()
        parts.append(part)
    return epoch, tuple(parts)
VERSION_PATTERN = r"""
    v?
    (?:
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
"""
class Version(_BaseVersion):
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    def __init__(self, version):
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion("Invalid version: '{0}'".format(version))
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )
    def __repr__(self):
        return "<Version({0})>".format(repr(str(self)))
    def __str__(self):
        parts = []
        if self.epoch != 0:
            parts.append("{0}!".format(self.epoch))
        parts.append(".".join(str(x) for x in self.release))
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))
        if self.post is not None:
            parts.append(".post{0}".format(self.post))
        if self.dev is not None:
            parts.append(".dev{0}".format(self.dev))
        if self.local is not None:
            parts.append("+{0}".format(self.local))
        return "".join(parts)
    @property
    def epoch(self):
        _epoch = self._version.epoch
        return _epoch
    @property
    def release(self):
        _release = self._version.release
        return _release
    @property
    def pre(self):
        _pre = self._version.pre
        return _pre
    @property
    def post(self):
        return self._version.post[1] if self._version.post else None
    @property
    def dev(self):
        return self._version.dev[1] if self._version.dev else None
    @property
    def local(self):
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None
    @property
    def public(self):
        return str(self).split("+", 1)[0]
    @property
    def base_version(self):
        parts = []
        if self.epoch != 0:
            parts.append("{0}!".format(self.epoch))
        parts.append(".".join(str(x) for x in self.release))
        return "".join(parts)
    @property
    def is_prerelease(self):
        return self.dev is not None or self.pre is not None
    @property
    def is_postrelease(self):
        return self.post is not None
    @property
    def is_devrelease(self):
        return self.dev is not None
    @property
    def major(self):
        return self.release[0] if len(self.release) >= 1 else 0
    @property
    def minor(self):
        return self.release[1] if len(self.release) >= 2 else 0
    @property
    def micro(self):
        return self.release[2] if len(self.release) >= 3 else 0
def _parse_letter_version(
    letter,
    number,
):
    if letter:
        if number is None:
            number = 0
        letter = letter.lower()
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"
        return letter, int(number)
    if not letter and number:
        letter = "post"
        return letter, int(number)
    return None
_local_version_separators = re.compile(r"[\._-]")
def _parse_local_version(local):
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None
def _cmpkey(
    epoch,
    release,
    pre,
    post,
    dev,
    local,
):
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )
    if pre is None and post is None and dev is not None:
        _pre = NegativeInfinity
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre
    if post is None:
        _post = NegativeInfinity
    else:
        _post = post
    if dev is None:
        _dev = Infinity
    else:
        _dev = dev
    if local is None:
        _local = NegativeInfinity
    else:
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )
    return epoch, _release, _pre, _post, _dev, _local
