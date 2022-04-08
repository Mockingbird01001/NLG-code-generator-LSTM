import sys
from typing import List, Optional, Tuple
from pip._vendor.packaging.tags import Tag
from pip._internal.utils.compatibility_tags import get_supported, version_info_to_nodot
from pip._internal.utils.misc import normalize_version_info
class TargetPython:
    __slots__ = [
        "_given_py_version_info",
        "abis",
        "implementation",
        "platforms",
        "py_version",
        "py_version_info",
        "_valid_tags",
    ]
    def __init__(
        self,
        platforms=None,
        py_version_info=None,
        abis=None,
        implementation=None,
    ):
        self._given_py_version_info = py_version_info
        if py_version_info is None:
            py_version_info = sys.version_info[:3]
        else:
            py_version_info = normalize_version_info(py_version_info)
        py_version = '.'.join(map(str, py_version_info[:2]))
        self.abis = abis
        self.implementation = implementation
        self.platforms = platforms
        self.py_version = py_version
        self.py_version_info = py_version_info
        self._valid_tags = None
    def format_given(self):
        display_version = None
        if self._given_py_version_info is not None:
            display_version = '.'.join(
                str(part) for part in self._given_py_version_info
            )
        key_values = [
            ('platforms', self.platforms),
            ('version_info', display_version),
            ('abis', self.abis),
            ('implementation', self.implementation),
        ]
        return ' '.join(
            f'{key}={value!r}' for key, value in key_values
            if value is not None
        )
    def get_tags(self):
        if self._valid_tags is None:
            py_version_info = self._given_py_version_info
            if py_version_info is None:
                version = None
            else:
                version = version_info_to_nodot(py_version_info)
            tags = get_supported(
                version=version,
                platforms=self.platforms,
                abis=self.abis,
                impl=self.implementation,
            )
            self._valid_tags = tags
        return self._valid_tags
