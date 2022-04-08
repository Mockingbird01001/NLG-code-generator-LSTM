from typing import Optional
from pip._internal.models.format_control import FormatControl
class SelectionPreferences:
    __slots__ = ['allow_yanked', 'allow_all_prereleases', 'format_control',
                 'prefer_binary', 'ignore_requires_python']
    def __init__(
        self,
        allow_yanked,
        allow_all_prereleases=False,
        format_control=None,
        prefer_binary=False,
        ignore_requires_python=None,
    ):
        if ignore_requires_python is None:
            ignore_requires_python = False
        self.allow_yanked = allow_yanked
        self.allow_all_prereleases = allow_all_prereleases
        self.format_control = format_control
        self.prefer_binary = prefer_binary
        self.ignore_requires_python = ignore_requires_python
