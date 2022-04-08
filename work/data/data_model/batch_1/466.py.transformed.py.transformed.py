
import functools
from typing import Callable, Iterator, Optional, Set, Tuple
from pip._vendor.packaging.version import _BaseVersion
from pip._vendor.six.moves import collections_abc
from .base import Candidate
IndexCandidateInfo = Tuple[_BaseVersion, Callable[[], Optional[Candidate]]]
def _iter_built(infos):
    versions_found = set()
    for version, func in infos:
        if version in versions_found:
            continue
        candidate = func()
        if candidate is None:
            continue
        yield candidate
        versions_found.add(version)
def _iter_built_with_prepended(installed, infos):
    yield installed
    versions_found = {installed.version}
    for version, func in infos:
        if version in versions_found:
            continue
        candidate = func()
        if candidate is None:
            continue
        yield candidate
        versions_found.add(version)
def _iter_built_with_inserted(installed, infos):
    versions_found = set()
    for version, func in infos:
        if version in versions_found:
            continue
        if installed.version >= version:
            yield installed
            versions_found.add(installed.version)
        candidate = func()
        if candidate is None:
            continue
        yield candidate
        versions_found.add(version)
    if installed.version not in versions_found:
        yield installed
class FoundCandidates(collections_abc.Sequence):
    def __init__(
        self,
        get_infos: Callable[[], Iterator[IndexCandidateInfo]],
        installed: Optional[Candidate],
        prefers_installed: bool,
        incompatible_ids: Set[int],
    ):
        self._get_infos = get_infos
        self._installed = installed
        self._prefers_installed = prefers_installed
        self._incompatible_ids = incompatible_ids
    def __getitem__(self, index):
        raise NotImplementedError("don't do this")
    def __iter__(self):
        infos = self._get_infos()
        if not self._installed:
            iterator = _iter_built(infos)
        elif self._prefers_installed:
            iterator = _iter_built_with_prepended(self._installed, infos)
        else:
            iterator = _iter_built_with_inserted(self._installed, infos)
        return (c for c in iterator if id(c) not in self._incompatible_ids)
    def __len__(self):
        raise NotImplementedError("don't do this")
    @functools.lru_cache(maxsize=1)
    def __bool__(self):
        if self._prefers_installed and self._installed:
            return True
        return any(self)
    __nonzero__ = __bool__
