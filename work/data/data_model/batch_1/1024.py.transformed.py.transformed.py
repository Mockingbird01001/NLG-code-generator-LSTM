import logging
import mimetypes
import os
import pathlib
from typing import Callable, Iterable, Optional, Tuple
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.link import Link
from pip._internal.utils.urls import path_to_url, url_to_path
from pip._internal.vcs import is_url
logger = logging.getLogger(__name__)
FoundCandidates = Iterable[InstallationCandidate]
FoundLinks = Iterable[Link]
CandidatesFromPage = Callable[[Link], Iterable[InstallationCandidate]]
PageValidator = Callable[[Link], bool]
class LinkSource:
    @property
    def link(self) -> Optional[Link]:
        raise NotImplementedError()
    def page_candidates(self) -> FoundCandidates:
        raise NotImplementedError()
    def file_links(self) -> FoundLinks:
        raise NotImplementedError()
def _is_html_file(file_url: str) -> bool:
    return mimetypes.guess_type(file_url, strict=False)[0] == "text/html"
class _FlatDirectorySource(LinkSource):
    def __init__(
        self,
        candidates_from_page: CandidatesFromPage,
        path: str,
    ) -> None:
        self._candidates_from_page = candidates_from_page
        self._path = pathlib.Path(os.path.realpath(path))
    @property
    def link(self) -> Optional[Link]:
        return None
    def page_candidates(self) -> FoundCandidates:
        for path in self._path.iterdir():
            url = path_to_url(str(path))
            if not _is_html_file(url):
                continue
            yield from self._candidates_from_page(Link(url))
    def file_links(self) -> FoundLinks:
        for path in self._path.iterdir():
            url = path_to_url(str(path))
            if _is_html_file(url):
                continue
            yield Link(url)
class _LocalFileSource(LinkSource):
    def __init__(
        self,
        candidates_from_page: CandidatesFromPage,
        link: Link,
    ) -> None:
        self._candidates_from_page = candidates_from_page
        self._link = link
    @property
    def link(self) -> Optional[Link]:
        return self._link
    def page_candidates(self) -> FoundCandidates:
        if not _is_html_file(self._link.url):
            return
        yield from self._candidates_from_page(self._link)
    def file_links(self) -> FoundLinks:
        if _is_html_file(self._link.url):
            return
        yield self._link
class _RemoteFileSource(LinkSource):
    def __init__(
        self,
        candidates_from_page: CandidatesFromPage,
        page_validator: PageValidator,
        link: Link,
    ) -> None:
        self._candidates_from_page = candidates_from_page
        self._page_validator = page_validator
        self._link = link
    @property
    def link(self) -> Optional[Link]:
        return self._link
    def page_candidates(self) -> FoundCandidates:
        if not self._page_validator(self._link):
            return
        yield from self._candidates_from_page(self._link)
    def file_links(self) -> FoundLinks:
        yield self._link
class _IndexDirectorySource(LinkSource):
    def __init__(
        self,
        candidates_from_page: CandidatesFromPage,
        link: Link,
    ) -> None:
        self._candidates_from_page = candidates_from_page
        self._link = link
    @property
    def link(self) -> Optional[Link]:
        return self._link
    def page_candidates(self) -> FoundCandidates:
        yield from self._candidates_from_page(self._link)
    def file_links(self) -> FoundLinks:
        return ()
def build_source(
    location: str,
    *,
    candidates_from_page: CandidatesFromPage,
    page_validator: PageValidator,
    expand_dir: bool,
    cache_link_parsing: bool,
) -> Tuple[Optional[str], Optional[LinkSource]]:
    path: Optional[str] = None
    url: Optional[str] = None
    if os.path.exists(location):
        url = path_to_url(location)
        path = location
    elif location.startswith("file:"):
        url = location
        path = url_to_path(location)
    elif is_url(location):
        url = location
    if url is None:
        msg = (
            "Location '%s' is ignored: "
            "it is either a non-existing path or lacks a specific scheme."
        )
        logger.warning(msg, location)
        return (None, None)
    if path is None:
        source: LinkSource = _RemoteFileSource(
            candidates_from_page=candidates_from_page,
            page_validator=page_validator,
            link=Link(url, cache_link_parsing=cache_link_parsing),
        )
        return (url, source)
    if os.path.isdir(path):
        if expand_dir:
            source = _FlatDirectorySource(
                candidates_from_page=candidates_from_page,
                path=path,
            )
        else:
            source = _IndexDirectorySource(
                candidates_from_page=candidates_from_page,
                link=Link(url, cache_link_parsing=cache_link_parsing),
            )
        return (url, source)
    elif os.path.isfile(path):
        source = _LocalFileSource(
            candidates_from_page=candidates_from_page,
            link=Link(url, cache_link_parsing=cache_link_parsing),
        )
        return (url, source)
    logger.warning(
        "Location '%s' is ignored: it is neither a file nor a directory.",
        location,
    )
    return (url, None)
