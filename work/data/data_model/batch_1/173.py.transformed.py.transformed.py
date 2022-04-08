
import cgi
import collections
import functools
import html
import itertools
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree
from optparse import Values
from typing import (
    Callable,
    Iterable,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)
from pip._vendor import html5lib, requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import pairwise, redact_auth_from_url
from pip._internal.vcs import vcs
from .sources import CandidatesFromPage, LinkSource, build_source
logger = logging.getLogger(__name__)
HTMLElement = xml.etree.ElementTree.Element
ResponseHeaders = MutableMapping[str, str]
def _match_vcs_scheme(url):
    for scheme in vcs.schemes:
        if url.lower().startswith(scheme) and url[len(scheme)] in '+:':
            return scheme
    return None
class _NotHTML(Exception):
    def __init__(self, content_type, request_desc):
        super().__init__(content_type, request_desc)
        self.content_type = content_type
        self.request_desc = request_desc
def _ensure_html_header(response):
    content_type = response.headers.get("Content-Type", "")
    if not content_type.lower().startswith("text/html"):
        raise _NotHTML(content_type, response.request.method)
class _NotHTTP(Exception):
    pass
def _ensure_html_response(url, session):
    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(url)
    if scheme not in {'http', 'https'}:
        raise _NotHTTP()
    resp = session.head(url, allow_redirects=True)
    raise_for_status(resp)
    _ensure_html_header(resp)
def _get_html_response(url, session):
    if is_archive_file(Link(url).filename):
        _ensure_html_response(url, session=session)
    logger.debug('Getting page %s', redact_auth_from_url(url))
    resp = session.get(
        url,
        headers={
            "Accept": "text/html",
            "Cache-Control": "max-age=0",
        },
    )
    raise_for_status(resp)
    _ensure_html_header(resp)
    return resp
def _get_encoding_from_headers(headers):
    if headers and "Content-Type" in headers:
        content_type, params = cgi.parse_header(headers["Content-Type"])
        if "charset" in params:
            return params['charset']
    return None
def _determine_base_url(document, page_url):
    for base in document.findall(".//base"):
        href = base.get("href")
        if href is not None:
            return href
    return page_url
def _clean_url_path_part(part):
    return urllib.parse.quote(urllib.parse.unquote(part))
def _clean_file_url_path(part):
    return urllib.request.pathname2url(urllib.request.url2pathname(part))
_reserved_chars_re = re.compile('(@|%2F)', re.IGNORECASE)
def _clean_url_path(path, is_local_path):
    if is_local_path:
        clean_func = _clean_file_url_path
    else:
        clean_func = _clean_url_path_part
    parts = _reserved_chars_re.split(path)
    cleaned_parts = []
    for to_clean, reserved in pairwise(itertools.chain(parts, [''])):
        cleaned_parts.append(clean_func(to_clean))
        cleaned_parts.append(reserved.upper())
    return ''.join(cleaned_parts)
def _clean_link(url):
    result = urllib.parse.urlparse(url)
    is_local_path = not result.netloc
    path = _clean_url_path(result.path, is_local_path=is_local_path)
    return urllib.parse.urlunparse(result._replace(path=path))
def _create_link_from_element(
    anchor,
    page_url,
    base_url,
):
    href = anchor.get("href")
    if not href:
        return None
    url = _clean_link(urllib.parse.urljoin(base_url, href))
    pyrequire = anchor.get('data-requires-python')
    pyrequire = html.unescape(pyrequire) if pyrequire else None
    yanked_reason = anchor.get('data-yanked')
    if yanked_reason:
        yanked_reason = html.unescape(yanked_reason)
    link = Link(
        url,
        comes_from=page_url,
        requires_python=pyrequire,
        yanked_reason=yanked_reason,
    )
    return link
class CacheablePageContent:
    def __init__(self, page):
        assert page.cache_link_parsing
        self.page = page
    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.page.url == other.page.url)
    def __hash__(self):
        return hash(self.page.url)
def with_cached_html_pages(
    fn,
):
    @functools.lru_cache(maxsize=None)
    def wrapper(cacheable_page):
        return list(fn(cacheable_page.page))
    @functools.wraps(fn)
    def wrapper_wrapper(page):
        if page.cache_link_parsing:
            return wrapper(CacheablePageContent(page))
        return list(fn(page))
    return wrapper_wrapper
@with_cached_html_pages
def parse_links(page):
    document = html5lib.parse(
        page.content,
        transport_encoding=page.encoding,
        namespaceHTMLElements=False,
    )
    url = page.url
    base_url = _determine_base_url(document, url)
    for anchor in document.findall(".//a"):
        link = _create_link_from_element(
            anchor,
            page_url=url,
            base_url=base_url,
        )
        if link is None:
            continue
        yield link
class HTMLPage:
    def __init__(
        self,
        content,
        encoding,
        url,
        cache_link_parsing=True,
    ):
        self.content = content
        self.encoding = encoding
        self.url = url
        self.cache_link_parsing = cache_link_parsing
    def __str__(self):
        return redact_auth_from_url(self.url)
def _handle_get_page_fail(
    link,
    reason,
    meth=None
):
    if meth is None:
        meth = logger.debug
    meth("Could not fetch URL %s: %s - skipping", link, reason)
def _make_html_page(response, cache_link_parsing=True):
    encoding = _get_encoding_from_headers(response.headers)
    return HTMLPage(
        response.content,
        encoding=encoding,
        url=response.url,
        cache_link_parsing=cache_link_parsing)
def _get_html_page(link, session=None):
    if session is None:
        raise TypeError(
            "_get_html_page() missing 1 required keyword argument: 'session'"
        )
    vcs_scheme = _match_vcs_scheme(url)
    if vcs_scheme:
        logger.warning('Cannot look at %s URL %s because it does not support '
                       'lookup as web pages.', vcs_scheme, link)
        return None
    scheme, _, path, _, _, _ = urllib.parse.urlparse(url)
    if (scheme == 'file' and os.path.isdir(urllib.request.url2pathname(path))):
        if not url.endswith('/'):
            url += '/'
        url = urllib.parse.urljoin(url, 'index.html')
        logger.debug(' file: URL is directory, getting %s', url)
    try:
        resp = _get_html_response(url, session=session)
    except _NotHTTP:
        logger.warning(
            'Skipping page %s because it looks like an archive, and cannot '
            'be checked by a HTTP HEAD request.', link,
        )
    except _NotHTML as exc:
        logger.warning(
            'Skipping page %s because the %s request got Content-Type: %s.'
            'The only supported Content-Type is text/html',
            link, exc.request_desc, exc.content_type,
        )
    except NetworkConnectionError as exc:
        _handle_get_page_fail(link, exc)
    except RetryError as exc:
        _handle_get_page_fail(link, exc)
    except SSLError as exc:
        reason = "There was a problem confirming the ssl certificate: "
        reason += str(exc)
        _handle_get_page_fail(link, reason, meth=logger.info)
    except requests.ConnectionError as exc:
        _handle_get_page_fail(link, f"connection error: {exc}")
    except requests.Timeout:
        _handle_get_page_fail(link, "timed out")
    else:
        return _make_html_page(resp,
                               cache_link_parsing=link.cache_link_parsing)
    return None
class CollectedSources(NamedTuple):
    find_links: Sequence[Optional[LinkSource]]
    index_urls: Sequence[Optional[LinkSource]]
class LinkCollector:
    def __init__(
        self,
        session,
        search_scope,
    ):
        self.search_scope = search_scope
        self.session = session
    @classmethod
    def create(cls, session, options, suppress_no_index=False):
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index and not suppress_no_index:
            logger.debug(
                'Ignoring indexes: %s',
                ','.join(redact_auth_from_url(url) for url in index_urls),
            )
            index_urls = []
        find_links = options.find_links or []
        search_scope = SearchScope.create(
            find_links=find_links, index_urls=index_urls,
        )
        link_collector = LinkCollector(
            session=session, search_scope=search_scope,
        )
        return link_collector
    @property
    def find_links(self):
        return self.search_scope.find_links
    def fetch_page(self, location):
        return _get_html_page(location, session=self.session)
    def collect_sources(
        self,
        project_name: str,
        candidates_from_page: CandidatesFromPage,
    ) -> CollectedSources:
        index_url_sources = collections.OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=False,
                cache_link_parsing=False,
            )
            for loc in self.search_scope.get_index_urls_locations(project_name)
        ).values()
        find_links_sources = collections.OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=True,
                cache_link_parsing=True,
            )
            for loc in self.find_links
        ).values()
        if logger.isEnabledFor(logging.DEBUG):
            lines = [
                f"* {s.link}"
                for s in itertools.chain(find_links_sources, index_url_sources)
                if s is not None and s.link is not None
            ]
            lines = [
                f"{len(lines)} location(s) to search "
                f"for versions of {project_name}:"
            ] + lines
            logger.debug("\n".join(lines))
        return CollectedSources(
            find_links=list(find_links_sources),
            index_urls=list(index_url_sources),
        )
