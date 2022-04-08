import os
import posixpath
import re
import urllib.parse
from typing import TYPE_CHECKING, Optional, Tuple, Union
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
    redact_auth_from_url,
    split_auth_from_netloc,
    splitext,
)
from pip._internal.utils.models import KeyBasedCompareMixin
from pip._internal.utils.urls import path_to_url, url_to_path
if TYPE_CHECKING:
    from pip._internal.index.collector import HTMLPage
class Link(KeyBasedCompareMixin):
    __slots__ = [
        "_parsed_url",
        "_url",
        "comes_from",
        "requires_python",
        "yanked_reason",
        "cache_link_parsing",
    ]
    def __init__(
        self,
        url,
        comes_from=None,
        requires_python=None,
        yanked_reason=None,
        cache_link_parsing=True,
    ):
        if url.startswith('\\\\'):
            url = path_to_url(url)
        self._parsed_url = urllib.parse.urlsplit(url)
        self._url = url
        self.comes_from = comes_from
        self.requires_python = requires_python if requires_python else None
        self.yanked_reason = yanked_reason
        super().__init__(key=url, defining_class=Link)
        self.cache_link_parsing = cache_link_parsing
    def __str__(self):
        if self.requires_python:
            rp = f' (requires-python:{self.requires_python})'
        else:
            rp = ''
        if self.comes_from:
            return '{} (from {}){}'.format(
                redact_auth_from_url(self._url), self.comes_from, rp)
        else:
            return redact_auth_from_url(str(self._url))
    def __repr__(self):
        return f'<Link {self}>'
    @property
    def url(self):
        return self._url
    @property
    def filename(self):
        path = self.path.rstrip('/')
        name = posixpath.basename(path)
        if not name:
            netloc, user_pass = split_auth_from_netloc(self.netloc)
            return netloc
        name = urllib.parse.unquote(name)
        assert name, f'URL {self._url!r} produced no filename'
        return name
    @property
    def file_path(self):
        return url_to_path(self.url)
    @property
    def scheme(self):
        return self._parsed_url.scheme
    @property
    def netloc(self):
        return self._parsed_url.netloc
    @property
    def path(self):
        return urllib.parse.unquote(self._parsed_url.path)
    def splitext(self):
        return splitext(posixpath.basename(self.path.rstrip('/')))
    @property
    def ext(self):
        return self.splitext()[1]
    @property
    def url_without_fragment(self):
        scheme, netloc, path, query, fragment = self._parsed_url
        return urllib.parse.urlunsplit((scheme, netloc, path, query, None))
    @property
    def egg_fragment(self):
        match = self._egg_fragment_re.search(self._url)
        if not match:
            return None
        return match.group(1)
    @property
    def subdirectory_fragment(self):
        match = self._subdirectory_fragment_re.search(self._url)
        if not match:
            return None
        return match.group(1)
    _hash_re = re.compile(
        r'(sha1|sha224|sha384|sha256|sha512|md5)=([a-f0-9]+)'
    )
    @property
    def hash(self):
        match = self._hash_re.search(self._url)
        if match:
            return match.group(2)
        return None
    @property
    def hash_name(self):
        match = self._hash_re.search(self._url)
        if match:
            return match.group(1)
        return None
    @property
    def show_url(self):
    @property
    def is_file(self):
        return self.scheme == 'file'
    def is_existing_dir(self):
        return self.is_file and os.path.isdir(self.file_path)
    @property
    def is_wheel(self):
        return self.ext == WHEEL_EXTENSION
    @property
    def is_vcs(self):
        from pip._internal.vcs import vcs
        return self.scheme in vcs.all_schemes
    @property
    def is_yanked(self):
        return self.yanked_reason is not None
    @property
    def has_hash(self):
        return self.hash_name is not None
    def is_hash_allowed(self, hashes):
        if hashes is None or not self.has_hash:
            return False
        assert self.hash_name is not None
        assert self.hash is not None
        return hashes.is_hash_allowed(self.hash_name, hex_digest=self.hash)
def links_equivalent(link1, link2):
    return link1 == link2
