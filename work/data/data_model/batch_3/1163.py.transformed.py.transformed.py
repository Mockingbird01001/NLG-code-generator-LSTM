import os
import sys
import urllib.parse
import urllib.request
from typing import Optional
def get_url_scheme(url):
    if ":" not in url:
        return None
    return url.split(":", 1)[0].lower()
def path_to_url(path):
    path = os.path.normpath(os.path.abspath(path))
    url = urllib.parse.urljoin("file:", urllib.request.pathname2url(path))
    return url
def url_to_path(url):
    assert url.startswith(
        "file:"
    ), f"You can only turn file: urls into filenames (not {url!r})"
    _, netloc, path, _, _ = urllib.parse.urlsplit(url)
    if not netloc or netloc == "localhost":
        netloc = ""
    elif sys.platform == "win32":
        netloc = "\\\\" + netloc
    else:
        raise ValueError(
            f"non-local file URIs are not supported on this platform: {url!r}"
        )
    path = urllib.request.url2pathname(netloc + path)
    return path
