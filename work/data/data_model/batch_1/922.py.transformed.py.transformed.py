from __future__ import absolute_import
from base64 import b64encode
from ..exceptions import UnrewindableBodyError
from ..packages.six import b, integer_types
SKIP_HEADER = "@@@SKIP_HEADER@@@"
SKIPPABLE_HEADERS = frozenset(["accept-encoding", "host", "user-agent"])
ACCEPT_ENCODING = "gzip,deflate"
try:
    import brotli as _unused_module_brotli
except ImportError:
    pass
else:
    ACCEPT_ENCODING += ",br"
_FAILEDTELL = object()
def make_headers(
    keep_alive=None,
    accept_encoding=None,
    user_agent=None,
    basic_auth=None,
    proxy_basic_auth=None,
    disable_cache=None,
):
    headers = {}
    if accept_encoding:
        if isinstance(accept_encoding, str):
            pass
        elif isinstance(accept_encoding, list):
            accept_encoding = ",".join(accept_encoding)
        else:
            accept_encoding = ACCEPT_ENCODING
        headers["accept-encoding"] = accept_encoding
    if user_agent:
        headers["user-agent"] = user_agent
    if keep_alive:
        headers["connection"] = "keep-alive"
    if basic_auth:
        headers["authorization"] = "Basic " + b64encode(b(basic_auth)).decode("utf-8")
    if proxy_basic_auth:
        headers["proxy-authorization"] = "Basic " + b64encode(
            b(proxy_basic_auth)
        ).decode("utf-8")
    if disable_cache:
        headers["cache-control"] = "no-cache"
    return headers
def set_file_position(body, pos):
    if pos is not None:
        rewind_body(body, pos)
    elif getattr(body, "tell", None) is not None:
        try:
            pos = body.tell()
        except (IOError, OSError):
            pos = _FAILEDTELL
    return pos
def rewind_body(body, body_pos):
    body_seek = getattr(body, "seek", None)
    if body_seek is not None and isinstance(body_pos, integer_types):
        try:
            body_seek(body_pos)
        except (IOError, OSError):
            raise UnrewindableBodyError(
                "An error occurred when rewinding request body for redirect/retry."
            )
    elif body_pos is _FAILEDTELL:
        raise UnrewindableBodyError(
            "Unable to record file position for rewinding "
            "request body during a redirect/retry."
        )
    else:
        raise ValueError(
            "body_pos must be of type integer, instead it was %s." % type(body_pos)
        )
