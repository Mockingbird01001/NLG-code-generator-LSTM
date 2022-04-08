
import email.utils
import ipaddress
import json
import logging
import mimetypes
import os
import platform
import sys
import urllib.parse
import warnings
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
from pip._vendor import requests, urllib3
from pip._vendor.cachecontrol import CacheControlAdapter
from pip._vendor.requests.adapters import BaseAdapter, HTTPAdapter
from pip._vendor.requests.models import PreparedRequest, Response
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.urllib3.connectionpool import ConnectionPool
from pip._vendor.urllib3.exceptions import InsecureRequestWarning
from pip import __version__
from pip._internal.metadata import get_default_environment
from pip._internal.models.link import Link
from pip._internal.network.auth import MultiDomainBasicAuth
from pip._internal.network.cache import SafeFileCache
from pip._internal.utils.compat import has_tls
from pip._internal.utils.glibc import libc_ver
from pip._internal.utils.misc import build_url_from_netloc, parse_netloc
from pip._internal.utils.urls import url_to_path
logger = logging.getLogger(__name__)
SecureOrigin = Tuple[str, str, Optional[Union[int, str]]]
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
SECURE_ORIGINS = [
    ("https", "*", "*"),
    ("*", "localhost", "*"),
    ("*", "127.0.0.0/8", "*"),
    ("*", "::1/128", "*"),
    ("file", "*", None),
    ("ssh", "*", "*"),
]
CI_ENVIRONMENT_VARIABLES = (
    'BUILD_BUILDID',
    'BUILD_ID',
    'CI',
    'PIP_IS_CI',
)
def looks_like_ci():
    return any(name in os.environ for name in CI_ENVIRONMENT_VARIABLES)
def user_agent():
    data = {
        "installer": {"name": "pip", "version": __version__},
        "python": platform.python_version(),
        "implementation": {
            "name": platform.python_implementation(),
        },
    }
    if data["implementation"]["name"] == 'CPython':
        data["implementation"]["version"] = platform.python_version()
    elif data["implementation"]["name"] == 'PyPy':
        pypy_version_info = sys.pypy_version_info
        if pypy_version_info.releaselevel == 'final':
            pypy_version_info = pypy_version_info[:3]
        data["implementation"]["version"] = ".".join(
            [str(x) for x in pypy_version_info]
        )
    elif data["implementation"]["name"] == 'Jython':
        data["implementation"]["version"] = platform.python_version()
    elif data["implementation"]["name"] == 'IronPython':
        data["implementation"]["version"] = platform.python_version()
    if sys.platform.startswith("linux"):
        from pip._vendor import distro
        linux_distribution = distro.linux_distribution()
        distro_infos = dict(filter(
            lambda x: x[1],
            zip(["name", "version", "id"], linux_distribution),
        ))
        libc = dict(filter(
            lambda x: x[1],
            zip(["lib", "version"], libc_ver()),
        ))
        if libc:
            distro_infos["libc"] = libc
        if distro_infos:
            data["distro"] = distro_infos
    if sys.platform.startswith("darwin") and platform.mac_ver()[0]:
        data["distro"] = {"name": "macOS", "version": platform.mac_ver()[0]}
    if platform.system():
        data.setdefault("system", {})["name"] = platform.system()
    if platform.release():
        data.setdefault("system", {})["release"] = platform.release()
    if platform.machine():
        data["cpu"] = platform.machine()
    if has_tls():
        import _ssl as ssl
        data["openssl_version"] = ssl.OPENSSL_VERSION
    setuptools_dist = get_default_environment().get_distribution("setuptools")
    if setuptools_dist is not None:
        data["setuptools_version"] = str(setuptools_dist.version)
    data["ci"] = True if looks_like_ci() else None
    user_data = os.environ.get("PIP_USER_AGENT_USER_DATA")
    if user_data is not None:
        data["user_data"] = user_data
    return "{data[installer][name]}/{data[installer][version]} {json}".format(
        data=data,
        json=json.dumps(data, separators=(",", ":"), sort_keys=True),
    )
class LocalFSAdapter(BaseAdapter):
    def send(
        self,
        request,
        stream=False,
        timeout=None,
        verify=True,
        cert=None,
        proxies=None,
    ):
        pathname = url_to_path(request.url)
        resp = Response()
        resp.status_code = 200
        resp.url = request.url
        try:
            stats = os.stat(pathname)
        except OSError as exc:
            resp.status_code = 404
            resp.raw = exc
        else:
            modified = email.utils.formatdate(stats.st_mtime, usegmt=True)
            content_type = mimetypes.guess_type(pathname)[0] or "text/plain"
            resp.headers = CaseInsensitiveDict({
                "Content-Type": content_type,
                "Content-Length": stats.st_size,
                "Last-Modified": modified,
            })
            resp.raw = open(pathname, "rb")
            resp.close = resp.raw.close
        return resp
    def close(self):
        pass
class InsecureHTTPAdapter(HTTPAdapter):
    def cert_verify(
        self,
        conn,
        url,
        verify,
        cert,
    ):
        super().cert_verify(conn=conn, url=url, verify=False, cert=cert)
class InsecureCacheControlAdapter(CacheControlAdapter):
    def cert_verify(
        self,
        conn,
        url,
        verify,
        cert,
    ):
        super().cert_verify(conn=conn, url=url, verify=False, cert=cert)
class PipSession(requests.Session):
    timeout = None
    def __init__(
        self,
        *args,
        retries=0,
        cache=None,
        trusted_hosts=(),
        index_urls=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pip_trusted_origins = []
        self.headers["User-Agent"] = user_agent()
        self.auth = MultiDomainBasicAuth(index_urls=index_urls)
        retries = urllib3.Retry(
            total=retries,
            status_forcelist=[500, 503, 520, 527],
            backoff_factor=0.25,
        )
        insecure_adapter = InsecureHTTPAdapter(max_retries=retries)
        if cache:
            secure_adapter = CacheControlAdapter(
                cache=SafeFileCache(cache),
                max_retries=retries,
            )
            self._trusted_host_adapter = InsecureCacheControlAdapter(
                cache=SafeFileCache(cache),
                max_retries=retries,
            )
        else:
            secure_adapter = HTTPAdapter(max_retries=retries)
            self._trusted_host_adapter = insecure_adapter
        self.mount("https://", secure_adapter)
        self.mount("http://", insecure_adapter)
        self.mount("file://", LocalFSAdapter())
        for host in trusted_hosts:
            self.add_trusted_host(host, suppress_logging=True)
    def update_index_urls(self, new_index_urls):
        self.auth.index_urls = new_index_urls
    def add_trusted_host(self, host, source=None, suppress_logging=False):
        if not suppress_logging:
            msg = f'adding trusted host: {host!r}'
            if source is not None:
                msg += f' (from {source})'
            logger.info(msg)
        host_port = parse_netloc(host)
        if host_port not in self.pip_trusted_origins:
            self.pip_trusted_origins.append(host_port)
        self.mount(
            build_url_from_netloc(host) + '/',
            self._trusted_host_adapter
        )
        if not host_port[1]:
            self.mount(
                build_url_from_netloc(host) + ':',
                self._trusted_host_adapter
            )
    def iter_secure_origins(self):
        yield from SECURE_ORIGINS
        for host, port in self.pip_trusted_origins:
            yield ('*', host, '*' if port is None else port)
    def is_secure_origin(self, location):
        parsed = urllib.parse.urlparse(str(location))
        origin_protocol, origin_host, origin_port = (
            parsed.scheme, parsed.hostname, parsed.port,
        )
        origin_protocol = origin_protocol.rsplit('+', 1)[-1]
        for secure_origin in self.iter_secure_origins():
            secure_protocol, secure_host, secure_port = secure_origin
            if origin_protocol != secure_protocol and secure_protocol != "*":
                continue
            try:
                addr = ipaddress.ip_address(origin_host)
                network = ipaddress.ip_network(secure_host)
            except ValueError:
                if (
                    origin_host and
                    origin_host.lower() != secure_host.lower() and
                    secure_host != "*"
                ):
                    continue
            else:
                if addr not in network:
                    continue
            if (
                origin_port != secure_port and
                secure_port != "*" and
                secure_port is not None
            ):
                continue
            return True
        logger.warning(
            "The repository located at %s is not a trusted or secure host and "
            "is being ignored. If this repository is available via HTTPS we "
            "recommend you use HTTPS instead, otherwise you may silence "
            "this warning and allow it anyway with '--trusted-host %s'.",
            origin_host,
            origin_host,
        )
        return False
    def request(self, method, url, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return super().request(method, url, *args, **kwargs)
