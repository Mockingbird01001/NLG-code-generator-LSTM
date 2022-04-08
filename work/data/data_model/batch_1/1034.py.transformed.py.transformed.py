from __future__ import absolute_import
import hmac
import os
import sys
import warnings
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from ..exceptions import (
    InsecurePlatformWarning,
    ProxySchemeUnsupported,
    SNIMissingWarning,
    SSLError,
)
from ..packages import six
from .url import BRACELESS_IPV6_ADDRZ_RE, IPV4_RE
SSLContext = None
SSLTransport = None
HAS_SNI = False
IS_PYOPENSSL = False
IS_SECURETRANSPORT = False
ALPN_PROTOCOLS = ["http/1.1"]
HASHFUNC_MAP = {32: md5, 40: sha1, 64: sha256}
def _const_compare_digest_backport(a, b):
    result = abs(len(a) - len(b))
    for left, right in zip(bytearray(a), bytearray(b)):
        result |= left ^ right
    return result == 0
_const_compare_digest = getattr(hmac, "compare_digest", _const_compare_digest_backport)
try:
    import ssl
    from ssl import CERT_REQUIRED, wrap_socket
except ImportError:
    pass
try:
    from ssl import HAS_SNI
except ImportError:
    pass
try:
    from .ssltransport import SSLTransport
except ImportError:
    pass
try:
    from ssl import PROTOCOL_TLS
    PROTOCOL_SSLv23 = PROTOCOL_TLS
except ImportError:
    try:
        from ssl import PROTOCOL_SSLv23 as PROTOCOL_TLS
        PROTOCOL_SSLv23 = PROTOCOL_TLS
    except ImportError:
        PROTOCOL_SSLv23 = PROTOCOL_TLS = 2
try:
    from ssl import OP_NO_COMPRESSION, OP_NO_SSLv2, OP_NO_SSLv3
except ImportError:
    OP_NO_SSLv2, OP_NO_SSLv3 = 0x1000000, 0x2000000
    OP_NO_COMPRESSION = 0x20000
try:
    from ssl import OP_NO_TICKET
except ImportError:
    OP_NO_TICKET = 0x4000
DEFAULT_CIPHERS = ":".join(
    [
        "ECDHE+AESGCM",
        "ECDHE+CHACHA20",
        "DHE+AESGCM",
        "DHE+CHACHA20",
        "ECDH+AESGCM",
        "DH+AESGCM",
        "ECDH+AES",
        "DH+AES",
        "RSA+AESGCM",
        "RSA+AES",
        "!aNULL",
        "!eNULL",
        "!MD5",
        "!DSS",
    ]
)
try:
    from ssl import SSLContext
except ImportError:
    class SSLContext(object):
        def __init__(self, protocol_version):
            self.protocol = protocol_version
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.ca_certs = None
            self.options = 0
            self.certfile = None
            self.keyfile = None
            self.ciphers = None
        def load_cert_chain(self, certfile, keyfile):
            self.certfile = certfile
            self.keyfile = keyfile
        def load_verify_locations(self, cafile=None, capath=None, cadata=None):
            self.ca_certs = cafile
            if capath is not None:
                raise SSLError("CA directories not supported in older Pythons")
            if cadata is not None:
                raise SSLError("CA data not supported in older Pythons")
        def set_ciphers(self, cipher_suite):
            self.ciphers = cipher_suite
        def wrap_socket(self, socket, server_hostname=None, server_side=False):
            warnings.warn(
                "A true SSLContext object is not available. This prevents "
                "urllib3 from configuring SSL appropriately and may cause "
                "certain SSL connections to fail. You can upgrade to a newer "
                "version of Python to solve this. For more information, see "
                "https://urllib3.readthedocs.io/en/latest/advanced-usage.html"
                InsecurePlatformWarning,
            )
            kwargs = {
                "keyfile": self.keyfile,
                "certfile": self.certfile,
                "ca_certs": self.ca_certs,
                "cert_reqs": self.verify_mode,
                "ssl_version": self.protocol,
                "server_side": server_side,
            }
            return wrap_socket(socket, ciphers=self.ciphers, **kwargs)
def assert_fingerprint(cert, fingerprint):
    fingerprint = fingerprint.replace(":", "").lower()
    digest_length = len(fingerprint)
    hashfunc = HASHFUNC_MAP.get(digest_length)
    if not hashfunc:
        raise SSLError("Fingerprint of invalid length: {0}".format(fingerprint))
    fingerprint_bytes = unhexlify(fingerprint.encode())
    cert_digest = hashfunc(cert).digest()
    if not _const_compare_digest(cert_digest, fingerprint_bytes):
        raise SSLError(
            'Fingerprints did not match. Expected "{0}", got "{1}".'.format(
                fingerprint, hexlify(cert_digest)
            )
        )
def resolve_cert_reqs(candidate):
    if candidate is None:
        return CERT_REQUIRED
    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "CERT_" + candidate)
        return res
    return candidate
def resolve_ssl_version(candidate):
    if candidate is None:
        return PROTOCOL_TLS
    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "PROTOCOL_" + candidate)
        return res
    return candidate
def create_urllib3_context(
    ssl_version=None, cert_reqs=None, options=None, ciphers=None
):
    context = SSLContext(ssl_version or PROTOCOL_TLS)
    context.set_ciphers(ciphers or DEFAULT_CIPHERS)
    cert_reqs = ssl.CERT_REQUIRED if cert_reqs is None else cert_reqs
    if options is None:
        options = 0
        options |= OP_NO_SSLv2
        options |= OP_NO_SSLv3
        options |= OP_NO_COMPRESSION
        options |= OP_NO_TICKET
    context.options |= options
    if (cert_reqs == ssl.CERT_REQUIRED or sys.version_info >= (3, 7, 4)) and getattr(
        context, "post_handshake_auth", None
    ) is not None:
        context.post_handshake_auth = True
    context.verify_mode = cert_reqs
    if (
        getattr(context, "check_hostname", None) is not None
    ):
        context.check_hostname = False
    if hasattr(context, "keylog_filename"):
        sslkeylogfile = os.environ.get("SSLKEYLOGFILE")
        if sslkeylogfile:
            context.keylog_filename = sslkeylogfile
    return context
def ssl_wrap_socket(
    sock,
    keyfile=None,
    certfile=None,
    cert_reqs=None,
    ca_certs=None,
    server_hostname=None,
    ssl_version=None,
    ciphers=None,
    ssl_context=None,
    ca_cert_dir=None,
    key_password=None,
    ca_cert_data=None,
    tls_in_tls=False,
):
    context = ssl_context
    if context is None:
        context = create_urllib3_context(ssl_version, cert_reqs, ciphers=ciphers)
    if ca_certs or ca_cert_dir or ca_cert_data:
        try:
            context.load_verify_locations(ca_certs, ca_cert_dir, ca_cert_data)
        except (IOError, OSError) as e:
            raise SSLError(e)
    elif ssl_context is None and hasattr(context, "load_default_certs"):
        context.load_default_certs()
    if keyfile and key_password is None and _is_key_file_encrypted(keyfile):
        raise SSLError("Client private key is encrypted, password is required")
    if certfile:
        if key_password is None:
            context.load_cert_chain(certfile, keyfile)
        else:
            context.load_cert_chain(certfile, keyfile, key_password)
    try:
        if hasattr(context, "set_alpn_protocols"):
            context.set_alpn_protocols(ALPN_PROTOCOLS)
    except NotImplementedError:
        pass
    use_sni_hostname = server_hostname and not is_ipaddress(server_hostname)
    send_sni = (use_sni_hostname and HAS_SNI) or (
        IS_SECURETRANSPORT and server_hostname
    )
    if not HAS_SNI and use_sni_hostname:
        warnings.warn(
            "An HTTPS request has been made, but the SNI (Server Name "
            "Indication) extension to TLS is not available on this platform. "
            "This may cause the server to present an incorrect TLS "
            "certificate, which can cause validation failures. You can upgrade to "
            "a newer version of Python to solve this. For more information, see "
            "https://urllib3.readthedocs.io/en/latest/advanced-usage.html"
            SNIMissingWarning,
        )
    if send_sni:
        ssl_sock = _ssl_wrap_socket_impl(
            sock, context, tls_in_tls, server_hostname=server_hostname
        )
    else:
        ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls)
    return ssl_sock
def is_ipaddress(hostname):
    if not six.PY2 and isinstance(hostname, bytes):
        hostname = hostname.decode("ascii")
    return bool(IPV4_RE.match(hostname) or BRACELESS_IPV6_ADDRZ_RE.match(hostname))
def _is_key_file_encrypted(key_file):
    with open(key_file, "r") as f:
        for line in f:
            if "ENCRYPTED" in line:
                return True
    return False
def _ssl_wrap_socket_impl(sock, ssl_context, tls_in_tls, server_hostname=None):
    if tls_in_tls:
        if not SSLTransport:
            raise ProxySchemeUnsupported(
                "TLS in TLS requires support for the 'ssl' module"
            )
        SSLTransport._validate_ssl_context_for_tls_in_tls(ssl_context)
        return SSLTransport(sock, ssl_context, server_hostname)
    if server_hostname:
        return ssl_context.wrap_socket(sock, server_hostname=server_hostname)
    else:
        return ssl_context.wrap_socket(sock)
