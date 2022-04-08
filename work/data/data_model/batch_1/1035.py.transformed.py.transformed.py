
try:
    import ssl
except ImportError:
    pass
class SSLContext(object):
    __slots__ = ('_cafile', '_certfile',
                 '_keyfile', '_protocol', '_verify_mode')
    def __init__(self, protocol):
        self._cafile = None
        self._certfile = None
        self._keyfile = None
        self._protocol = protocol
        self._verify_mode = ssl.CERT_NONE
    @property
    def protocol(self):
        return self._protocol
    def __get_verify_mode(self):
        return self._verify_mode
    def __set_verify_mode(self, value):
        self._verify_mode = value
    verify_mode = property(__get_verify_mode, __set_verify_mode)
    def load_cert_chain(self, certfile, keyfile=None):
        self._certfile = certfile
        self._keyfile = keyfile
    def load_verify_locations(self, cafile=None, dummy=None):
        self._cafile = cafile
    def wrap_socket(self, sock, server_side=False,
                    do_handshake_on_connect=True,
                    suppress_ragged_eofs=True, dummy=None):
        return ssl.wrap_socket(sock, keyfile=self._keyfile,
                               certfile=self._certfile,
                               server_side=server_side,
                               cert_reqs=self._verify_mode,
                               ssl_version=self._protocol,
                               ca_certs=self._cafile,
                               do_handshake_on_connect=do_handshake_on_connect,
                               suppress_ragged_eofs=suppress_ragged_eofs)
