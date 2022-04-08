
"""
    werkzeug.serving
    ~~~~~~~~~~~~~~~~
    There are many ways to serve a WSGI application.  While you're developing
    it you usually don't want a full blown webserver like Apache but a simple
    standalone one.  From Python 2.5 onwards there is the `wsgiref`_ server in
    the standard library.  If you're using older versions of Python you can
    download the package from the cheeseshop.
    However there are some caveats. Sourcecode won't reload itself when
    changed and each time you kill the server using ``^C`` you get an
    `KeyboardInterrupt` error.  While the latter is easy to solve the first
    one can be a pain in the ass in some situations.
    The easiest way is creating a small ``start-myproject.py`` that runs the
    application::
        from myproject import make_app
        from werkzeug.serving import run_simple
        app = make_app(...)
        run_simple('localhost', 8080, app, use_reloader=True)
    You can also pass it a `extra_files` keyword argument with a list of
    additional files (like configuration files) you want to observe.
    For bigger applications you should consider using `werkzeug.script`
    instead of a simple start file.
    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from __future__ import with_statement
import os
import socket
import sys
import signal
can_fork = hasattr(os, "fork")
try:
    import termcolor
except ImportError:
    termcolor = None
try:
    import ssl
except ImportError:
    class _SslDummy(object):
        def __getattr__(self, name):
            raise RuntimeError('SSL support unavailable')
    ssl = _SslDummy()
def _get_openssl_crypto_module():
    try:
        from OpenSSL import crypto
    except ImportError:
        raise TypeError('Using ad-hoc certificates requires the pyOpenSSL '
                        'library.')
    else:
        return crypto
try:
    import SocketServer as socketserver
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
except ImportError:
    import socketserver
    from http.server import HTTPServer, BaseHTTPRequestHandler
ThreadingMixIn = socketserver.ThreadingMixIn
if can_fork:
    ForkingMixIn = socketserver.ForkingMixIn
else:
    class ForkingMixIn(object):
        pass
import werkzeug
from werkzeug._internal import _log
from werkzeug._compat import PY2, WIN, reraise, wsgi_encoding_dance
from werkzeug.urls import url_parse, url_unquote
from werkzeug.exceptions import InternalServerError
LISTEN_QUEUE = 128
can_open_by_fd = not WIN and hasattr(socket, 'fromfd')
class WSGIRequestHandler(BaseHTTPRequestHandler, object):
    @property
    def server_version(self):
        return 'Werkzeug/' + werkzeug.__version__
    def make_environ(self):
        request_url = url_parse(self.path)
        def shutdown_server():
            self.server.shutdown_signal = True
        url_scheme = self.server.ssl_context is None and 'http' or 'https'
        path_info = url_unquote(request_url.path)
        environ = {
            'wsgi.version':         (1, 0),
            'wsgi.url_scheme':      url_scheme,
            'wsgi.input':           self.rfile,
            'wsgi.errors':          sys.stderr,
            'wsgi.multithread':     self.server.multithread,
            'wsgi.multiprocess':    self.server.multiprocess,
            'wsgi.run_once':        False,
            'werkzeug.server.shutdown': shutdown_server,
            'SERVER_SOFTWARE':      self.server_version,
            'REQUEST_METHOD':       self.command,
            'SCRIPT_NAME':          '',
            'PATH_INFO':            wsgi_encoding_dance(path_info),
            'QUERY_STRING':         wsgi_encoding_dance(request_url.query),
            'REMOTE_ADDR':          self.address_string(),
            'REMOTE_PORT':          self.port_integer(),
            'SERVER_NAME':          self.server.server_address[0],
            'SERVER_PORT':          str(self.server.server_address[1]),
            'SERVER_PROTOCOL':      self.request_version
        }
        for key, value in self.headers.items():
            key = key.upper().replace('-', '_')
            if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                key = 'HTTP_' + key
            environ[key] = value
        if request_url.scheme and request_url.netloc:
            environ['HTTP_HOST'] = request_url.netloc
        return environ
    def run_wsgi(self):
        if self.headers.get('Expect', '').lower().strip() == '100-continue':
            self.wfile.write(b'HTTP/1.1 100 Continue\r\n\r\n')
        self.environ = environ = self.make_environ()
        headers_set = []
        headers_sent = []
        def write(data):
            assert headers_set, 'write() before start_response'
            if not headers_sent:
                status, response_headers = headers_sent[:] = headers_set
                try:
                    code, msg = status.split(None, 1)
                except ValueError:
                    code, msg = status, ""
                self.send_response(int(code), msg)
                header_keys = set()
                for key, value in response_headers:
                    self.send_header(key, value)
                    key = key.lower()
                    header_keys.add(key)
                if 'content-length' not in header_keys:
                    self.close_connection = True
                    self.send_header('Connection', 'close')
                if 'server' not in header_keys:
                    self.send_header('Server', self.version_string())
                if 'date' not in header_keys:
                    self.send_header('Date', self.date_time_string())
                self.end_headers()
            assert isinstance(data, bytes), 'applications must write bytes'
            self.wfile.write(data)
            self.wfile.flush()
        def start_response(status, response_headers, exc_info=None):
            if exc_info:
                try:
                    if headers_sent:
                        reraise(*exc_info)
                finally:
                    exc_info = None
            elif headers_set:
                raise AssertionError('Headers already set')
            headers_set[:] = [status, response_headers]
            return write
        def execute(app):
            application_iter = app(environ, start_response)
            try:
                for data in application_iter:
                    write(data)
                if not headers_sent:
                    write(b'')
            finally:
                if hasattr(application_iter, 'close'):
                    application_iter.close()
                application_iter = None
        try:
            execute(self.server.app)
        except (socket.error, socket.timeout) as e:
            self.connection_dropped(e, environ)
        except Exception:
            if self.server.passthrough_errors:
                raise
            from werkzeug.debug.tbtools import get_current_traceback
            traceback = get_current_traceback(ignore_system_exceptions=True)
            try:
                if not headers_sent:
                    del headers_set[:]
                execute(InternalServerError())
            except Exception:
                pass
            self.server.log('error', 'Error on request:\n%s',
                            traceback.plaintext)
    def handle(self):
        rv = None
        try:
            rv = BaseHTTPRequestHandler.handle(self)
        except (socket.error, socket.timeout) as e:
            self.connection_dropped(e)
        except Exception:
            if self.server.ssl_context is None or not is_ssl_error():
                raise
        if self.server.shutdown_signal:
            self.initiate_shutdown()
        return rv
    def initiate_shutdown(self):
        sig = getattr(signal, 'SIGKILL', signal.SIGTERM)
        if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
            os.kill(os.getpid(), sig)
        self.server._BaseServer__shutdown_request = True
        self.server._BaseServer__serving = False
    def connection_dropped(self, error, environ=None):
    def handle_one_request(self):
        self.raw_requestline = self.rfile.readline()
        if not self.raw_requestline:
            self.close_connection = 1
        elif self.parse_request():
            return self.run_wsgi()
    def send_response(self, code, message=None):
        self.log_request(code)
        if message is None:
            message = code in self.responses and self.responses[code][0] or ''
        if self.request_version != 'HTTP/0.9':
            hdr = "%s %d %s\r\n" % (self.protocol_version, code, message)
            self.wfile.write(hdr.encode('ascii'))
    def version_string(self):
        return BaseHTTPRequestHandler.version_string(self).strip()
    def address_string(self):
        if getattr(self, 'environ', None):
            return self.environ['REMOTE_ADDR']
        else:
            return self.client_address[0]
    def port_integer(self):
        return self.client_address[1]
    def log_request(self, code='-', size='-'):
        msg = self.requestline
        code = str(code)
        if termcolor:
            color = termcolor.colored
            if code[0] == '1':
                msg = color(msg, attrs=['bold'])
            if code[0] == '2':
                msg = color(msg, color='white')
            elif code == '304':
                msg = color(msg, color='cyan')
            elif code[0] == '3':
                msg = color(msg, color='green')
            elif code == '404':
                msg = color(msg, color='yellow')
            elif code[0] == '4':
                msg = color(msg, color='red', attrs=['bold'])
            else:
                msg = color(msg, color='magenta', attrs=['bold'])
        self.log('info', '"%s" %s %s', msg, code, size)
    def log_error(self, *args):
        self.log('error', *args)
    def log_message(self, format, *args):
        self.log('info', format, *args)
    def log(self, type, message, *args):
        _log(type, '%s - - [%s] %s\n' % (self.address_string(),
                                         self.log_date_time_string(),
                                         message % args))
BaseRequestHandler = WSGIRequestHandler
def generate_adhoc_ssl_pair(cn=None):
    from random import random
    crypto = _get_openssl_crypto_module()
    if cn is None:
        cn = '*'
    cert = crypto.X509()
    cert.set_serial_number(int(random() * sys.maxsize))
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(60 * 60 * 24 * 365)
    subject = cert.get_subject()
    subject.CN = cn
    subject.O = 'Dummy Certificate'
    issuer = cert.get_issuer()
    issuer.CN = 'Untrusted Authority'
    issuer.O = 'Self-Signed'
    pkey = crypto.PKey()
    pkey.generate_key(crypto.TYPE_RSA, 2048)
    cert.set_pubkey(pkey)
    cert.sign(pkey, 'sha256')
    return cert, pkey
def make_ssl_devcert(base_path, host=None, cn=None):
    from OpenSSL import crypto
    if host is not None:
        cn = '*.%s/CN=%s' % (host, host)
    cert, pkey = generate_adhoc_ssl_pair(cn=cn)
    cert_file = base_path + '.crt'
    pkey_file = base_path + '.key'
    with open(cert_file, 'wb') as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open(pkey_file, 'wb') as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, pkey))
    return cert_file, pkey_file
def generate_adhoc_ssl_context():
    crypto = _get_openssl_crypto_module()
    import tempfile
    import atexit
    cert, pkey = generate_adhoc_ssl_pair()
    cert_handle, cert_file = tempfile.mkstemp()
    pkey_handle, pkey_file = tempfile.mkstemp()
    atexit.register(os.remove, pkey_file)
    atexit.register(os.remove, cert_file)
    os.write(cert_handle, crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    os.write(pkey_handle, crypto.dump_privatekey(crypto.FILETYPE_PEM, pkey))
    os.close(cert_handle)
    os.close(pkey_handle)
    ctx = load_ssl_context(cert_file, pkey_file)
    return ctx
def load_ssl_context(cert_file, pkey_file=None, protocol=None):
    if protocol is None:
        protocol = ssl.PROTOCOL_SSLv23
    ctx = _SSLContext(protocol)
    ctx.load_cert_chain(cert_file, pkey_file)
    return ctx
class _SSLContext(object):
    def __init__(self, protocol):
        self._protocol = protocol
        self._certfile = None
        self._keyfile = None
        self._password = None
    def load_cert_chain(self, certfile, keyfile=None, password=None):
        self._certfile = certfile
        self._keyfile = keyfile or certfile
        self._password = password
    def wrap_socket(self, sock, **kwargs):
        return ssl.wrap_socket(sock, keyfile=self._keyfile,
                               certfile=self._certfile,
                               ssl_version=self._protocol, **kwargs)
def is_ssl_error(error=None):
    exc_types = (ssl.SSLError,)
    try:
        from OpenSSL.SSL import Error
        exc_types += (Error,)
    except ImportError:
        pass
    if error is None:
        error = sys.exc_info()[1]
    return isinstance(error, exc_types)
def select_ip_version(host, port):
    if ':' in host and hasattr(socket, 'AF_INET6'):
        return socket.AF_INET6
    return socket.AF_INET
class BaseWSGIServer(HTTPServer, object):
    multithread = False
    multiprocess = False
    request_queue_size = LISTEN_QUEUE
    def __init__(self, host, port, app, handler=None,
                 passthrough_errors=False, ssl_context=None, fd=None):
        if handler is None:
            handler = WSGIRequestHandler
        self.address_family = select_ip_version(host, port)
        if fd is not None:
            real_sock = socket.fromfd(fd, self.address_family,
                                      socket.SOCK_STREAM)
            port = 0
        HTTPServer.__init__(self, (host, int(port)), handler)
        self.app = app
        self.passthrough_errors = passthrough_errors
        self.shutdown_signal = False
        self.host = host
        self.port = self.socket.getsockname()[1]
        if fd is not None:
            self.socket.close()
            self.socket = real_sock
            self.server_address = self.socket.getsockname()
        if ssl_context is not None:
            if isinstance(ssl_context, tuple):
                ssl_context = load_ssl_context(*ssl_context)
            if ssl_context == 'adhoc':
                ssl_context = generate_adhoc_ssl_context()
            sock = self.socket
            if PY2 and not isinstance(sock, socket.socket):
                sock = socket.socket(sock.family, sock.type, sock.proto, sock)
            self.socket = ssl_context.wrap_socket(sock, server_side=True)
            self.ssl_context = ssl_context
        else:
            self.ssl_context = None
    def log(self, type, message, *args):
        _log(type, message, *args)
    def serve_forever(self):
        self.shutdown_signal = False
        try:
            HTTPServer.serve_forever(self)
        except KeyboardInterrupt:
            pass
        finally:
            self.server_close()
    def handle_error(self, request, client_address):
        if self.passthrough_errors:
            raise
        return HTTPServer.handle_error(self, request, client_address)
    def get_request(self):
        con, info = self.socket.accept()
        return con, info
class ThreadedWSGIServer(ThreadingMixIn, BaseWSGIServer):
    multithread = True
    daemon_threads = True
class ForkingWSGIServer(ForkingMixIn, BaseWSGIServer):
    multiprocess = True
    def __init__(self, host, port, app, processes=40, handler=None,
                 passthrough_errors=False, ssl_context=None, fd=None):
        if not can_fork:
            raise ValueError('Your platform does not support forking.')
        BaseWSGIServer.__init__(self, host, port, app, handler,
                                passthrough_errors, ssl_context, fd)
        self.max_children = processes
def make_server(host=None, port=None, app=None, threaded=False, processes=1,
                request_handler=None, passthrough_errors=False,
                ssl_context=None, fd=None):
    if threaded and processes > 1:
        raise ValueError("cannot have a multithreaded and "
                         "multi process server.")
    elif threaded:
        return ThreadedWSGIServer(host, port, app, request_handler,
                                  passthrough_errors, ssl_context, fd=fd)
    elif processes > 1:
        return ForkingWSGIServer(host, port, app, processes, request_handler,
                                 passthrough_errors, ssl_context, fd=fd)
    else:
        return BaseWSGIServer(host, port, app, request_handler,
                              passthrough_errors, ssl_context, fd=fd)
def is_running_from_reloader():
    return os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
def run_simple(hostname, port, application, use_reloader=False,
               use_debugger=False, use_evalex=True,
               extra_files=None, reloader_interval=1,
               reloader_type='auto', threaded=False,
               processes=1, request_handler=None, static_files=None,
               passthrough_errors=False, ssl_context=None):
    if use_debugger:
        from werkzeug.debug import DebuggedApplication
        application = DebuggedApplication(application, use_evalex)
    if static_files:
        from werkzeug.wsgi import SharedDataMiddleware
        application = SharedDataMiddleware(application, static_files)
    def log_startup(sock):
        display_hostname = hostname not in ('', '*') and hostname or 'localhost'
        if ':' in display_hostname:
            display_hostname = '[%s]' % display_hostname
        quit_msg = '(Press CTRL+C to quit)'
        port = sock.getsockname()[1]
        _log('info', ' * Running on %s://%s:%d/ %s',
             ssl_context is None and 'http' or 'https',
             display_hostname, port, quit_msg)
    def inner():
        try:
            fd = int(os.environ['WERKZEUG_SERVER_FD'])
        except (LookupError, ValueError):
            fd = None
        srv = make_server(hostname, port, application, threaded,
                          processes, request_handler,
                          passthrough_errors, ssl_context,
                          fd=fd)
        if fd is None:
            log_startup(srv.socket)
        srv.serve_forever()
    if use_reloader:
        if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
            if port == 0 and not can_open_by_fd:
                raise ValueError('Cannot bind to a random port with enabled '
                                 'reloader if the Python interpreter does '
                                 'not support socket opening by fd.')
            address_family = select_ip_version(hostname, port)
            s = socket.socket(address_family, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((hostname, port))
            if hasattr(s, 'set_inheritable'):
                s.set_inheritable(True)
            if can_open_by_fd:
                os.environ['WERKZEUG_SERVER_FD'] = str(s.fileno())
                s.listen(LISTEN_QUEUE)
                log_startup(s)
            else:
                s.close()
        from werkzeug._reloader import run_with_reloader
        run_with_reloader(inner, extra_files, reloader_interval,
                          reloader_type)
    else:
        inner()
def run_with_reloader(*args, **kwargs):
    from werkzeug._reloader import run_with_reloader
    return run_with_reloader(*args, **kwargs)
def main():
    import optparse
    from werkzeug.utils import import_string
    parser = optparse.OptionParser(
        usage='Usage: %prog [options] app_module:app_object')
    parser.add_option('-b', '--bind', dest='address',
                      help='The hostname:port the app should listen on.')
    parser.add_option('-d', '--debug', dest='use_debugger',
                      action='store_true', default=False,
                      help='Use Werkzeug\'s debugger.')
    parser.add_option('-r', '--reload', dest='use_reloader',
                      action='store_true', default=False,
                      help='Reload Python process if modules change.')
    options, args = parser.parse_args()
    hostname, port = None, None
    if options.address:
        address = options.address.split(':')
        hostname = address[0]
        if len(address) > 1:
            port = address[1]
    if len(args) != 1:
        sys.stdout.write('No application supplied, or too much. See --help\n')
        sys.exit(1)
    app = import_string(args[0])
    run_simple(
        hostname=(hostname or '127.0.0.1'), port=int(port or 5000),
        application=app, use_reloader=options.use_reloader,
        use_debugger=options.use_debugger
    )
if __name__ == '__main__':
    main()
