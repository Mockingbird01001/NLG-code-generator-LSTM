
try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote
from werkzeug.http import parse_options_header, parse_cache_control_header,    parse_set_header
from werkzeug.useragents import UserAgent
from werkzeug.datastructures import Headers, ResponseCacheControl
class CGIRootFix(object):
    def __init__(self, app, app_root='/'):
        self.app = app
        self.app_root = app_root
    def __call__(self, environ, start_response):
        if 'SERVER_SOFTWARE' not in environ or           environ['SERVER_SOFTWARE'] < 'lighttpd/1.4.28':
            environ['PATH_INFO'] = environ.get('SCRIPT_NAME', '') +                environ.get('PATH_INFO', '')
        environ['SCRIPT_NAME'] = self.app_root.strip('/')
        return self.app(environ, start_response)
LighttpdCGIRootFix = CGIRootFix
class PathInfoFromRequestUriFix(object):
    def __init__(self, app):
        self.app = app
    def __call__(self, environ, start_response):
        for key in 'REQUEST_URL', 'REQUEST_URI', 'UNENCODED_URL':
            if key not in environ:
                continue
            request_uri = unquote(environ[key])
            script_name = unquote(environ.get('SCRIPT_NAME', ''))
            if request_uri.startswith(script_name):
                environ['PATH_INFO'] = request_uri[len(script_name):]                    .split('?', 1)[0]
                break
        return self.app(environ, start_response)
class ProxyFix(object):
    def __init__(self, app, num_proxies=1):
        self.app = app
        self.num_proxies = num_proxies
    def get_remote_addr(self, forwarded_for):
        if len(forwarded_for) >= self.num_proxies:
            return forwarded_for[-self.num_proxies]
    def __call__(self, environ, start_response):
        getter = environ.get
        forwarded_proto = getter('HTTP_X_FORWARDED_PROTO', '')
        forwarded_for = getter('HTTP_X_FORWARDED_FOR', '').split(',')
        forwarded_host = getter('HTTP_X_FORWARDED_HOST', '')
        environ.update({
            'werkzeug.proxy_fix.orig_wsgi_url_scheme':  getter('wsgi.url_scheme'),
            'werkzeug.proxy_fix.orig_remote_addr':      getter('REMOTE_ADDR'),
            'werkzeug.proxy_fix.orig_http_host':        getter('HTTP_HOST')
        })
        forwarded_for = [x for x in [x.strip() for x in forwarded_for] if x]
        remote_addr = self.get_remote_addr(forwarded_for)
        if remote_addr is not None:
            environ['REMOTE_ADDR'] = remote_addr
        if forwarded_host:
            environ['HTTP_HOST'] = forwarded_host
        if forwarded_proto:
            environ['wsgi.url_scheme'] = forwarded_proto
        return self.app(environ, start_response)
class HeaderRewriterFix(object):
    def __init__(self, app, remove_headers=None, add_headers=None):
        self.app = app
        self.remove_headers = set(x.lower() for x in (remove_headers or ()))
        self.add_headers = list(add_headers or ())
    def __call__(self, environ, start_response):
        def rewriting_start_response(status, headers, exc_info=None):
            new_headers = []
            for key, value in headers:
                if key.lower() not in self.remove_headers:
                    new_headers.append((key, value))
            new_headers += self.add_headers
            return start_response(status, new_headers, exc_info)
        return self.app(environ, rewriting_start_response)
class InternetExplorerFix(object):
    def __init__(self, app, fix_vary=True, fix_attach=True):
        self.app = app
        self.fix_vary = fix_vary
        self.fix_attach = fix_attach
    def fix_headers(self, environ, headers, status=None):
        if self.fix_vary:
            header = headers.get('content-type', '')
            mimetype, options = parse_options_header(header)
            if mimetype not in ('text/html', 'text/plain', 'text/sgml'):
                headers.pop('vary', None)
        if self.fix_attach and 'content-disposition' in headers:
            pragma = parse_set_header(headers.get('pragma', ''))
            pragma.discard('no-cache')
            header = pragma.to_header()
            if not header:
                headers.pop('pragma', '')
            else:
                headers['Pragma'] = header
            header = headers.get('cache-control', '')
            if header:
                cc = parse_cache_control_header(header,
                                                cls=ResponseCacheControl)
                cc.no_cache = None
                cc.no_store = False
                header = cc.to_header()
                if not header:
                    headers.pop('cache-control', '')
                else:
                    headers['Cache-Control'] = header
    def run_fixed(self, environ, start_response):
        def fixing_start_response(status, headers, exc_info=None):
            headers = Headers(headers)
            self.fix_headers(environ, headers, status)
            return start_response(status, headers.to_wsgi_list(), exc_info)
        return self.app(environ, fixing_start_response)
    def __call__(self, environ, start_response):
        ua = UserAgent(environ)
        if ua.browser != 'msie':
            return self.app(environ, start_response)
        return self.run_fixed(environ, start_response)
