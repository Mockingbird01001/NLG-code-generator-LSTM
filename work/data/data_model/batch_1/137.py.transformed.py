
"""
    flask.ctx
    ~~~~~~~~~
    Implements the objects required to keep the context.
    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""
import sys
from functools import update_wrapper
from werkzeug.exceptions import HTTPException
from .globals import _request_ctx_stack, _app_ctx_stack
from .signals import appcontext_pushed, appcontext_popped
from ._compat import BROKEN_PYPY_CTXMGR_EXIT, reraise
_sentinel = object()
class _AppCtxGlobals(object):
    def get(self, name, default=None):
        return self.__dict__.get(name, default)
    def pop(self, name, default=_sentinel):
        if default is _sentinel:
            return self.__dict__.pop(name)
        else:
            return self.__dict__.pop(name, default)
    def setdefault(self, name, default=None):
        return self.__dict__.setdefault(name, default)
    def __contains__(self, item):
        return item in self.__dict__
    def __iter__(self):
        return iter(self.__dict__)
    def __repr__(self):
        top = _app_ctx_stack.top
        if top is not None:
            return '<flask.g of %r>' % top.app.name
        return object.__repr__(self)
def after_this_request(f):
    _request_ctx_stack.top._after_request_functions.append(f)
    return f
def copy_current_request_context(f):
    top = _request_ctx_stack.top
    if top is None:
        raise RuntimeError('This decorator can only be used at local scopes '
            'when a request context is on the stack.  For instance within '
            'view functions.')
    reqctx = top.copy()
    def wrapper(*args, **kwargs):
        with reqctx:
            return f(*args, **kwargs)
    return update_wrapper(wrapper, f)
def has_request_context():
    return _request_ctx_stack.top is not None
def has_app_context():
    return _app_ctx_stack.top is not None
class AppContext(object):
    def __init__(self, app):
        self.app = app
        self.url_adapter = app.create_url_adapter(None)
        self.g = app.app_ctx_globals_class()
        self._refcnt = 0
    def push(self):
        self._refcnt += 1
        if hasattr(sys, 'exc_clear'):
            sys.exc_clear()
        _app_ctx_stack.push(self)
        appcontext_pushed.send(self.app)
    def pop(self, exc=_sentinel):
        try:
            self._refcnt -= 1
            if self._refcnt <= 0:
                if exc is _sentinel:
                    exc = sys.exc_info()[1]
                self.app.do_teardown_appcontext(exc)
        finally:
            rv = _app_ctx_stack.pop()
        assert rv is self, 'Popped wrong app context.  (%r instead of %r)'            % (rv, self)
        appcontext_popped.send(self.app)
    def __enter__(self):
        self.push()
        return self
    def __exit__(self, exc_type, exc_value, tb):
        self.pop(exc_value)
        if BROKEN_PYPY_CTXMGR_EXIT and exc_type is not None:
            reraise(exc_type, exc_value, tb)
class RequestContext(object):
    def __init__(self, app, environ, request=None):
        self.app = app
        if request is None:
            request = app.request_class(environ)
        self.request = request
        self.url_adapter = app.create_url_adapter(self.request)
        self.flashes = None
        self.session = None
        self._implicit_app_ctx_stack = []
        self.preserved = False
        self._preserved_exc = None
        self._after_request_functions = []
        self.match_request()
    def _get_g(self):
        return _app_ctx_stack.top.g
    def _set_g(self, value):
        _app_ctx_stack.top.g = value
    g = property(_get_g, _set_g)
    del _get_g, _set_g
    def copy(self):
        return self.__class__(self.app,
            environ=self.request.environ,
            request=self.request
        )
    def match_request(self):
        try:
            url_rule, self.request.view_args =                self.url_adapter.match(return_rule=True)
            self.request.url_rule = url_rule
        except HTTPException as e:
            self.request.routing_exception = e
    def push(self):
        top = _request_ctx_stack.top
        if top is not None and top.preserved:
            top.pop(top._preserved_exc)
        app_ctx = _app_ctx_stack.top
        if app_ctx is None or app_ctx.app != self.app:
            app_ctx = self.app.app_context()
            app_ctx.push()
            self._implicit_app_ctx_stack.append(app_ctx)
        else:
            self._implicit_app_ctx_stack.append(None)
        if hasattr(sys, 'exc_clear'):
            sys.exc_clear()
        _request_ctx_stack.push(self)
        self.session = self.app.open_session(self.request)
        if self.session is None:
            self.session = self.app.make_null_session()
    def pop(self, exc=_sentinel):
        app_ctx = self._implicit_app_ctx_stack.pop()
        try:
            clear_request = False
            if not self._implicit_app_ctx_stack:
                self.preserved = False
                self._preserved_exc = None
                if exc is _sentinel:
                    exc = sys.exc_info()[1]
                self.app.do_teardown_request(exc)
                if hasattr(sys, 'exc_clear'):
                    sys.exc_clear()
                request_close = getattr(self.request, 'close', None)
                if request_close is not None:
                    request_close()
                clear_request = True
        finally:
            rv = _request_ctx_stack.pop()
            if clear_request:
                rv.request.environ['werkzeug.request'] = None
            if app_ctx is not None:
                app_ctx.pop(exc)
            assert rv is self, 'Popped wrong request context.  '                '(%r instead of %r)' % (rv, self)
    def auto_pop(self, exc):
        if self.request.environ.get('flask._preserve_context') or           (exc is not None and self.app.preserve_context_on_exception):
            self.preserved = True
            self._preserved_exc = exc
        else:
            self.pop(exc)
    def __enter__(self):
        self.push()
        return self
    def __exit__(self, exc_type, exc_value, tb):
        self.auto_pop(exc_value)
        if BROKEN_PYPY_CTXMGR_EXIT and exc_type is not None:
            reraise(exc_type, exc_value, tb)
    def __repr__(self):
        return '<%s \'%s\' [%s] of %s>' % (
            self.__class__.__name__,
            self.request.url,
            self.request.method,
            self.app.name,
        )
