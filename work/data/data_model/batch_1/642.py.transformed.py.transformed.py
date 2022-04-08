
'''
    flask_login.login_manager
    -------------------------
    The LoginManager class.
'''
import warnings
from datetime import datetime
from flask import (_request_ctx_stack, abort, current_app, flash, redirect,
                   request, session)
from ._compat import text_type
from .config import (COOKIE_NAME, COOKIE_DURATION, COOKIE_SECURE,
                     COOKIE_HTTPONLY, LOGIN_MESSAGE, LOGIN_MESSAGE_CATEGORY,
                     REFRESH_MESSAGE, REFRESH_MESSAGE_CATEGORY, ID_ATTRIBUTE,
                     AUTH_HEADER_NAME, SESSION_KEYS)
from .mixins import AnonymousUserMixin
from .signals import (user_loaded_from_cookie, user_loaded_from_header,
                      user_loaded_from_request, user_unauthorized,
                      user_needs_refresh, user_accessed, session_protected)
from .utils import (_get_user, login_url, _create_identifier,
                    _user_context_processor, encode_cookie, decode_cookie)
class LoginManager(object):
    def __init__(self, app=None, add_context_processor=True):
        self.anonymous_user = AnonymousUserMixin
        self.login_view = None
        self.blueprint_login_views = {}
        self.login_message = LOGIN_MESSAGE
        self.login_message_category = LOGIN_MESSAGE_CATEGORY
        self.refresh_view = None
        self.needs_refresh_message = REFRESH_MESSAGE
        self.needs_refresh_message_category = REFRESH_MESSAGE_CATEGORY
        self.session_protection = 'basic'
        self.localize_callback = None
        self.user_callback = None
        self.unauthorized_callback = None
        self.needs_refresh_callback = None
        self.id_attribute = ID_ATTRIBUTE
        self.header_callback = None
        self.request_callback = None
        if app is not None:
            self.init_app(app, add_context_processor)
    def setup_app(self, app, add_context_processor=True):
        warnings.warn('Warning setup_app is deprecated. Please use init_app.',
                      DeprecationWarning)
        self.init_app(app, add_context_processor)
    def init_app(self, app, add_context_processor=True):
        app.login_manager = self
        app.after_request(self._update_remember_cookie)
        self._login_disabled = app.config.get('LOGIN_DISABLED', False)
        if add_context_processor:
            app.context_processor(_user_context_processor)
    def unauthorized(self):
        user_unauthorized.send(current_app._get_current_object())
        if self.unauthorized_callback:
            return self.unauthorized_callback()
        if request.blueprint in self.blueprint_login_views:
            login_view = self.blueprint_login_views[request.blueprint]
        else:
            login_view = self.login_view
        if not login_view:
            abort(401)
        if self.login_message:
            if self.localize_callback is not None:
                flash(self.localize_callback(self.login_message),
                      category=self.login_message_category)
            else:
                flash(self.login_message, category=self.login_message_category)
        return redirect(login_url(login_view, request.url))
    def user_loader(self, callback):
        self.user_callback = callback
        return callback
    def header_loader(self, callback):
        self.header_callback = callback
        return callback
    def request_loader(self, callback):
        self.request_callback = callback
        return callback
    def unauthorized_handler(self, callback):
        self.unauthorized_callback = callback
        return callback
    def needs_refresh_handler(self, callback):
        self.needs_refresh_callback = callback
        return callback
    def needs_refresh(self):
        user_needs_refresh.send(current_app._get_current_object())
        if self.needs_refresh_callback:
            return self.needs_refresh_callback()
        if not self.refresh_view:
            abort(401)
        if self.localize_callback is not None:
            flash(self.localize_callback(self.needs_refresh_message),
                  category=self.needs_refresh_message_category)
        else:
            flash(self.needs_refresh_message,
                  category=self.needs_refresh_message_category)
        return redirect(login_url(self.refresh_view, request.url))
    def reload_user(self, user=None):
        ctx = _request_ctx_stack.top
        if user is None:
            user_id = session.get('user_id')
            if user_id is None:
                ctx.user = self.anonymous_user()
            else:
                if self.user_callback is None:
                    raise Exception(
                        "No user_loader has been installed for this "
                        "LoginManager. Add one with the "
                        "'LoginManager.user_loader' decorator.")
                user = self.user_callback(user_id)
                if user is None:
                    ctx.user = self.anonymous_user()
                else:
                    ctx.user = user
        else:
            ctx.user = user
    def _load_user(self):
        user_accessed.send(current_app._get_current_object())
        config = current_app.config
        if config.get('SESSION_PROTECTION', self.session_protection):
            deleted = self._session_protection()
            if deleted:
                return self.reload_user()
        is_missing_user_id = 'user_id' not in session
        if is_missing_user_id:
            cookie_name = config.get('REMEMBER_COOKIE_NAME', COOKIE_NAME)
            header_name = config.get('AUTH_HEADER_NAME', AUTH_HEADER_NAME)
            has_cookie = (cookie_name in request.cookies and
                          session.get('remember') != 'clear')
            if has_cookie:
                return self._load_from_cookie(request.cookies[cookie_name])
            elif self.request_callback:
                return self._load_from_request(request)
            elif header_name in request.headers:
                return self._load_from_header(request.headers[header_name])
        return self.reload_user()
    def _session_protection(self):
        sess = session._get_current_object()
        ident = _create_identifier()
        app = current_app._get_current_object()
        mode = app.config.get('SESSION_PROTECTION', self.session_protection)
        if sess and ident != sess.get('_id', None):
            if mode == 'basic' or sess.permanent:
                sess['_fresh'] = False
                session_protected.send(app)
                return False
            elif mode == 'strong':
                for k in SESSION_KEYS:
                    sess.pop(k, None)
                sess['remember'] = 'clear'
                session_protected.send(app)
                return True
        return False
    def _load_from_cookie(self, cookie):
        user_id = decode_cookie(cookie)
        if user_id is not None:
            session['user_id'] = user_id
            session['_fresh'] = False
        self.reload_user()
        if _request_ctx_stack.top.user is not None:
            app = current_app._get_current_object()
            user_loaded_from_cookie.send(app, user=_get_user())
    def _load_from_header(self, header):
        user = None
        if self.header_callback:
            user = self.header_callback(header)
        if user is not None:
            self.reload_user(user=user)
            app = current_app._get_current_object()
            user_loaded_from_header.send(app, user=_get_user())
        else:
            self.reload_user()
    def _load_from_request(self, request):
        user = None
        if self.request_callback:
            user = self.request_callback(request)
        if user is not None:
            self.reload_user(user=user)
            app = current_app._get_current_object()
            user_loaded_from_request.send(app, user=_get_user())
        else:
            self.reload_user()
    def _update_remember_cookie(self, response):
        if 'remember' in session:
            operation = session.pop('remember', None)
            if operation == 'set' and 'user_id' in session:
                self._set_cookie(response)
            elif operation == 'clear':
                self._clear_cookie(response)
        return response
    def _set_cookie(self, response):
        config = current_app.config
        cookie_name = config.get('REMEMBER_COOKIE_NAME', COOKIE_NAME)
        duration = config.get('REMEMBER_COOKIE_DURATION', COOKIE_DURATION)
        domain = config.get('REMEMBER_COOKIE_DOMAIN')
        path = config.get('REMEMBER_COOKIE_PATH', '/')
        secure = config.get('REMEMBER_COOKIE_SECURE', COOKIE_SECURE)
        httponly = config.get('REMEMBER_COOKIE_HTTPONLY', COOKIE_HTTPONLY)
        data = encode_cookie(text_type(session['user_id']))
        try:
            expires = datetime.utcnow() + duration
        except TypeError:
            raise Exception('REMEMBER_COOKIE_DURATION must be a ' +
                            'datetime.timedelta, instead got: {0}'.format(
                            duration))
        response.set_cookie(cookie_name,
                            value=data,
                            expires=expires,
                            domain=domain,
                            path=path,
                            secure=secure,
                            httponly=httponly)
    def _clear_cookie(self, response):
        config = current_app.config
        cookie_name = config.get('REMEMBER_COOKIE_NAME', COOKIE_NAME)
        domain = config.get('REMEMBER_COOKIE_DOMAIN')
        path = config.get('REMEMBER_COOKIE_PATH', '/')
        response.delete_cookie(cookie_name, domain=domain, path=path)
