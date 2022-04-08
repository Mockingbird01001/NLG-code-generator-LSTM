
r"""
    werkzeug.contrib.securecookie
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This module implements a cookie that is not alterable from the client
    because it adds a checksum the server checks for.  You can use it as
    session replacement if all you have is a user id or something to mark
    a logged in user.
    Keep in mind that the data is still readable from the client as a
    normal cookie is.  However you don't have to store and flush the
    sessions you have at the server.
    Example usage:
    >>> from werkzeug.contrib.securecookie import SecureCookie
    >>> x = SecureCookie({"foo": 42, "baz": (1, 2, 3)}, "deadbeef")
    Dumping into a string so that one can store it in a cookie:
    >>> value = x.serialize()
    Loading from that string again:
    >>> x = SecureCookie.unserialize(value, "deadbeef")
    >>> x["baz"]
    (1, 2, 3)
    If someone modifies the cookie and the checksum is wrong the unserialize
    method will fail silently and return a new empty `SecureCookie` object.
    Keep in mind that the values will be visible in the cookie so do not
    store data in a cookie you don't want the user to see.
    Application Integration
    =======================
    If you are using the werkzeug request objects you could integrate the
    secure cookie into your application like this::
        from werkzeug.utils import cached_property
        from werkzeug.wrappers import BaseRequest
        from werkzeug.contrib.securecookie import SecureCookie
        SECRET_KEY = '\xfa\xdd\xb8z\xae\xe0}4\x8b\xea'
        class Request(BaseRequest):
            @cached_property
            def client_session(self):
                data = self.cookies.get('session_data')
                if not data:
                    return SecureCookie(secret_key=SECRET_KEY)
                return SecureCookie.unserialize(data, SECRET_KEY)
        def application(environ, start_response):
            request = Request(environ)
            response = ...
            if request.client_session.should_save:
                session_data = request.client_session.serialize()
                response.set_cookie('session_data', session_data,
                                    httponly=True)
            return response(environ, start_response)
    A less verbose integration can be achieved by using shorthand methods::
        class Request(BaseRequest):
            @cached_property
            def client_session(self):
                return SecureCookie.load_cookie(self, secret_key=COOKIE_SECRET)
        def application(environ, start_response):
            request = Request(environ)
            response = ...
            request.client_session.save_cookie(response)
            return response(environ, start_response)
    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import pickle
import base64
from hmac import new as hmac
from time import time
from hashlib import sha1 as _default_hash
from werkzeug._compat import iteritems, text_type
from werkzeug.urls import url_quote_plus, url_unquote_plus
from werkzeug._internal import _date_to_unix
from werkzeug.contrib.sessions import ModificationTrackingDict
from werkzeug.security import safe_str_cmp
from werkzeug._compat import to_native
class UnquoteError(Exception):
class SecureCookie(ModificationTrackingDict):
    hash_method = staticmethod(_default_hash)
    serialization_method = pickle
    quote_base64 = True
    def __init__(self, data=None, secret_key=None, new=True):
        ModificationTrackingDict.__init__(self, data or ())
        if secret_key is not None:
            secret_key = bytes(secret_key)
        self.secret_key = secret_key
        self.new = new
    def __repr__(self):
        return '<%s %s%s>' % (
            self.__class__.__name__,
            dict.__repr__(self),
            self.should_save and '*' or ''
        )
    @property
    def should_save(self):
        return self.modified
    @classmethod
    def quote(cls, value):
        if cls.serialization_method is not None:
            value = cls.serialization_method.dumps(value)
        if cls.quote_base64:
            value = b''.join(base64.b64encode(value).splitlines()).strip()
        return value
    @classmethod
    def unquote(cls, value):
        try:
            if cls.quote_base64:
                value = base64.b64decode(value)
            if cls.serialization_method is not None:
                value = cls.serialization_method.loads(value)
            return value
        except Exception:
            raise UnquoteError()
    def serialize(self, expires=None):
        if self.secret_key is None:
            raise RuntimeError('no secret key defined')
        if expires:
            self['_expires'] = _date_to_unix(expires)
        result = []
        mac = hmac(self.secret_key, None, self.hash_method)
        for key, value in sorted(self.items()):
            result.append(('%s=%s' % (
                url_quote_plus(key),
                self.quote(value).decode('ascii')
            )).encode('ascii'))
            mac.update(b'|' + result[-1])
        return b'?'.join([
            base64.b64encode(mac.digest()).strip(),
            b'&'.join(result)
        ])
    @classmethod
    def unserialize(cls, string, secret_key):
        if isinstance(string, text_type):
            string = string.encode('utf-8', 'replace')
        if isinstance(secret_key, text_type):
            secret_key = secret_key.encode('utf-8', 'replace')
        try:
            base64_hash, data = string.split(b'?', 1)
        except (ValueError, IndexError):
            items = ()
        else:
            items = {}
            mac = hmac(secret_key, None, cls.hash_method)
            for item in data.split(b'&'):
                mac.update(b'|' + item)
                if b'=' not in item:
                    items = None
                    break
                key, value = item.split(b'=', 1)
                key = url_unquote_plus(key.decode('ascii'))
                try:
                    key = to_native(key)
                except UnicodeError:
                    pass
                items[key] = value
            try:
                client_hash = base64.b64decode(base64_hash)
            except TypeError:
                items = client_hash = None
            if items is not None and safe_str_cmp(client_hash, mac.digest()):
                try:
                    for key, value in iteritems(items):
                        items[key] = cls.unquote(value)
                except UnquoteError:
                    items = ()
                else:
                    if '_expires' in items:
                        if time() > items['_expires']:
                            items = ()
                        else:
                            del items['_expires']
            else:
                items = ()
        return cls(items, secret_key, False)
    @classmethod
    def load_cookie(cls, request, key='session', secret_key=None):
        data = request.cookies.get(key)
        if not data:
            return cls(secret_key=secret_key)
        return cls.unserialize(data, secret_key)
    def save_cookie(self, response, key='session', expires=None,
                    session_expires=None, max_age=None, path='/', domain=None,
                    secure=None, httponly=False, force=False):
        if force or self.should_save:
            data = self.serialize(session_expires or expires)
            response.set_cookie(key, data, expires=expires, max_age=max_age,
                                path=path, domain=domain, secure=secure,
                                httponly=httponly)
