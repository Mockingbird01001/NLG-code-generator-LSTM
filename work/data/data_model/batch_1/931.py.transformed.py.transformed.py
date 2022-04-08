from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
    ConnectTimeoutError,
    InvalidHeader,
    MaxRetryError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    ResponseError,
)
from ..packages import six
log = logging.getLogger(__name__)
RequestHistory = namedtuple(
    "RequestHistory", ["method", "url", "error", "status", "redirect_location"]
)
_Default = object()
class _RetryMeta(type):
    @property
    def DEFAULT_METHOD_WHITELIST(cls):
        warnings.warn(
            "Using 'Retry.DEFAULT_METHOD_WHITELIST' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_METHODS_ALLOWED' instead",
            DeprecationWarning,
        )
        return cls.DEFAULT_ALLOWED_METHODS
    @DEFAULT_METHOD_WHITELIST.setter
    def DEFAULT_METHOD_WHITELIST(cls, value):
        warnings.warn(
            "Using 'Retry.DEFAULT_METHOD_WHITELIST' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_ALLOWED_METHODS' instead",
            DeprecationWarning,
        )
        cls.DEFAULT_ALLOWED_METHODS = value
    @property
    def DEFAULT_REDIRECT_HEADERS_BLACKLIST(cls):
        warnings.warn(
            "Using 'Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT' instead",
            DeprecationWarning,
        )
        return cls.DEFAULT_REMOVE_HEADERS_ON_REDIRECT
    @DEFAULT_REDIRECT_HEADERS_BLACKLIST.setter
    def DEFAULT_REDIRECT_HEADERS_BLACKLIST(cls, value):
        warnings.warn(
            "Using 'Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT' instead",
            DeprecationWarning,
        )
        cls.DEFAULT_REMOVE_HEADERS_ON_REDIRECT = value
@six.add_metaclass(_RetryMeta)
class Retry(object):
    DEFAULT_ALLOWED_METHODS = frozenset(
        ["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    RETRY_AFTER_STATUS_CODES = frozenset([413, 429, 503])
    DEFAULT_REMOVE_HEADERS_ON_REDIRECT = frozenset(["Authorization"])
    BACKOFF_MAX = 120
    def __init__(
        self,
        total=10,
        connect=None,
        read=None,
        redirect=None,
        status=None,
        other=None,
        allowed_methods=_Default,
        status_forcelist=None,
        backoff_factor=0,
        raise_on_redirect=True,
        raise_on_status=True,
        history=None,
        respect_retry_after_header=True,
        remove_headers_on_redirect=_Default,
        method_whitelist=_Default,
    ):
        if method_whitelist is not _Default:
            if allowed_methods is not _Default:
                raise ValueError(
                    "Using both 'allowed_methods' and "
                    "'method_whitelist' together is not allowed. "
                    "Instead only use 'allowed_methods'"
                )
            warnings.warn(
                "Using 'method_whitelist' with Retry is deprecated and "
                "will be removed in v2.0. Use 'allowed_methods' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            allowed_methods = method_whitelist
        if allowed_methods is _Default:
            allowed_methods = self.DEFAULT_ALLOWED_METHODS
        if remove_headers_on_redirect is _Default:
            remove_headers_on_redirect = self.DEFAULT_REMOVE_HEADERS_ON_REDIRECT
        self.total = total
        self.connect = connect
        self.read = read
        self.status = status
        self.other = other
        if redirect is False or total is False:
            redirect = 0
            raise_on_redirect = False
        self.redirect = redirect
        self.status_forcelist = status_forcelist or set()
        self.allowed_methods = allowed_methods
        self.backoff_factor = backoff_factor
        self.raise_on_redirect = raise_on_redirect
        self.raise_on_status = raise_on_status
        self.history = history or tuple()
        self.respect_retry_after_header = respect_retry_after_header
        self.remove_headers_on_redirect = frozenset(
            [h.lower() for h in remove_headers_on_redirect]
        )
    def new(self, **kw):
        params = dict(
            total=self.total,
            connect=self.connect,
            read=self.read,
            redirect=self.redirect,
            status=self.status,
            other=self.other,
            status_forcelist=self.status_forcelist,
            backoff_factor=self.backoff_factor,
            raise_on_redirect=self.raise_on_redirect,
            raise_on_status=self.raise_on_status,
            history=self.history,
            remove_headers_on_redirect=self.remove_headers_on_redirect,
            respect_retry_after_header=self.respect_retry_after_header,
        )
        if "method_whitelist" not in kw and "allowed_methods" not in kw:
            if "method_whitelist" in self.__dict__:
                warnings.warn(
                    "Using 'method_whitelist' with Retry is deprecated and "
                    "will be removed in v2.0. Use 'allowed_methods' instead",
                    DeprecationWarning,
                )
                params["method_whitelist"] = self.allowed_methods
            else:
                params["allowed_methods"] = self.allowed_methods
        params.update(kw)
        return type(self)(**params)
    @classmethod
    def from_int(cls, retries, redirect=True, default=None):
        if retries is None:
            retries = default if default is not None else cls.DEFAULT
        if isinstance(retries, Retry):
            return retries
        redirect = bool(redirect) and None
        new_retries = cls(retries, redirect=redirect)
        log.debug("Converted retries value: %r -> %r", retries, new_retries)
        return new_retries
    def get_backoff_time(self):
        consecutive_errors_len = len(
            list(
                takewhile(lambda x: x.redirect_location is None, reversed(self.history))
            )
        )
        if consecutive_errors_len <= 1:
            return 0
        backoff_value = self.backoff_factor * (2 ** (consecutive_errors_len - 1))
        return min(self.BACKOFF_MAX, backoff_value)
    def parse_retry_after(self, retry_after):
        if re.match(r"^\s*[0-9]+\s*$", retry_after):
            seconds = int(retry_after)
        else:
            retry_date_tuple = email.utils.parsedate_tz(retry_after)
            if retry_date_tuple is None:
                raise InvalidHeader("Invalid Retry-After header: %s" % retry_after)
            if retry_date_tuple[9] is None:
                retry_date_tuple = retry_date_tuple[:9] + (0,) + retry_date_tuple[10:]
            retry_date = email.utils.mktime_tz(retry_date_tuple)
            seconds = retry_date - time.time()
        if seconds < 0:
            seconds = 0
        return seconds
    def get_retry_after(self, response):
        retry_after = response.getheader("Retry-After")
        if retry_after is None:
            return None
        return self.parse_retry_after(retry_after)
    def sleep_for_retry(self, response=None):
        retry_after = self.get_retry_after(response)
        if retry_after:
            time.sleep(retry_after)
            return True
        return False
    def _sleep_backoff(self):
        backoff = self.get_backoff_time()
        if backoff <= 0:
            return
        time.sleep(backoff)
    def sleep(self, response=None):
        if self.respect_retry_after_header and response:
            slept = self.sleep_for_retry(response)
            if slept:
                return
        self._sleep_backoff()
    def _is_connection_error(self, err):
        if isinstance(err, ProxyError):
            err = err.original_error
        return isinstance(err, ConnectTimeoutError)
    def _is_read_error(self, err):
        return isinstance(err, (ReadTimeoutError, ProtocolError))
    def _is_method_retryable(self, method):
        if "method_whitelist" in self.__dict__:
            warnings.warn(
                "Using 'method_whitelist' with Retry is deprecated and "
                "will be removed in v2.0. Use 'allowed_methods' instead",
                DeprecationWarning,
            )
            allowed_methods = self.method_whitelist
        else:
            allowed_methods = self.allowed_methods
        if allowed_methods and method.upper() not in allowed_methods:
            return False
        return True
    def is_retry(self, method, status_code, has_retry_after=False):
        if not self._is_method_retryable(method):
            return False
        if self.status_forcelist and status_code in self.status_forcelist:
            return True
        return (
            self.total
            and self.respect_retry_after_header
            and has_retry_after
            and (status_code in self.RETRY_AFTER_STATUS_CODES)
        )
    def is_exhausted(self):
        retry_counts = (
            self.total,
            self.connect,
            self.read,
            self.redirect,
            self.status,
            self.other,
        )
        retry_counts = list(filter(None, retry_counts))
        if not retry_counts:
            return False
        return min(retry_counts) < 0
    def increment(
        self,
        method=None,
        url=None,
        response=None,
        error=None,
        _pool=None,
        _stacktrace=None,
    ):
        if self.total is False and error:
            raise six.reraise(type(error), error, _stacktrace)
        total = self.total
        if total is not None:
            total -= 1
        connect = self.connect
        read = self.read
        redirect = self.redirect
        status_count = self.status
        other = self.other
        cause = "unknown"
        status = None
        redirect_location = None
        if error and self._is_connection_error(error):
            if connect is False:
                raise six.reraise(type(error), error, _stacktrace)
            elif connect is not None:
                connect -= 1
        elif error and self._is_read_error(error):
            if read is False or not self._is_method_retryable(method):
                raise six.reraise(type(error), error, _stacktrace)
            elif read is not None:
                read -= 1
        elif error:
            if other is not None:
                other -= 1
        elif response and response.get_redirect_location():
            if redirect is not None:
                redirect -= 1
            cause = "too many redirects"
            redirect_location = response.get_redirect_location()
            status = response.status
        else:
            cause = ResponseError.GENERIC_ERROR
            if response and response.status:
                if status_count is not None:
                    status_count -= 1
                cause = ResponseError.SPECIFIC_ERROR.format(status_code=response.status)
                status = response.status
        history = self.history + (
            RequestHistory(method, url, error, status, redirect_location),
        )
        new_retry = self.new(
            total=total,
            connect=connect,
            read=read,
            redirect=redirect,
            status=status_count,
            other=other,
            history=history,
        )
        if new_retry.is_exhausted():
            raise MaxRetryError(_pool, url, error or ResponseError(cause))
        log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)
        return new_retry
    def __repr__(self):
        return (
            "{cls.__name__}(total={self.total}, connect={self.connect}, "
            "read={self.read}, redirect={self.redirect}, status={self.status})"
        ).format(cls=type(self), self=self)
    def __getattr__(self, item):
        if item == "method_whitelist":
            warnings.warn(
                "Using 'method_whitelist' with Retry is deprecated and "
                "will be removed in v2.0. Use 'allowed_methods' instead",
                DeprecationWarning,
            )
            return self.allowed_methods
        try:
            return getattr(super(Retry, self), item)
        except AttributeError:
            return getattr(Retry, item)
Retry.DEFAULT = Retry(3)
