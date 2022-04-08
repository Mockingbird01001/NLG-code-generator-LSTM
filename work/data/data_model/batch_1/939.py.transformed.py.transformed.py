
"""
    werkzeug.routing
    ~~~~~~~~~~~~~~~~
    When it comes to combining multiple controller or view functions (however
    you want to call them) you need a dispatcher.  A simple way would be
    applying regular expression tests on the ``PATH_INFO`` and calling
    registered callback functions that return the value then.
    This module implements a much more powerful system than simple regular
    expression matching because it can also convert values in the URLs and
    build URLs.
    Here a simple example that creates an URL map for an application with
    two subdomains (www and kb) and some URL rules:
    >>> m = Map([
    ...     Rule('/', endpoint='static/index'),
    ...     Rule('/about', endpoint='static/about'),
    ...     Rule('/help', endpoint='static/help'),
    ...     Subdomain('kb', [
    ...         Rule('/', endpoint='kb/index'),
    ...         Rule('/browse/', endpoint='kb/browse'),
    ...         Rule('/browse/<int:id>/', endpoint='kb/browse'),
    ...         Rule('/browse/<int:id>/<int:page>', endpoint='kb/browse')
    ...     ])
    ... ], default_subdomain='www')
    If the application doesn't use subdomains it's perfectly fine to not set
    the default subdomain and not use the `Subdomain` rule factory.  The endpoint
    in the rules can be anything, for example import paths or unique
    identifiers.  The WSGI application can use those endpoints to get the
    handler for that URL.  It doesn't have to be a string at all but it's
    recommended.
    Now it's possible to create a URL adapter for one of the subdomains and
    build URLs:
    >>> c = m.bind('example.com')
    >>> c.build("kb/browse", dict(id=42))
    'http://kb.example.com/browse/42/'
    >>> c.build("kb/browse", dict())
    'http://kb.example.com/browse/'
    >>> c.build("kb/browse", dict(id=42, page=3))
    'http://kb.example.com/browse/42/3'
    >>> c.build("static/about")
    '/about'
    >>> c.build("static/index", force_external=True)
    'http://www.example.com/'
    >>> c = m.bind('example.com', subdomain='kb')
    >>> c.build("static/about")
    'http://www.example.com/about'
    The first argument to bind is the server name *without* the subdomain.
    Per default it will assume that the script is mounted on the root, but
    often that's not the case so you can provide the real mount point as
    second argument:
    >>> c = m.bind('example.com', '/applications/example')
    The third argument can be the subdomain, if not given the default
    subdomain is used.  For more details about binding have a look at the
    documentation of the `MapAdapter`.
    And here is how you can match URLs:
    >>> c = m.bind('example.com')
    >>> c.match("/")
    ('static/index', {})
    >>> c.match("/about")
    ('static/about', {})
    >>> c = m.bind('example.com', '/', 'kb')
    >>> c.match("/")
    ('kb/index', {})
    >>> c.match("/browse/42/23")
    ('kb/browse', {'id': 42, 'page': 23})
    If matching fails you get a `NotFound` exception, if the rule thinks
    it's a good idea to redirect (for example because the URL was defined
    to have a slash at the end but the request was missing that slash) it
    will raise a `RequestRedirect` exception.  Both are subclasses of the
    `HTTPException` so you can use those errors as responses in the
    application.
    If matching succeeded but the URL rule was incompatible to the given
    method (for example there were only rules for `GET` and `HEAD` and
    routing system tried to match a `POST` request) a `MethodNotAllowed`
    exception is raised.
    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import difflib
import re
import uuid
import posixpath
from pprint import pformat
from threading import Lock
from werkzeug.urls import url_encode, url_quote, url_join
from werkzeug.utils import redirect, format_string
from werkzeug.exceptions import HTTPException, NotFound, MethodNotAllowed,     BadHost
from werkzeug._internal import _get_environ, _encode_idna
from werkzeug._compat import itervalues, iteritems, to_unicode, to_bytes,    text_type, string_types, native_string_result,    implements_to_string, wsgi_decoding_dance
from werkzeug.datastructures import ImmutableDict, MultiDict
from werkzeug.utils import cached_property
_rule_re = re.compile(r'''
    <
    (?:
    )?
    >
''', re.VERBOSE)
_simple_rule_re = re.compile(r'<([^>]+)>')
_converter_args_re = re.compile(r'''
    ((?P<name>\w+)\s*=\s*)?
    (?P<value>
        True|False|
        \d+.\d+|
        \d+.|
        \d+|
        \w+|
        [urUR]?(?P<stringval>"[^"]*?"|'[^']*')
    )\s*,
''', re.VERBOSE | re.UNICODE)
_PYTHON_CONSTANTS = {
    'None':     None,
    'True':     True,
    'False':    False
}
def _pythonize(value):
    if value in _PYTHON_CONSTANTS:
        return _PYTHON_CONSTANTS[value]
    for convert in int, float:
        try:
            return convert(value)
        except ValueError:
            pass
    if value[:1] == value[-1:] and value[0] in '"\'':
        value = value[1:-1]
    return text_type(value)
def parse_converter_args(argstr):
    argstr += ','
    args = []
    kwargs = {}
    for item in _converter_args_re.finditer(argstr):
        value = item.group('stringval')
        if value is None:
            value = item.group('value')
        value = _pythonize(value)
        if not item.group('name'):
            args.append(value)
        else:
            name = item.group('name')
            kwargs[name] = value
    return tuple(args), kwargs
def parse_rule(rule):
    pos = 0
    end = len(rule)
    do_match = _rule_re.match
    used_names = set()
    while pos < end:
        m = do_match(rule, pos)
        if m is None:
            break
        data = m.groupdict()
        if data['static']:
            yield None, None, data['static']
        variable = data['variable']
        converter = data['converter'] or 'default'
        if variable in used_names:
            raise ValueError('variable name %r used twice.' % variable)
        used_names.add(variable)
        yield converter, data['args'] or None, variable
        pos = m.end()
    if pos < end:
        remaining = rule[pos:]
        if '>' in remaining or '<' in remaining:
            raise ValueError('malformed url rule: %r' % rule)
        yield None, None, remaining
class RoutingException(Exception):
class RequestRedirect(HTTPException, RoutingException):
    code = 301
    def __init__(self, new_url):
        RoutingException.__init__(self, new_url)
        self.new_url = new_url
    def get_response(self, environ):
        return redirect(self.new_url, self.code)
class RequestSlash(RoutingException):
class RequestAliasRedirect(RoutingException):
    def __init__(self, matched_values):
        self.matched_values = matched_values
@implements_to_string
class BuildError(RoutingException, LookupError):
    def __init__(self, endpoint, values, method, adapter=None):
        LookupError.__init__(self, endpoint, values, method)
        self.endpoint = endpoint
        self.values = values
        self.method = method
        self.adapter = adapter
    @cached_property
    def suggested(self):
        return self.closest_rule(self.adapter)
    def closest_rule(self, adapter):
        def _score_rule(rule):
            return sum([
                0.98 * difflib.SequenceMatcher(
                    None, rule.endpoint, self.endpoint
                ).ratio(),
                0.01 * bool(set(self.values or ()).issubset(rule.arguments)),
                0.01 * bool(rule.methods and self.method in rule.methods)
            ])
        if adapter and adapter.map._rules:
            return max(adapter.map._rules, key=_score_rule)
    def __str__(self):
        message = []
        message.append('Could not build url for endpoint %r' % self.endpoint)
        if self.method:
            message.append(' (%r)' % self.method)
        if self.values:
            message.append(' with values %r' % sorted(self.values.keys()))
        message.append('.')
        if self.suggested:
            if self.endpoint == self.suggested.endpoint:
                if self.method and self.method not in self.suggested.methods:
                    message.append(' Did you mean to use methods %r?' % sorted(
                        self.suggested.methods
                    ))
                missing_values = self.suggested.arguments.union(
                    set(self.suggested.defaults or ())
                ) - set(self.values.keys())
                if missing_values:
                    message.append(
                        ' Did you forget to specify values %r?' %
                        sorted(missing_values)
                    )
            else:
                message.append(
                    ' Did you mean %r instead?' % self.suggested.endpoint
                )
        return u''.join(message)
class ValidationError(ValueError):
class RuleFactory(object):
    def get_rules(self, map):
        raise NotImplementedError()
class Subdomain(RuleFactory):
    def __init__(self, subdomain, rules):
        self.subdomain = subdomain
        self.rules = rules
    def get_rules(self, map):
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.subdomain = self.subdomain
                yield rule
class Submount(RuleFactory):
    def __init__(self, path, rules):
        self.path = path.rstrip('/')
        self.rules = rules
    def get_rules(self, map):
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.rule = self.path + rule.rule
                yield rule
class EndpointPrefix(RuleFactory):
    def __init__(self, prefix, rules):
        self.prefix = prefix
        self.rules = rules
    def get_rules(self, map):
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.endpoint = self.prefix + rule.endpoint
                yield rule
class RuleTemplate(object):
    def __init__(self, rules):
        self.rules = list(rules)
    def __call__(self, *args, **kwargs):
        return RuleTemplateFactory(self.rules, dict(*args, **kwargs))
class RuleTemplateFactory(RuleFactory):
    def __init__(self, rules, context):
        self.rules = rules
        self.context = context
    def get_rules(self, map):
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                new_defaults = subdomain = None
                if rule.defaults:
                    new_defaults = {}
                    for key, value in iteritems(rule.defaults):
                        if isinstance(value, string_types):
                            value = format_string(value, self.context)
                        new_defaults[key] = value
                if rule.subdomain is not None:
                    subdomain = format_string(rule.subdomain, self.context)
                new_endpoint = rule.endpoint
                if isinstance(new_endpoint, string_types):
                    new_endpoint = format_string(new_endpoint, self.context)
                yield Rule(
                    format_string(rule.rule, self.context),
                    new_defaults,
                    subdomain,
                    rule.methods,
                    rule.build_only,
                    new_endpoint,
                    rule.strict_slashes
                )
@implements_to_string
class Rule(RuleFactory):
    def __init__(self, string, defaults=None, subdomain=None, methods=None,
                 build_only=False, endpoint=None, strict_slashes=None,
                 redirect_to=None, alias=False, host=None):
        if not string.startswith('/'):
            raise ValueError('urls must start with a leading slash')
        self.rule = string
        self.is_leaf = not string.endswith('/')
        self.map = None
        self.strict_slashes = strict_slashes
        self.subdomain = subdomain
        self.host = host
        self.defaults = defaults
        self.build_only = build_only
        self.alias = alias
        if methods is None:
            self.methods = None
        else:
            if isinstance(methods, str):
                raise TypeError('param `methods` should be `Iterable[str]`, not `str`')
            self.methods = set([x.upper() for x in methods])
            if 'HEAD' not in self.methods and 'GET' in self.methods:
                self.methods.add('HEAD')
        self.endpoint = endpoint
        self.redirect_to = redirect_to
        if defaults:
            self.arguments = set(map(str, defaults))
        else:
            self.arguments = set()
        self._trace = self._converters = self._regex = self._weights = None
    def empty(self):
        return type(self)(self.rule, **self.get_empty_kwargs())
    def get_empty_kwargs(self):
        defaults = None
        if self.defaults:
            defaults = dict(self.defaults)
        return dict(defaults=defaults, subdomain=self.subdomain,
                    methods=self.methods, build_only=self.build_only,
                    endpoint=self.endpoint, strict_slashes=self.strict_slashes,
                    redirect_to=self.redirect_to, alias=self.alias,
                    host=self.host)
    def get_rules(self, map):
        yield self
    def refresh(self):
        self.bind(self.map, rebind=True)
    def bind(self, map, rebind=False):
        if self.map is not None and not rebind:
            raise RuntimeError('url rule %r already bound to map %r' %
                               (self, self.map))
        self.map = map
        if self.strict_slashes is None:
            self.strict_slashes = map.strict_slashes
        if self.subdomain is None:
            self.subdomain = map.default_subdomain
        self.compile()
    def get_converter(self, variable_name, converter_name, args, kwargs):
        if converter_name not in self.map.converters:
            raise LookupError('the converter %r does not exist' % converter_name)
        return self.map.converters[converter_name](self.map, *args, **kwargs)
    def compile(self):
        assert self.map is not None, 'rule not bound'
        if self.map.host_matching:
            domain_rule = self.host or ''
        else:
            domain_rule = self.subdomain or ''
        self._trace = []
        self._converters = {}
        self._weights = []
        regex_parts = []
        def _build_regex(rule):
            for converter, arguments, variable in parse_rule(rule):
                if converter is None:
                    regex_parts.append(re.escape(variable))
                    self._trace.append((False, variable))
                    for part in variable.split('/'):
                        if part:
                            self._weights.append((0, -len(part)))
                else:
                    if arguments:
                        c_args, c_kwargs = parse_converter_args(arguments)
                    else:
                        c_args = ()
                        c_kwargs = {}
                    convobj = self.get_converter(
                        variable, converter, c_args, c_kwargs)
                    regex_parts.append('(?P<%s>%s)' % (variable, convobj.regex))
                    self._converters[variable] = convobj
                    self._trace.append((True, variable))
                    self._weights.append((1, convobj.weight))
                    self.arguments.add(str(variable))
        _build_regex(domain_rule)
        regex_parts.append('\\|')
        self._trace.append((False, '|'))
        _build_regex(self.is_leaf and self.rule or self.rule.rstrip('/'))
        if not self.is_leaf:
            self._trace.append((False, '/'))
        if self.build_only:
            return
        regex = r'^%s%s$' % (
            u''.join(regex_parts),
            (not self.is_leaf or not self.strict_slashes) and
            '(?<!/)(?P<__suffix__>/?)' or ''
        )
        self._regex = re.compile(regex, re.UNICODE)
    def match(self, path, method=None):
        if not self.build_only:
            m = self._regex.search(path)
            if m is not None:
                groups = m.groupdict()
                if self.strict_slashes and not self.is_leaf and                        not groups.pop('__suffix__') and                        (method is None or self.methods is None or
                         method in self.methods):
                    raise RequestSlash()
                elif not self.strict_slashes:
                    del groups['__suffix__']
                result = {}
                for name, value in iteritems(groups):
                    try:
                        value = self._converters[name].to_python(value)
                    except ValidationError:
                        return
                    result[str(name)] = value
                if self.defaults:
                    result.update(self.defaults)
                if self.alias and self.map.redirect_defaults:
                    raise RequestAliasRedirect(result)
                return result
    def build(self, values, append_unknown=True):
        tmp = []
        add = tmp.append
        processed = set(self.arguments)
        for is_dynamic, data in self._trace:
            if is_dynamic:
                try:
                    add(self._converters[data].to_url(values[data]))
                except ValidationError:
                    return
                processed.add(data)
            else:
                add(url_quote(to_bytes(data, self.map.charset), safe='/:|+'))
        domain_part, url = (u''.join(tmp)).split(u'|', 1)
        if append_unknown:
            query_vars = MultiDict(values)
            for key in processed:
                if key in query_vars:
                    del query_vars[key]
            if query_vars:
                url += u'?' + url_encode(query_vars, charset=self.map.charset,
                                         sort=self.map.sort_parameters,
                                         key=self.map.sort_key)
        return domain_part, url
    def provides_defaults_for(self, rule):
        return not self.build_only and self.defaults and            self.endpoint == rule.endpoint and self != rule and            self.arguments == rule.arguments
    def suitable_for(self, values, method=None):
        if method is not None and self.methods is not None           and method not in self.methods:
            return False
        defaults = self.defaults or ()
        for key in self.arguments:
            if key not in defaults and key not in values:
                return False
        if defaults:
            for key, value in iteritems(defaults):
                if key in values and value != values[key]:
                    return False
        return True
    def match_compare_key(self):
        return bool(self.arguments), -len(self._weights), self._weights
    def build_compare_key(self):
        return self.alias and 1 or 0, -len(self.arguments),            -len(self.defaults or ())
    def __eq__(self, other):
        return self.__class__ is other.__class__ and            self._trace == other._trace
    __hash__ = None
    def __ne__(self, other):
        return not self.__eq__(other)
    def __str__(self):
        return self.rule
    @native_string_result
    def __repr__(self):
        if self.map is None:
            return u'<%s (unbound)>' % self.__class__.__name__
        tmp = []
        for is_dynamic, data in self._trace:
            if is_dynamic:
                tmp.append(u'<%s>' % data)
            else:
                tmp.append(data)
        return u'<%s %s%s -> %s>' % (
            self.__class__.__name__,
            repr((u''.join(tmp)).lstrip(u'|')).lstrip(u'u'),
            self.methods is not None
            and u' (%s)' % u', '.join(self.methods)
            or u'',
            self.endpoint
        )
class BaseConverter(object):
    regex = '[^/]+'
    weight = 100
    def __init__(self, map):
        self.map = map
    def to_python(self, value):
        return value
    def to_url(self, value):
        return url_quote(value, charset=self.map.charset)
class UnicodeConverter(BaseConverter):
    def __init__(self, map, minlength=1, maxlength=None, length=None):
        BaseConverter.__init__(self, map)
        if length is not None:
            length = '{%d}' % int(length)
        else:
            if maxlength is None:
                maxlength = ''
            else:
                maxlength = int(maxlength)
            length = '{%s,%s}' % (
                int(minlength),
                maxlength
            )
        self.regex = '[^/]' + length
class AnyConverter(BaseConverter):
    def __init__(self, map, *items):
        BaseConverter.__init__(self, map)
        self.regex = '(?:%s)' % '|'.join([re.escape(x) for x in items])
class PathConverter(BaseConverter):
    regex = '[^/].*?'
    weight = 200
class NumberConverter(BaseConverter):
    weight = 50
    def __init__(self, map, fixed_digits=0, min=None, max=None):
        BaseConverter.__init__(self, map)
        self.fixed_digits = fixed_digits
        self.min = min
        self.max = max
    def to_python(self, value):
        if (self.fixed_digits and len(value) != self.fixed_digits):
            raise ValidationError()
        value = self.num_convert(value)
        if (self.min is not None and value < self.min) or           (self.max is not None and value > self.max):
            raise ValidationError()
        return value
    def to_url(self, value):
        value = self.num_convert(value)
        if self.fixed_digits:
            value = ('%%0%sd' % self.fixed_digits) % value
        return str(value)
class IntegerConverter(NumberConverter):
    regex = r'\d+'
    num_convert = int
class FloatConverter(NumberConverter):
    regex = r'\d+\.\d+'
    num_convert = float
    def __init__(self, map, min=None, max=None):
        NumberConverter.__init__(self, map, 0, min, max)
class UUIDConverter(BaseConverter):
    regex = r'[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-'            r'[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}'
    def to_python(self, value):
        return uuid.UUID(value)
    def to_url(self, value):
        return str(value)
DEFAULT_CONVERTERS = {
    'default':          UnicodeConverter,
    'string':           UnicodeConverter,
    'any':              AnyConverter,
    'path':             PathConverter,
    'int':              IntegerConverter,
    'float':            FloatConverter,
    'uuid':             UUIDConverter,
}
class Map(object):
    default_converters = ImmutableDict(DEFAULT_CONVERTERS)
    def __init__(self, rules=None, default_subdomain='', charset='utf-8',
                 strict_slashes=True, redirect_defaults=True,
                 converters=None, sort_parameters=False, sort_key=None,
                 encoding_errors='replace', host_matching=False):
        self._rules = []
        self._rules_by_endpoint = {}
        self._remap = True
        self._remap_lock = Lock()
        self.default_subdomain = default_subdomain
        self.charset = charset
        self.encoding_errors = encoding_errors
        self.strict_slashes = strict_slashes
        self.redirect_defaults = redirect_defaults
        self.host_matching = host_matching
        self.converters = self.default_converters.copy()
        if converters:
            self.converters.update(converters)
        self.sort_parameters = sort_parameters
        self.sort_key = sort_key
        for rulefactory in rules or ():
            self.add(rulefactory)
    def is_endpoint_expecting(self, endpoint, *arguments):
        self.update()
        arguments = set(arguments)
        for rule in self._rules_by_endpoint[endpoint]:
            if arguments.issubset(rule.arguments):
                return True
        return False
    def iter_rules(self, endpoint=None):
        self.update()
        if endpoint is not None:
            return iter(self._rules_by_endpoint[endpoint])
        return iter(self._rules)
    def add(self, rulefactory):
        for rule in rulefactory.get_rules(self):
            rule.bind(self)
            self._rules.append(rule)
            self._rules_by_endpoint.setdefault(rule.endpoint, []).append(rule)
        self._remap = True
    def bind(self, server_name, script_name=None, subdomain=None,
             url_scheme='http', default_method='GET', path_info=None,
             query_args=None):
        server_name = server_name.lower()
        if self.host_matching:
            if subdomain is not None:
                raise RuntimeError('host matching enabled and a '
                                   'subdomain was provided')
        elif subdomain is None:
            subdomain = self.default_subdomain
        if script_name is None:
            script_name = '/'
        try:
            server_name = _encode_idna(server_name)
        except UnicodeError:
            raise BadHost()
        return MapAdapter(self, server_name, script_name, subdomain,
                          url_scheme, path_info, default_method, query_args)
    def bind_to_environ(self, environ, server_name=None, subdomain=None):
        environ = _get_environ(environ)
        if 'HTTP_HOST' in environ:
            wsgi_server_name = environ['HTTP_HOST']
            if environ['wsgi.url_scheme'] == 'http'                    and wsgi_server_name.endswith(':80'):
                wsgi_server_name = wsgi_server_name[:-3]
            elif environ['wsgi.url_scheme'] == 'https'                    and wsgi_server_name.endswith(':443'):
                wsgi_server_name = wsgi_server_name[:-4]
        else:
            wsgi_server_name = environ['SERVER_NAME']
            if (environ['wsgi.url_scheme'], environ['SERVER_PORT']) not               in (('https', '443'), ('http', '80')):
                wsgi_server_name += ':' + environ['SERVER_PORT']
        wsgi_server_name = wsgi_server_name.lower()
        if server_name is None:
            server_name = wsgi_server_name
        else:
            server_name = server_name.lower()
        if subdomain is None and not self.host_matching:
            cur_server_name = wsgi_server_name.split('.')
            real_server_name = server_name.split('.')
            offset = -len(real_server_name)
            if cur_server_name[offset:] != real_server_name:
                subdomain = '<invalid>'
            else:
                subdomain = '.'.join(filter(None, cur_server_name[:offset]))
        def _get_wsgi_string(name):
            val = environ.get(name)
            if val is not None:
                return wsgi_decoding_dance(val, self.charset)
        script_name = _get_wsgi_string('SCRIPT_NAME')
        path_info = _get_wsgi_string('PATH_INFO')
        query_args = _get_wsgi_string('QUERY_STRING')
        return Map.bind(self, server_name, script_name,
                        subdomain, environ['wsgi.url_scheme'],
                        environ['REQUEST_METHOD'], path_info,
                        query_args=query_args)
    def update(self):
        if not self._remap:
            return
        with self._remap_lock:
            if not self._remap:
                return
            self._rules.sort(key=lambda x: x.match_compare_key())
            for rules in itervalues(self._rules_by_endpoint):
                rules.sort(key=lambda x: x.build_compare_key())
            self._remap = False
    def __repr__(self):
        rules = self.iter_rules()
        return '%s(%s)' % (self.__class__.__name__, pformat(list(rules)))
class MapAdapter(object):
    def __init__(self, map, server_name, script_name, subdomain,
                 url_scheme, path_info, default_method, query_args=None):
        self.map = map
        self.server_name = to_unicode(server_name)
        script_name = to_unicode(script_name)
        if not script_name.endswith(u'/'):
            script_name += u'/'
        self.script_name = script_name
        self.subdomain = to_unicode(subdomain)
        self.url_scheme = to_unicode(url_scheme)
        self.path_info = to_unicode(path_info)
        self.default_method = to_unicode(default_method)
        self.query_args = query_args
    def dispatch(self, view_func, path_info=None, method=None,
                 catch_http_exceptions=False):
        try:
            try:
                endpoint, args = self.match(path_info, method)
            except RequestRedirect as e:
                return e
            return view_func(endpoint, args)
        except HTTPException as e:
            if catch_http_exceptions:
                return e
            raise
    def match(self, path_info=None, method=None, return_rule=False,
              query_args=None):
        self.map.update()
        if path_info is None:
            path_info = self.path_info
        else:
            path_info = to_unicode(path_info, self.map.charset)
        if query_args is None:
            query_args = self.query_args
        method = (method or self.default_method).upper()
        path = u'%s|%s' % (
            self.map.host_matching and self.server_name or self.subdomain,
            path_info and '/%s' % path_info.lstrip('/')
        )
        have_match_for = set()
        for rule in self.map._rules:
            try:
                rv = rule.match(path, method)
            except RequestSlash:
                raise RequestRedirect(self.make_redirect_url(
                    url_quote(path_info, self.map.charset,
                              safe='/:|+') + '/', query_args))
            except RequestAliasRedirect as e:
                raise RequestRedirect(self.make_alias_redirect_url(
                    path, rule.endpoint, e.matched_values, method, query_args))
            if rv is None:
                continue
            if rule.methods is not None and method not in rule.methods:
                have_match_for.update(rule.methods)
                continue
            if self.map.redirect_defaults:
                redirect_url = self.get_default_redirect(rule, method, rv,
                                                         query_args)
                if redirect_url is not None:
                    raise RequestRedirect(redirect_url)
            if rule.redirect_to is not None:
                if isinstance(rule.redirect_to, string_types):
                    def _handle_match(match):
                        value = rv[match.group(1)]
                        return rule._converters[match.group(1)].to_url(value)
                    redirect_url = _simple_rule_re.sub(_handle_match,
                                                       rule.redirect_to)
                else:
                    redirect_url = rule.redirect_to(self, **rv)
                raise RequestRedirect(str(url_join('%s://%s%s%s' % (
                    self.url_scheme or 'http',
                    self.subdomain and self.subdomain + '.' or '',
                    self.server_name,
                    self.script_name
                ), redirect_url)))
            if return_rule:
                return rule, rv
            else:
                return rule.endpoint, rv
        if have_match_for:
            raise MethodNotAllowed(valid_methods=list(have_match_for))
        raise NotFound()
    def test(self, path_info=None, method=None):
        try:
            self.match(path_info, method)
        except RequestRedirect:
            pass
        except HTTPException:
            return False
        return True
    def allowed_methods(self, path_info=None):
        try:
            self.match(path_info, method='--')
        except MethodNotAllowed as e:
            return e.valid_methods
        except HTTPException as e:
            pass
        return []
    def get_host(self, domain_part):
        if self.map.host_matching:
            if domain_part is None:
                return self.server_name
            return to_unicode(domain_part, 'ascii')
        subdomain = domain_part
        if subdomain is None:
            subdomain = self.subdomain
        else:
            subdomain = to_unicode(subdomain, 'ascii')
        return (subdomain and subdomain + u'.' or u'') + self.server_name
    def get_default_redirect(self, rule, method, values, query_args):
        assert self.map.redirect_defaults
        for r in self.map._rules_by_endpoint[rule.endpoint]:
            if r is rule:
                break
            if r.provides_defaults_for(rule) and               r.suitable_for(values, method):
                values.update(r.defaults)
                domain_part, path = r.build(values)
                return self.make_redirect_url(
                    path, query_args, domain_part=domain_part)
    def encode_query_args(self, query_args):
        if not isinstance(query_args, string_types):
            query_args = url_encode(query_args, self.map.charset)
        return query_args
    def make_redirect_url(self, path_info, query_args=None, domain_part=None):
        suffix = ''
        if query_args:
            suffix = '?' + self.encode_query_args(query_args)
        return str('%s://%s/%s%s' % (
            self.url_scheme or 'http',
            self.get_host(domain_part),
            posixpath.join(self.script_name[:-1].lstrip('/'),
                           path_info.lstrip('/')),
            suffix
        ))
    def make_alias_redirect_url(self, path, endpoint, values, method, query_args):
        url = self.build(endpoint, values, method, append_unknown=False,
                         force_external=True)
        if query_args:
            url += '?' + self.encode_query_args(query_args)
        assert url != path, 'detected invalid alias setting.  No canonical '            'URL found'
        return url
    def _partial_build(self, endpoint, values, method, append_unknown):
        if method is None:
            rv = self._partial_build(endpoint, values, self.default_method,
                                     append_unknown)
            if rv is not None:
                return rv
        for rule in self.map._rules_by_endpoint.get(endpoint, ()):
            if rule.suitable_for(values, method):
                rv = rule.build(values, append_unknown)
                if rv is not None:
                    return rv
    def build(self, endpoint, values=None, method=None, force_external=False,
              append_unknown=True):
        self.map.update()
        if values:
            if isinstance(values, MultiDict):
                valueiter = iteritems(values, multi=True)
            else:
                valueiter = iteritems(values)
            values = dict((k, v) for k, v in valueiter if v is not None)
        else:
            values = {}
        rv = self._partial_build(endpoint, values, method, append_unknown)
        if rv is None:
            raise BuildError(endpoint, values, method, self)
        domain_part, path = rv
        host = self.get_host(domain_part)
        if not force_external and (
            (self.map.host_matching and host == self.server_name) or
            (not self.map.host_matching and domain_part == self.subdomain)
        ):
            return str(url_join(self.script_name, './' + path.lstrip('/')))
        return str('%s//%s%s/%s' % (
            self.url_scheme + ':' if self.url_scheme else '',
            host,
            self.script_name[:-1],
            path.lstrip('/')
        ))
