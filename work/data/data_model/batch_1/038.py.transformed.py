
"""
    flask.blueprints
    ~~~~~~~~~~~~~~~~
    Blueprints are the recommended way to implement larger or more
    pluggable applications in Flask 0.7 and later.
    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""
from functools import update_wrapper
from .helpers import _PackageBoundObject, _endpoint_from_view_func
class BlueprintSetupState(object):
    def __init__(self, blueprint, app, options, first_registration):
        self.app = app
        self.blueprint = blueprint
        self.options = options
        self.first_registration = first_registration
        subdomain = self.options.get('subdomain')
        if subdomain is None:
            subdomain = self.blueprint.subdomain
        self.subdomain = subdomain
        url_prefix = self.options.get('url_prefix')
        if url_prefix is None:
            url_prefix = self.blueprint.url_prefix
        self.url_prefix = url_prefix
        self.url_defaults = dict(self.blueprint.url_values_defaults)
        self.url_defaults.update(self.options.get('url_defaults', ()))
    def add_url_rule(self, rule, endpoint=None, view_func=None, **options):
        if self.url_prefix:
            rule = self.url_prefix + rule
        options.setdefault('subdomain', self.subdomain)
        if endpoint is None:
            endpoint = _endpoint_from_view_func(view_func)
        defaults = self.url_defaults
        if 'defaults' in options:
            defaults = dict(defaults, **options.pop('defaults'))
        self.app.add_url_rule(rule, '%s.%s' % (self.blueprint.name, endpoint),
                              view_func, defaults=defaults, **options)
class Blueprint(_PackageBoundObject):
    warn_on_modifications = False
    _got_registered_once = False
    def __init__(self, name, import_name, static_folder=None,
                 static_url_path=None, template_folder=None,
                 url_prefix=None, subdomain=None, url_defaults=None,
                 root_path=None):
        _PackageBoundObject.__init__(self, import_name, template_folder,
                                     root_path=root_path)
        self.name = name
        self.url_prefix = url_prefix
        self.subdomain = subdomain
        self.static_folder = static_folder
        self.static_url_path = static_url_path
        self.deferred_functions = []
        if url_defaults is None:
            url_defaults = {}
        self.url_values_defaults = url_defaults
    def record(self, func):
        if self._got_registered_once and self.warn_on_modifications:
            from warnings import warn
            warn(Warning('The blueprint was already registered once '
                         'but is getting modified now.  These changes '
                         'will not show up.'))
        self.deferred_functions.append(func)
    def record_once(self, func):
        def wrapper(state):
            if state.first_registration:
                func(state)
        return self.record(update_wrapper(wrapper, func))
    def make_setup_state(self, app, options, first_registration=False):
        return BlueprintSetupState(self, app, options, first_registration)
    def register(self, app, options, first_registration=False):
        self._got_registered_once = True
        state = self.make_setup_state(app, options, first_registration)
        if self.has_static_folder:
            state.add_url_rule(self.static_url_path + '/<path:filename>',
                               view_func=self.send_static_file,
                               endpoint='static')
        for deferred in self.deferred_functions:
            deferred(state)
    def route(self, rule, **options):
        def decorator(f):
            endpoint = options.pop("endpoint", f.__name__)
            self.add_url_rule(rule, endpoint, f, **options)
            return f
        return decorator
    def add_url_rule(self, rule, endpoint=None, view_func=None, **options):
        if endpoint:
            assert '.' not in endpoint, "Blueprint endpoints should not contain dots"
        self.record(lambda s:
            s.add_url_rule(rule, endpoint, view_func, **options))
    def endpoint(self, endpoint):
        def decorator(f):
            def register_endpoint(state):
                state.app.view_functions[endpoint] = f
            self.record_once(register_endpoint)
            return f
        return decorator
    def app_template_filter(self, name=None):
        def decorator(f):
            self.add_app_template_filter(f, name=name)
            return f
        return decorator
    def add_app_template_filter(self, f, name=None):
        def register_template(state):
            state.app.jinja_env.filters[name or f.__name__] = f
        self.record_once(register_template)
    def app_template_test(self, name=None):
        def decorator(f):
            self.add_app_template_test(f, name=name)
            return f
        return decorator
    def add_app_template_test(self, f, name=None):
        def register_template(state):
            state.app.jinja_env.tests[name or f.__name__] = f
        self.record_once(register_template)
    def app_template_global(self, name=None):
        def decorator(f):
            self.add_app_template_global(f, name=name)
            return f
        return decorator
    def add_app_template_global(self, f, name=None):
        def register_template(state):
            state.app.jinja_env.globals[name or f.__name__] = f
        self.record_once(register_template)
    def before_request(self, f):
        self.record_once(lambda s: s.app.before_request_funcs
            .setdefault(self.name, []).append(f))
        return f
    def before_app_request(self, f):
        self.record_once(lambda s: s.app.before_request_funcs
            .setdefault(None, []).append(f))
        return f
    def before_app_first_request(self, f):
        self.record_once(lambda s: s.app.before_first_request_funcs.append(f))
        return f
    def after_request(self, f):
        self.record_once(lambda s: s.app.after_request_funcs
            .setdefault(self.name, []).append(f))
        return f
    def after_app_request(self, f):
        self.record_once(lambda s: s.app.after_request_funcs
            .setdefault(None, []).append(f))
        return f
    def teardown_request(self, f):
        self.record_once(lambda s: s.app.teardown_request_funcs
            .setdefault(self.name, []).append(f))
        return f
    def teardown_app_request(self, f):
        self.record_once(lambda s: s.app.teardown_request_funcs
            .setdefault(None, []).append(f))
        return f
    def context_processor(self, f):
        self.record_once(lambda s: s.app.template_context_processors
            .setdefault(self.name, []).append(f))
        return f
    def app_context_processor(self, f):
        self.record_once(lambda s: s.app.template_context_processors
            .setdefault(None, []).append(f))
        return f
    def app_errorhandler(self, code):
        def decorator(f):
            self.record_once(lambda s: s.app.errorhandler(code)(f))
            return f
        return decorator
    def url_value_preprocessor(self, f):
        self.record_once(lambda s: s.app.url_value_preprocessors
            .setdefault(self.name, []).append(f))
        return f
    def url_defaults(self, f):
        self.record_once(lambda s: s.app.url_default_functions
            .setdefault(self.name, []).append(f))
        return f
    def app_url_value_preprocessor(self, f):
        self.record_once(lambda s: s.app.url_value_preprocessors
            .setdefault(None, []).append(f))
        return f
    def app_url_defaults(self, f):
        self.record_once(lambda s: s.app.url_default_functions
            .setdefault(None, []).append(f))
        return f
    def errorhandler(self, code_or_exception):
        def decorator(f):
            self.record_once(lambda s: s.app._register_error_handler(
                self.name, code_or_exception, f))
            return f
        return decorator
    def register_error_handler(self, code_or_exception, f):
        self.record_once(lambda s: s.app._register_error_handler(
            self.name, code_or_exception, f))
