
"""
    jinja2.runtime
    ~~~~~~~~~~~~~~
    Runtime helpers.
    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD.
"""
import sys
from itertools import chain
from types import MethodType
from jinja2.nodes import EvalContext, _context_function_types
from jinja2.utils import Markup, soft_unicode, escape, missing, concat,     internalcode, object_type_repr, evalcontextfunction
from jinja2.exceptions import UndefinedError, TemplateRuntimeError,     TemplateNotFound
from jinja2._compat import imap, text_type, iteritems,     implements_iterator, implements_to_string, string_types, PY2,     with_metaclass
__all__ = ['LoopContext', 'TemplateReference', 'Macro', 'Markup',
           'TemplateRuntimeError', 'missing', 'concat', 'escape',
           'markup_join', 'unicode_join', 'to_string', 'identity',
           'TemplateNotFound']
to_string = text_type
identity = lambda x: x
_last_iteration = object()
def markup_join(seq):
    buf = []
    iterator = imap(soft_unicode, seq)
    for arg in iterator:
        buf.append(arg)
        if hasattr(arg, '__html__'):
            return Markup(u'').join(chain(buf, iterator))
    return concat(buf)
def unicode_join(seq):
    return concat(imap(text_type, seq))
def new_context(environment, template_name, blocks, vars=None,
                shared=None, globals=None, locals=None):
    if vars is None:
        vars = {}
    if shared:
        parent = vars
    else:
        parent = dict(globals or (), **vars)
    if locals:
        if shared:
            parent = dict(parent)
        for key, value in iteritems(locals):
            if value is not missing:
                parent[key] = value
    return environment.context_class(environment, parent, template_name,
                                     blocks)
class TemplateReference(object):
    def __init__(self, context):
        self.__context = context
    def __getitem__(self, name):
        blocks = self.__context.blocks[name]
        return BlockReference(name, self.__context, blocks, 0)
    def __repr__(self):
        return '<%s %r>' % (
            self.__class__.__name__,
            self.__context.name
        )
def _get_func(x):
    return getattr(x, '__func__', x)
class ContextMeta(type):
    def __new__(cls, name, bases, d):
        rv = type.__new__(cls, name, bases, d)
        if bases == ():
            return rv
        resolve = _get_func(rv.resolve)
        default_resolve = _get_func(Context.resolve)
        resolve_or_missing = _get_func(rv.resolve_or_missing)
        default_resolve_or_missing = _get_func(Context.resolve_or_missing)
        if resolve is not default_resolve and           resolve_or_missing is default_resolve_or_missing:
            rv._legacy_resolve_mode = True
        elif resolve is default_resolve and             resolve_or_missing is default_resolve_or_missing:
            rv._fast_resolve_mode = True
        return rv
def resolve_or_missing(context, key, missing=missing):
    if key in context.vars:
        return context.vars[key]
    if key in context.parent:
        return context.parent[key]
    return missing
class Context(with_metaclass(ContextMeta)):
    _legacy_resolve_mode = False
    _fast_resolve_mode = False
    def __init__(self, environment, parent, name, blocks):
        self.parent = parent
        self.vars = {}
        self.environment = environment
        self.eval_ctx = EvalContext(self.environment, name)
        self.exported_vars = set()
        self.name = name
        self.blocks = dict((k, [v]) for k, v in iteritems(blocks))
        if self._fast_resolve_mode:
            self.resolve_or_missing = MethodType(resolve_or_missing, self)
    def super(self, name, current):
        try:
            blocks = self.blocks[name]
            index = blocks.index(current) + 1
            blocks[index]
        except LookupError:
            return self.environment.undefined('there is no parent block '
                                              'called %r.' % name,
                                              name='super')
        return BlockReference(name, self, blocks, index)
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    def resolve(self, key):
        if self._legacy_resolve_mode:
            rv = resolve_or_missing(self, key)
        else:
            rv = self.resolve_or_missing(key)
        if rv is missing:
            return self.environment.undefined(name=key)
        return rv
    def resolve_or_missing(self, key):
        if self._legacy_resolve_mode:
            rv = self.resolve(key)
            if isinstance(rv, Undefined):
                rv = missing
            return rv
        return resolve_or_missing(self, key)
    def get_exported(self):
        return dict((k, self.vars[k]) for k in self.exported_vars)
    def get_all(self):
        if not self.vars:
            return self.parent
        if not self.parent:
            return self.vars
        return dict(self.parent, **self.vars)
    @internalcode
    def call(__self, __obj, *args, **kwargs):
        if __debug__:
            __traceback_hide__ = True
        fn = __obj.__call__
        for fn_type in ('contextfunction',
                        'evalcontextfunction',
                        'environmentfunction'):
            if hasattr(fn, fn_type):
                __obj = fn
                break
        if isinstance(__obj, _context_function_types):
            if getattr(__obj, 'contextfunction', 0):
                args = (__self,) + args
            elif getattr(__obj, 'evalcontextfunction', 0):
                args = (__self.eval_ctx,) + args
            elif getattr(__obj, 'environmentfunction', 0):
                args = (__self.environment,) + args
        try:
            return __obj(*args, **kwargs)
        except StopIteration:
            return __self.environment.undefined('value was undefined because '
                                                'a callable raised a '
                                                'StopIteration exception')
    def derived(self, locals=None):
        context = new_context(self.environment, self.name, {},
                              self.get_all(), True, None, locals)
        context.eval_ctx = self.eval_ctx
        context.blocks.update((k, list(v)) for k, v in iteritems(self.blocks))
        return context
    def _all(meth):
        proxy = lambda self: getattr(self.get_all(), meth)()
        proxy.__doc__ = getattr(dict, meth).__doc__
        proxy.__name__ = meth
        return proxy
    keys = _all('keys')
    values = _all('values')
    items = _all('items')
    if PY2:
        iterkeys = _all('iterkeys')
        itervalues = _all('itervalues')
        iteritems = _all('iteritems')
    del _all
    def __contains__(self, name):
        return name in self.vars or name in self.parent
    def __getitem__(self, key):
        item = self.resolve_or_missing(key)
        if item is missing:
            raise KeyError(key)
        return item
    def __repr__(self):
        return '<%s %s of %r>' % (
            self.__class__.__name__,
            repr(self.get_all()),
            self.name
        )
try:
    from collections import Mapping
    Mapping.register(Context)
except ImportError:
    pass
class BlockReference(object):
    def __init__(self, name, context, stack, depth):
        self.name = name
        self._context = context
        self._stack = stack
        self._depth = depth
    @property
    def super(self):
        if self._depth + 1 >= len(self._stack):
            return self._context.environment.                undefined('there is no parent block called %r.' %
                          self.name, name='super')
        return BlockReference(self.name, self._context, self._stack,
                              self._depth + 1)
    @internalcode
    def __call__(self):
        rv = concat(self._stack[self._depth](self._context))
        if self._context.eval_ctx.autoescape:
            rv = Markup(rv)
        return rv
class LoopContextBase(object):
    _after = _last_iteration
    _length = None
    def __init__(self, recurse=None, depth0=0):
        self._recurse = recurse
        self.index0 = -1
        self.depth0 = depth0
    def cycle(self, *args):
        if not args:
            raise TypeError('no items for cycling given')
        return args[self.index0 % len(args)]
    first = property(lambda x: x.index0 == 0)
    last = property(lambda x: x._after is _last_iteration)
    index = property(lambda x: x.index0 + 1)
    revindex = property(lambda x: x.length - x.index0)
    revindex0 = property(lambda x: x.length - x.index)
    depth = property(lambda x: x.depth0 + 1)
    def __len__(self):
        return self.length
    @internalcode
    def loop(self, iterable):
        if self._recurse is None:
            raise TypeError('Tried to call non recursive loop.  Maybe you '
                            "forgot the 'recursive' modifier.")
        return self._recurse(iterable, self._recurse, self.depth0 + 1)
    __call__ = loop
    del loop
    def __repr__(self):
        return '<%s %r/%r>' % (
            self.__class__.__name__,
            self.index,
            self.length
        )
class LoopContext(LoopContextBase):
    def __init__(self, iterable, recurse=None, depth0=0):
        LoopContextBase.__init__(self, recurse, depth0)
        self._iterator = iter(iterable)
        try:
            self._length = len(iterable)
        except (TypeError, AttributeError):
            self._length = None
        self._after = self._safe_next()
    @property
    def length(self):
        if self._length is None:
            iterable = tuple(self._iterator)
            self._iterator = iter(iterable)
            iterations_done = self.index0 + 2
            self._length = len(iterable) + iterations_done
        return self._length
    def __iter__(self):
        return LoopContextIterator(self)
    def _safe_next(self):
        try:
            return next(self._iterator)
        except StopIteration:
            return _last_iteration
@implements_iterator
class LoopContextIterator(object):
    __slots__ = ('context',)
    def __init__(self, context):
        self.context = context
    def __iter__(self):
        return self
    def __next__(self):
        ctx = self.context
        ctx.index0 += 1
        if ctx._after is _last_iteration:
            raise StopIteration()
        next_elem = ctx._after
        ctx._after = ctx._safe_next()
        return next_elem, ctx
class Macro(object):
    def __init__(self, environment, func, name, arguments,
                 catch_kwargs, catch_varargs, caller,
                 default_autoescape=None):
        self._environment = environment
        self._func = func
        self._argument_count = len(arguments)
        self.name = name
        self.arguments = arguments
        self.catch_kwargs = catch_kwargs
        self.catch_varargs = catch_varargs
        self.caller = caller
        self.explicit_caller = 'caller' in arguments
        if default_autoescape is None:
            default_autoescape = environment.autoescape
        self._default_autoescape = default_autoescape
    @internalcode
    @evalcontextfunction
    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], EvalContext):
            autoescape = args[0].autoescape
            args = args[1:]
        else:
            autoescape = self._default_autoescape
        arguments = list(args[:self._argument_count])
        off = len(arguments)
        found_caller = False
        if off != self._argument_count:
            for idx, name in enumerate(self.arguments[len(arguments):]):
                try:
                    value = kwargs.pop(name)
                except KeyError:
                    value = missing
                if name == 'caller':
                    found_caller = True
                arguments.append(value)
        else:
            found_caller = self.explicit_caller
        if self.caller and not found_caller:
            caller = kwargs.pop('caller', None)
            if caller is None:
                caller = self._environment.undefined('No caller defined',
                                                     name='caller')
            arguments.append(caller)
        if self.catch_kwargs:
            arguments.append(kwargs)
        elif kwargs:
            if 'caller' in kwargs:
                raise TypeError('macro %r was invoked with two values for '
                                'the special caller argument.  This is '
                                'most likely a bug.' % self.name)
            raise TypeError('macro %r takes no keyword argument %r' %
                            (self.name, next(iter(kwargs))))
        if self.catch_varargs:
            arguments.append(args[self._argument_count:])
        elif len(args) > self._argument_count:
            raise TypeError('macro %r takes not more than %d argument(s)' %
                            (self.name, len(self.arguments)))
        return self._invoke(arguments, autoescape)
    def _invoke(self, arguments, autoescape):
        rv = self._func(*arguments)
        if autoescape:
            rv = Markup(rv)
        return rv
    def __repr__(self):
        return '<%s %s>' % (
            self.__class__.__name__,
            self.name is None and 'anonymous' or repr(self.name)
        )
@implements_to_string
class Undefined(object):
    __slots__ = ('_undefined_hint', '_undefined_obj', '_undefined_name',
                 '_undefined_exception')
    def __init__(self, hint=None, obj=missing, name=None, exc=UndefinedError):
        self._undefined_hint = hint
        self._undefined_obj = obj
        self._undefined_name = name
        self._undefined_exception = exc
    @internalcode
    def _fail_with_undefined_error(self, *args, **kwargs):
        if self._undefined_hint is None:
            if self._undefined_obj is missing:
                hint = '%r is undefined' % self._undefined_name
            elif not isinstance(self._undefined_name, string_types):
                hint = '%s has no element %r' % (
                    object_type_repr(self._undefined_obj),
                    self._undefined_name
                )
            else:
                hint = '%r has no attribute %r' % (
                    object_type_repr(self._undefined_obj),
                    self._undefined_name
                )
        else:
            hint = self._undefined_hint
        raise self._undefined_exception(hint)
    @internalcode
    def __getattr__(self, name):
        if name[:2] == '__':
            raise AttributeError(name)
        return self._fail_with_undefined_error()
    __add__ = __radd__ = __mul__ = __rmul__ = __div__ = __rdiv__ =        __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ =        __mod__ = __rmod__ = __pos__ = __neg__ = __call__ =        __getitem__ = __lt__ = __le__ = __gt__ = __ge__ = __int__ =        __float__ = __complex__ = __pow__ = __rpow__ = __sub__ =        __rsub__ = _fail_with_undefined_error
    def __eq__(self, other):
        return type(self) is type(other)
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return id(type(self))
    def __str__(self):
        return u''
    def __len__(self):
        return 0
    def __iter__(self):
        if 0:
            yield None
    def __nonzero__(self):
        return False
    __bool__ = __nonzero__
    def __repr__(self):
        return 'Undefined'
def make_logging_undefined(logger=None, base=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler(sys.stderr))
    if base is None:
        base = Undefined
    def _log_message(undef):
        if undef._undefined_hint is None:
            if undef._undefined_obj is missing:
                hint = '%s is undefined' % undef._undefined_name
            elif not isinstance(undef._undefined_name, string_types):
                hint = '%s has no element %s' % (
                    object_type_repr(undef._undefined_obj),
                    undef._undefined_name)
            else:
                hint = '%s has no attribute %s' % (
                    object_type_repr(undef._undefined_obj),
                    undef._undefined_name)
        else:
            hint = undef._undefined_hint
        logger.warning('Template variable warning: %s', hint)
    class LoggingUndefined(base):
        def _fail_with_undefined_error(self, *args, **kwargs):
            try:
                return base._fail_with_undefined_error(self, *args, **kwargs)
            except self._undefined_exception as e:
                logger.error('Template variable error: %s', str(e))
                raise e
        def __str__(self):
            rv = base.__str__(self)
            _log_message(self)
            return rv
        def __iter__(self):
            rv = base.__iter__(self)
            _log_message(self)
            return rv
        if PY2:
            def __nonzero__(self):
                rv = base.__nonzero__(self)
                _log_message(self)
                return rv
            def __unicode__(self):
                rv = base.__unicode__(self)
                _log_message(self)
                return rv
        else:
            def __bool__(self):
                rv = base.__bool__(self)
                _log_message(self)
                return rv
    return LoggingUndefined
@implements_to_string
class DebugUndefined(Undefined):
    __slots__ = ()
    def __str__(self):
        if self._undefined_hint is None:
            if self._undefined_obj is missing:
                return u'{{ %s }}' % self._undefined_name
            return '{{ no such element: %s[%r] }}' % (
                object_type_repr(self._undefined_obj),
                self._undefined_name
            )
        return u'{{ undefined value printed: %s }}' % self._undefined_hint
@implements_to_string
class StrictUndefined(Undefined):
    __slots__ = ()
    __iter__ = __str__ = __len__ = __nonzero__ = __eq__ =        __ne__ = __bool__ = __hash__ =        Undefined._fail_with_undefined_error
del Undefined.__slots__, DebugUndefined.__slots__, StrictUndefined.__slots__
