import os
import stat
from ._compat import open_stream, text_type, filename_to_ui,    get_filesystem_encoding, get_streerror, _get_argv_encoding, PY2
from .exceptions import BadParameter
from .utils import safecall, LazyFile
class ParamType(object):
    is_composite = False
    name = None
    envvar_list_splitter = None
    def __call__(self, value, param=None, ctx=None):
        if value is not None:
            return self.convert(value, param, ctx)
    def get_metavar(self, param):
    def get_missing_message(self, param):
    def convert(self, value, param, ctx):
        return value
    def split_envvar_value(self, rv):
        return (rv or '').split(self.envvar_list_splitter)
    def fail(self, message, param=None, ctx=None):
        raise BadParameter(message, ctx=ctx, param=param)
class CompositeParamType(ParamType):
    is_composite = True
    @property
    def arity(self):
        raise NotImplementedError()
class FuncParamType(ParamType):
    def __init__(self, func):
        self.name = func.__name__
        self.func = func
    def convert(self, value, param, ctx):
        try:
            return self.func(value)
        except ValueError:
            try:
                value = text_type(value)
            except UnicodeError:
                value = str(value).decode('utf-8', 'replace')
            self.fail(value, param, ctx)
class UnprocessedParamType(ParamType):
    name = 'text'
    def convert(self, value, param, ctx):
        return value
    def __repr__(self):
        return 'UNPROCESSED'
class StringParamType(ParamType):
    name = 'text'
    def convert(self, value, param, ctx):
        if isinstance(value, bytes):
            enc = _get_argv_encoding()
            try:
                value = value.decode(enc)
            except UnicodeError:
                fs_enc = get_filesystem_encoding()
                if fs_enc != enc:
                    try:
                        value = value.decode(fs_enc)
                    except UnicodeError:
                        value = value.decode('utf-8', 'replace')
            return value
        return value
    def __repr__(self):
        return 'STRING'
class Choice(ParamType):
    name = 'choice'
    def __init__(self, choices):
        self.choices = choices
    def get_metavar(self, param):
        return '[%s]' % '|'.join(self.choices)
    def get_missing_message(self, param):
        return 'Choose from %s.' % ', '.join(self.choices)
    def convert(self, value, param, ctx):
        if value in self.choices:
            return value
        if ctx is not None and           ctx.token_normalize_func is not None:
            value = ctx.token_normalize_func(value)
            for choice in self.choices:
                if ctx.token_normalize_func(choice) == value:
                    return choice
        self.fail('invalid choice: %s. (choose from %s)' %
                  (value, ', '.join(self.choices)), param, ctx)
    def __repr__(self):
        return 'Choice(%r)' % list(self.choices)
class IntParamType(ParamType):
    name = 'integer'
    def convert(self, value, param, ctx):
        try:
            return int(value)
        except (ValueError, UnicodeError):
            self.fail('%s is not a valid integer' % value, param, ctx)
    def __repr__(self):
        return 'INT'
class IntRange(IntParamType):
    name = 'integer range'
    def __init__(self, min=None, max=None, clamp=False):
        self.min = min
        self.max = max
        self.clamp = clamp
    def convert(self, value, param, ctx):
        rv = IntParamType.convert(self, value, param, ctx)
        if self.clamp:
            if self.min is not None and rv < self.min:
                return self.min
            if self.max is not None and rv > self.max:
                return self.max
        if self.min is not None and rv < self.min or           self.max is not None and rv > self.max:
            if self.min is None:
                self.fail('%s is bigger than the maximum valid value '
                          '%s.' % (rv, self.max), param, ctx)
            elif self.max is None:
                self.fail('%s is smaller than the minimum valid value '
                          '%s.' % (rv, self.min), param, ctx)
            else:
                self.fail('%s is not in the valid range of %s to %s.'
                          % (rv, self.min, self.max), param, ctx)
        return rv
    def __repr__(self):
        return 'IntRange(%r, %r)' % (self.min, self.max)
class BoolParamType(ParamType):
    name = 'boolean'
    def convert(self, value, param, ctx):
        if isinstance(value, bool):
            return bool(value)
        value = value.lower()
        if value in ('true', '1', 'yes', 'y'):
            return True
        elif value in ('false', '0', 'no', 'n'):
            return False
        self.fail('%s is not a valid boolean' % value, param, ctx)
    def __repr__(self):
        return 'BOOL'
class FloatParamType(ParamType):
    name = 'float'
    def convert(self, value, param, ctx):
        try:
            return float(value)
        except (UnicodeError, ValueError):
            self.fail('%s is not a valid floating point value' %
                      value, param, ctx)
    def __repr__(self):
        return 'FLOAT'
class UUIDParameterType(ParamType):
    name = 'uuid'
    def convert(self, value, param, ctx):
        import uuid
        try:
            if PY2 and isinstance(value, text_type):
                value = value.encode('ascii')
            return uuid.UUID(value)
        except (UnicodeError, ValueError):
            self.fail('%s is not a valid UUID value' % value, param, ctx)
    def __repr__(self):
        return 'UUID'
class File(ParamType):
    name = 'filename'
    envvar_list_splitter = os.path.pathsep
    def __init__(self, mode='r', encoding=None, errors='strict', lazy=None,
                 atomic=False):
        self.mode = mode
        self.encoding = encoding
        self.errors = errors
        self.lazy = lazy
        self.atomic = atomic
    def resolve_lazy_flag(self, value):
        if self.lazy is not None:
            return self.lazy
        if value == '-':
            return False
        elif 'w' in self.mode:
            return True
        return False
    def convert(self, value, param, ctx):
        try:
            if hasattr(value, 'read') or hasattr(value, 'write'):
                return value
            lazy = self.resolve_lazy_flag(value)
            if lazy:
                f = LazyFile(value, self.mode, self.encoding, self.errors,
                             atomic=self.atomic)
                if ctx is not None:
                    ctx.call_on_close(f.close_intelligently)
                return f
            f, should_close = open_stream(value, self.mode,
                                          self.encoding, self.errors,
                                          atomic=self.atomic)
            if ctx is not None:
                if should_close:
                    ctx.call_on_close(safecall(f.close))
                else:
                    ctx.call_on_close(safecall(f.flush))
            return f
        except (IOError, OSError) as e:
            self.fail('Could not open file: %s: %s' % (
                filename_to_ui(value),
                get_streerror(e),
            ), param, ctx)
class Path(ParamType):
    envvar_list_splitter = os.path.pathsep
    def __init__(self, exists=False, file_okay=True, dir_okay=True,
                 writable=False, readable=True, resolve_path=False,
                 allow_dash=False, path_type=None):
        self.exists = exists
        self.file_okay = file_okay
        self.dir_okay = dir_okay
        self.writable = writable
        self.readable = readable
        self.resolve_path = resolve_path
        self.allow_dash = allow_dash
        self.type = path_type
        if self.file_okay and not self.dir_okay:
            self.name = 'file'
            self.path_type = 'File'
        if self.dir_okay and not self.file_okay:
            self.name = 'directory'
            self.path_type = 'Directory'
        else:
            self.name = 'path'
            self.path_type = 'Path'
    def coerce_path_result(self, rv):
        if self.type is not None and not isinstance(rv, self.type):
            if self.type is text_type:
                rv = rv.decode(get_filesystem_encoding())
            else:
                rv = rv.encode(get_filesystem_encoding())
        return rv
    def convert(self, value, param, ctx):
        rv = value
        is_dash = self.file_okay and self.allow_dash and rv in (b'-', '-')
        if not is_dash:
            if self.resolve_path:
                rv = os.path.realpath(rv)
            try:
                st = os.stat(rv)
            except OSError:
                if not self.exists:
                    return self.coerce_path_result(rv)
                self.fail('%s "%s" does not exist.' % (
                    self.path_type,
                    filename_to_ui(value)
                ), param, ctx)
            if not self.file_okay and stat.S_ISREG(st.st_mode):
                self.fail('%s "%s" is a file.' % (
                    self.path_type,
                    filename_to_ui(value)
                ), param, ctx)
            if not self.dir_okay and stat.S_ISDIR(st.st_mode):
                self.fail('%s "%s" is a directory.' % (
                    self.path_type,
                    filename_to_ui(value)
                ), param, ctx)
            if self.writable and not os.access(value, os.W_OK):
                self.fail('%s "%s" is not writable.' % (
                    self.path_type,
                    filename_to_ui(value)
                ), param, ctx)
            if self.readable and not os.access(value, os.R_OK):
                self.fail('%s "%s" is not readable.' % (
                    self.path_type,
                    filename_to_ui(value)
                ), param, ctx)
        return self.coerce_path_result(rv)
class Tuple(CompositeParamType):
    def __init__(self, types):
        self.types = [convert_type(ty) for ty in types]
    @property
    def name(self):
        return "<" + " ".join(ty.name for ty in self.types) + ">"
    @property
    def arity(self):
        return len(self.types)
    def convert(self, value, param, ctx):
        if len(value) != len(self.types):
            raise TypeError('It would appear that nargs is set to conflict '
                            'with the composite type arity.')
        return tuple(ty(x, param, ctx) for ty, x in zip(self.types, value))
def convert_type(ty, default=None):
    guessed_type = False
    if ty is None and default is not None:
        if isinstance(default, tuple):
            ty = tuple(map(type, default))
        else:
            ty = type(default)
        guessed_type = True
    if isinstance(ty, tuple):
        return Tuple(ty)
    if isinstance(ty, ParamType):
        return ty
    if ty is text_type or ty is str or ty is None:
        return STRING
    if ty is int:
        return INT
    if ty is bool and not guessed_type:
        return BOOL
    if ty is float:
        return FLOAT
    if guessed_type:
        return STRING
    if __debug__:
        try:
            if issubclass(ty, ParamType):
                raise AssertionError('Attempted to use an uninstantiated '
                                     'parameter type (%s).' % ty)
        except TypeError:
            pass
    return FuncParamType(ty)
UNPROCESSED = UnprocessedParamType()
STRING = StringParamType()
INT = IntParamType()
FLOAT = FloatParamType()
BOOL = BoolParamType()
UUID = UUIDParameterType()
