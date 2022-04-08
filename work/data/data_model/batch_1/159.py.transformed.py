
import collections.abc
from .utils import SkipTest, assert_warns, HAS_REFCOUNT
__all__ = ['slow', 'setastest', 'skipif', 'knownfailureif', 'deprecated',
           'parametrize', '_needs_refcount',]
def slow(t):
    t.slow = True
    return t
def setastest(tf=True):
    def set_test(t):
        t.__test__ = tf
        return t
    return set_test
def skipif(skip_condition, msg=None):
    def skip_decorator(f):
        import nose
        if isinstance(skip_condition, collections.abc.Callable):
            skip_val = lambda: skip_condition()
        else:
            skip_val = lambda: skip_condition
        def get_msg(func,msg=None):
            if msg is None:
                out = 'Test skipped due to test condition'
            else:
                out = msg
            return f'Skipping test: {func.__name__}: {out}'
        def skipper_func(*args, **kwargs):
            if skip_val():
                raise SkipTest(get_msg(f, msg))
            else:
                return f(*args, **kwargs)
        def skipper_gen(*args, **kwargs):
            if skip_val():
                raise SkipTest(get_msg(f, msg))
            else:
                yield from f(*args, **kwargs)
        if nose.util.isgenerator(f):
            skipper = skipper_gen
        else:
            skipper = skipper_func
        return nose.tools.make_decorator(f)(skipper)
    return skip_decorator
def knownfailureif(fail_condition, msg=None):
    if msg is None:
        msg = 'Test skipped due to known failure'
    if isinstance(fail_condition, collections.abc.Callable):
        fail_val = lambda: fail_condition()
    else:
        fail_val = lambda: fail_condition
    def knownfail_decorator(f):
        import nose
        from .noseclasses import KnownFailureException
        def knownfailer(*args, **kwargs):
            if fail_val():
                raise KnownFailureException(msg)
            else:
                return f(*args, **kwargs)
        return nose.tools.make_decorator(f)(knownfailer)
    return knownfail_decorator
def deprecated(conditional=True):
    def deprecate_decorator(f):
        import nose
        def _deprecated_imp(*args, **kwargs):
            with assert_warns(DeprecationWarning):
                f(*args, **kwargs)
        if isinstance(conditional, collections.abc.Callable):
            cond = conditional()
        else:
            cond = conditional
        if cond:
            return nose.tools.make_decorator(f)(_deprecated_imp)
        else:
            return f
    return deprecate_decorator
def parametrize(vars, input):
    from .parameterized import parameterized
    return parameterized(input)
_needs_refcount = skipif(not HAS_REFCOUNT, "python has no sys.getrefcount")
