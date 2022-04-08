
import inspect
import sys
import time
from functools import update_wrapper
from pip._vendor import six
try:
    MAX_WAIT = sys.maxint / 2
except AttributeError:
    MAX_WAIT = 1073741823
if six.PY2:
    from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES
    def wraps(fn):
        def filter_hasattr(obj, attrs):
            return tuple(a for a in attrs if hasattr(obj, a))
        return six.wraps(
            fn,
            assigned=filter_hasattr(fn, WRAPPER_ASSIGNMENTS),
            updated=filter_hasattr(fn, WRAPPER_UPDATES),
        )
    def capture(fut, tb):
        fut.set_exception_info(tb[1], tb[2])
    def getargspec(func):
        return inspect.getargspec(func)
else:
    from functools import wraps
    def capture(fut, tb):
        fut.set_exception(tb[1])
    def getargspec(func):
        return inspect.getfullargspec(func)
def visible_attrs(obj, attrs=None):
    if attrs is None:
        attrs = {}
    for attr_name, attr in inspect.getmembers(obj):
        if attr_name.startswith("_"):
            continue
        attrs[attr_name] = attr
    return attrs
def find_ordinal(pos_num):
    if pos_num == 0:
        return "th"
    elif pos_num == 1:
        return "st"
    elif pos_num == 2:
        return "nd"
    elif pos_num == 3:
        return "rd"
    elif pos_num >= 4 and pos_num <= 20:
        return "th"
    else:
        return find_ordinal(pos_num % 10)
def to_ordinal(pos_num):
    return "%i%s" % (pos_num, find_ordinal(pos_num))
def get_callback_name(cb):
    segments = []
    try:
        segments.append(cb.__qualname__)
    except AttributeError:
        try:
            segments.append(cb.__name__)
            if inspect.ismethod(cb):
                try:
                    segments.insert(0, cb.im_class.__name__)
                except AttributeError:
                    pass
        except AttributeError:
            pass
    if not segments:
        return repr(cb)
    else:
        try:
            if cb.__module__:
                segments.insert(0, cb.__module__)
        except AttributeError:
            pass
        return ".".join(segments)
try:
    now = time.monotonic
except AttributeError:
    from monotonic import monotonic as now
class cached_property(object):
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func
    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
