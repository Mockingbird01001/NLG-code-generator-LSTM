
from pip._vendor import six
def get_exc_info_from_future(future):
    if six.PY3:
        return future.exception()
    else:
        ex, tb = future.exception_info()
        if ex is None:
            return None
        return type(ex), ex, tb
