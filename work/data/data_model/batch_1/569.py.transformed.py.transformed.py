
"""
requests._internal_utils
~~~~~~~~~~~~~~
Provides utility functions that are consumed internally by Requests
which depend on extremely few external helpers (such as compat)
"""
from .compat import is_py2, builtin_str, str
def to_native_string(string, encoding='ascii'):
    if isinstance(string, builtin_str):
        out = string
    else:
        if is_py2:
            out = string.encode(encoding)
        else:
            out = string.decode(encoding)
    return out
def unicode_is_ascii(u_string):
    assert isinstance(u_string, str)
    try:
        u_string.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False
