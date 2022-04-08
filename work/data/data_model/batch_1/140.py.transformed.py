
import warnings
import weakref
from bson.py3compat import integer_types
class CursorManager(object):
    def __init__(self, client):
        warnings.warn(
            "Cursor managers are deprecated.",
            DeprecationWarning,
            stacklevel=2)
        self.__client = weakref.ref(client)
    def close(self, cursor_id, address):
        if not isinstance(cursor_id, integer_types):
            raise TypeError("cursor_id must be an integer")
        self.__client().kill_cursors([cursor_id], address)
