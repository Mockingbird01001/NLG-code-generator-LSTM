
from bson.py3compat import PY3
if PY3:
    long = int
class Int64(long):
    _type_marker = 18
