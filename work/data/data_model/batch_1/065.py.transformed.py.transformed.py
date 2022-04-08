
import sys
from past.utils import with_metaclass, PY2
if PY2:
    str = unicode
ver = sys.version_info[:2]
class BaseBaseString(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, (bytes, str))
    def __subclasshook__(cls, thing):
        raise NotImplemented
class basestring(with_metaclass(BaseBaseString)):
__all__ = ['basestring']
