
import platform
from setuptools.extern import six
def get_all_headers(message, key):
    return message.get_all(key)
if six.PY2:
    def get_all_headers(message, key):
        return message.getheaders(key)
linux_py2_ascii = (
    platform.system() == 'Linux' and
    six.PY2
)
rmtree_safe = str if linux_py2_ascii else lambda x: x
