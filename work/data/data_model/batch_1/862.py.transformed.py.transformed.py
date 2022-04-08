
import sys
try:
    from urllib.parse import splittag
except ImportError:
    from urllib import splittag
def strip_fragment(url):
    url, fragment = splittag(url)
    return url
if sys.version_info >= (2, 7):
    strip_fragment = lambda x: x
try:
    from importlib import import_module
except ImportError:
    def import_module(module_name):
        return __import__(module_name, fromlist=['__name__'])
