
import pymongo
import six
if pymongo.version_tuple[0] < 3:
    IS_PYMONGO_3 = False
else:
    IS_PYMONGO_3 = True
StringIO = six.BytesIO
if not six.PY3:
    try:
        import cStringIO
    except ImportError:
        pass
    else:
        StringIO = cStringIO.StringIO
