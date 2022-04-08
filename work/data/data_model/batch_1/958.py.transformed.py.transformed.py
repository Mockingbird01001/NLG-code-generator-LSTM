
SCHEME_KEYS = ['platlib', 'purelib', 'headers', 'scripts', 'data']
class Scheme:
    __slots__ = SCHEME_KEYS
    def __init__(
        self,
        platlib,
        purelib,
        headers,
        scripts,
        data,
    ):
        self.platlib = platlib
        self.purelib = purelib
        self.headers = headers
        self.scripts = scripts
        self.data = data
