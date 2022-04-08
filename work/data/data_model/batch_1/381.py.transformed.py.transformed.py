
pattern_unformatted = u"%s=%s"
message_unformatted =
class Feature(object):
    def __init__(self, name, PATTERN, version):
        self.name = name
        self._pattern = PATTERN
        self.version = version
    def message_text(self):
        return message_unformatted % (self.name, self.version)
class Features(set):
    mapping = {}
    def update_mapping(self):
        self.mapping = dict([(f.name, f) for f in iter(self)])
    @property
    def PATTERN(self):
        self.update_mapping()
        return u" |\n".join([pattern_unformatted % (f.name, f._pattern) for f in iter(self)])
    def __getitem__(self, key):
        return self.mapping[key]
