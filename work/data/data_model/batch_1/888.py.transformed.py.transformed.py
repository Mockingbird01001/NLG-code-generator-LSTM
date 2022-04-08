
from collections import Mapping
from bson.py3compat import integer_types
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (member_with_tags_server_selector,
                                      secondary_with_tags_server_selector)
_PRIMARY = 0
_PRIMARY_PREFERRED = 1
_SECONDARY = 2
_SECONDARY_PREFERRED = 3
_NEAREST = 4
_MONGOS_MODES = (
    'primary',
    'primaryPreferred',
    'secondary',
    'secondaryPreferred',
    'nearest',
)
def _validate_tag_sets(tag_sets):
    if tag_sets is None:
        return tag_sets
    if not isinstance(tag_sets, list):
        raise TypeError((
            "Tag sets %r invalid, must be a list") % (tag_sets,))
    if len(tag_sets) == 0:
        raise ValueError((
            "Tag sets %r invalid, must be None or contain at least one set of"
            " tags") % (tag_sets,))
    for tags in tag_sets:
        if not isinstance(tags, Mapping):
            raise TypeError(
                "Tag set %r invalid, must be an instance of dict, "
                "bson.son.SON or other type that inherits from "
                "collection.Mapping" % (tags,))
    return tag_sets
def _invalid_max_staleness_msg(max_staleness):
    return ("maxStalenessSeconds must be a positive integer, not %s" %
            max_staleness)
def _validate_max_staleness(max_staleness):
    if max_staleness == -1:
        return -1
    if not isinstance(max_staleness, integer_types):
        raise TypeError(_invalid_max_staleness_msg(max_staleness))
    if max_staleness <= 0:
        raise ValueError(_invalid_max_staleness_msg(max_staleness))
    return max_staleness
class _ServerMode(object):
    __slots__ = ("__mongos_mode", "__mode", "__tag_sets", "__max_staleness")
    def __init__(self, mode, tag_sets=None, max_staleness=-1):
        self.__mongos_mode = _MONGOS_MODES[mode]
        self.__mode = mode
        self.__tag_sets = _validate_tag_sets(tag_sets)
        self.__max_staleness = _validate_max_staleness(max_staleness)
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def mongos_mode(self):
        return self.__mongos_mode
    @property
    def document(self):
        doc = {'mode': self.__mongos_mode}
        if self.__tag_sets not in (None, [{}]):
            doc['tags'] = self.__tag_sets
        if self.__max_staleness != -1:
            doc['maxStalenessSeconds'] = self.__max_staleness
        return doc
    @property
    def mode(self):
        return self.__mode
    @property
    def tag_sets(self):
        return list(self.__tag_sets) if self.__tag_sets else [{}]
    @property
    def max_staleness(self):
        return self.__max_staleness
    @property
    def min_wire_version(self):
        return 0 if self.__max_staleness == -1 else 5
    def __repr__(self):
        return "%s(tag_sets=%r, max_staleness=%r)" % (
            self.name, self.__tag_sets, self.__max_staleness)
    def __eq__(self, other):
        if isinstance(other, _ServerMode):
            return (self.mode == other.mode and
                    self.tag_sets == other.tag_sets and
                    self.max_staleness == other.max_staleness)
        return NotImplemented
    def __ne__(self, other):
        return not self == other
    def __getstate__(self):
        return {'mode': self.__mode,
                'tag_sets': self.__tag_sets,
                'max_staleness': self.__max_staleness}
    def __setstate__(self, value):
        self.__mode = value['mode']
        self.__mongos_mode = _MONGOS_MODES[self.__mode]
        self.__tag_sets = _validate_tag_sets(value['tag_sets'])
        self.__max_staleness = _validate_max_staleness(value['max_staleness'])
class Primary(_ServerMode):
    __slots__ = ()
    def __init__(self):
        super(Primary, self).__init__(_PRIMARY)
    def __call__(self, selection):
        return selection.primary_selection
    def __repr__(self):
        return "Primary()"
    def __eq__(self, other):
        if isinstance(other, _ServerMode):
            return other.mode == _PRIMARY
        return NotImplemented
class PrimaryPreferred(_ServerMode):
    __slots__ = ()
    def __init__(self, tag_sets=None, max_staleness=-1):
        super(PrimaryPreferred, self).__init__(_PRIMARY_PREFERRED,
                                               tag_sets,
                                               max_staleness)
    def __call__(self, selection):
        if selection.primary:
            return selection.primary_selection
        else:
            return secondary_with_tags_server_selector(
                self.tag_sets,
                max_staleness_selectors.select(
                    self.max_staleness, selection))
class Secondary(_ServerMode):
    __slots__ = ()
    def __init__(self, tag_sets=None, max_staleness=-1):
        super(Secondary, self).__init__(_SECONDARY, tag_sets, max_staleness)
    def __call__(self, selection):
        return secondary_with_tags_server_selector(
            self.tag_sets,
            max_staleness_selectors.select(
                self.max_staleness, selection))
class SecondaryPreferred(_ServerMode):
    __slots__ = ()
    def __init__(self, tag_sets=None, max_staleness=-1):
        super(SecondaryPreferred, self).__init__(_SECONDARY_PREFERRED,
                                                 tag_sets,
                                                 max_staleness)
    def __call__(self, selection):
        secondaries = secondary_with_tags_server_selector(
            self.tag_sets,
            max_staleness_selectors.select(
                self.max_staleness, selection))
        if secondaries:
            return secondaries
        else:
            return selection.primary_selection
class Nearest(_ServerMode):
    __slots__ = ()
    def __init__(self, tag_sets=None, max_staleness=-1):
        super(Nearest, self).__init__(_NEAREST, tag_sets, max_staleness)
    def __call__(self, selection):
        return member_with_tags_server_selector(
            self.tag_sets,
            max_staleness_selectors.select(
                self.max_staleness, selection))
_ALL_READ_PREFERENCES = (Primary, PrimaryPreferred,
                         Secondary, SecondaryPreferred, Nearest)
def make_read_preference(mode, tag_sets, max_staleness=-1):
    if mode == _PRIMARY:
        if tag_sets not in (None, [{}]):
            raise ConfigurationError("Read preference primary "
                                     "cannot be combined with tags")
        if max_staleness != -1:
            raise ConfigurationError("Read preference primary cannot be "
                                     "combined with maxStalenessSeconds")
        return Primary()
    return _ALL_READ_PREFERENCES[mode](tag_sets, max_staleness)
_MODES = (
    'PRIMARY',
    'PRIMARY_PREFERRED',
    'SECONDARY',
    'SECONDARY_PREFERRED',
    'NEAREST',
)
class ReadPreference(object):
    PRIMARY = Primary()
    PRIMARY_PREFERRED = PrimaryPreferred()
    SECONDARY = Secondary()
    SECONDARY_PREFERRED = SecondaryPreferred()
    NEAREST = Nearest()
def read_pref_mode_from_name(name):
    return _MONGOS_MODES.index(name)
class MovingAverage(object):
    def __init__(self):
        self.average = None
    def add_sample(self, sample):
        if sample < 0:
            return
        if self.average is None:
            self.average = sample
        else:
            self.average = 0.8 * self.average + 0.2 * sample
    def get(self):
        return self.average
    def reset(self):
        self.average = None
