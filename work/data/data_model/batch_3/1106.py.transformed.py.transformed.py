
import calendar
import datetime
from bson.py3compat import integer_types
from bson.tz_util import utc
UPPERBOUND = 4294967296
class Timestamp(object):
    _type_marker = 17
    def __init__(self, time, inc):
        if isinstance(time, datetime.datetime):
            if time.utcoffset() is not None:
                time = time - time.utcoffset()
            time = int(calendar.timegm(time.timetuple()))
        if not isinstance(time, integer_types):
            raise TypeError("time must be an instance of int")
        if not isinstance(inc, integer_types):
            raise TypeError("inc must be an instance of int")
        if not 0 <= time < UPPERBOUND:
            raise ValueError("time must be contained in [0, 2**32)")
        if not 0 <= inc < UPPERBOUND:
            raise ValueError("inc must be contained in [0, 2**32)")
        self.__time = time
        self.__inc = inc
    @property
    def time(self):
        return self.__time
    @property
    def inc(self):
        return self.__inc
    def __eq__(self, other):
        if isinstance(other, Timestamp):
            return (self.__time == other.time and self.__inc == other.inc)
        else:
            return NotImplemented
    def __hash__(self):
        return hash(self.time) ^ hash(self.inc)
    def __ne__(self, other):
        return not self == other
    def __lt__(self, other):
        if isinstance(other, Timestamp):
            return (self.time, self.inc) < (other.time, other.inc)
        return NotImplemented
    def __le__(self, other):
        if isinstance(other, Timestamp):
            return (self.time, self.inc) <= (other.time, other.inc)
        return NotImplemented
    def __gt__(self, other):
        if isinstance(other, Timestamp):
            return (self.time, self.inc) > (other.time, other.inc)
        return NotImplemented
    def __ge__(self, other):
        if isinstance(other, Timestamp):
            return (self.time, self.inc) >= (other.time, other.inc)
        return NotImplemented
    def __repr__(self):
        return "Timestamp(%s, %s)" % (self.__time, self.__inc)
    def as_datetime(self):
        return datetime.datetime.fromtimestamp(self.__time, utc)
