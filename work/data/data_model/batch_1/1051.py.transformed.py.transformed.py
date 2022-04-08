
from __future__ import absolute_import, division, print_function
class InfinityType(object):
    def __repr__(self):
        return "Infinity"
    def __hash__(self):
        return hash(repr(self))
    def __lt__(self, other):
        return False
    def __le__(self, other):
        return False
    def __eq__(self, other):
        return isinstance(other, self.__class__)
    def __ne__(self, other):
        return not isinstance(other, self.__class__)
    def __gt__(self, other):
        return True
    def __ge__(self, other):
        return True
    def __neg__(self):
        return NegativeInfinity
Infinity = InfinityType()
class NegativeInfinityType(object):
    def __repr__(self):
        return "-Infinity"
    def __hash__(self):
        return hash(repr(self))
    def __lt__(self, other):
        return True
    def __le__(self, other):
        return True
    def __eq__(self, other):
        return isinstance(other, self.__class__)
    def __ne__(self, other):
        return not isinstance(other, self.__class__)
    def __gt__(self, other):
        return False
    def __ge__(self, other):
        return False
    def __neg__(self):
        return Infinity
NegativeInfinity = NegativeInfinityType()
