
class MaxKey(object):
    _type_marker = 127
    def __eq__(self, other):
        return isinstance(other, MaxKey)
    def __hash__(self):
        return hash(self._type_marker)
    def __ne__(self, other):
        return not self == other
    def __le__(self, other):
        return isinstance(other, MaxKey)
    def __lt__(self, dummy):
        return False
    def __ge__(self, dummy):
        return True
    def __gt__(self, other):
        return not isinstance(other, MaxKey)
    def __repr__(self):
        return "MaxKey()"
