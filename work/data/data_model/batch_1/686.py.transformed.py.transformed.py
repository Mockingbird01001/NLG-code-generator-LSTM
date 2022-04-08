
class MinKey(object):
    _type_marker = 255
    def __eq__(self, other):
        return isinstance(other, MinKey)
    def __hash__(self):
        return hash(self._type_marker)
    def __ne__(self, other):
        return not self == other
    def __le__(self, dummy):
        return True
    def __lt__(self, other):
        return not isinstance(other, MinKey)
    def __ge__(self, other):
        return isinstance(other, MinKey)
    def __gt__(self, dummy):
        return False
    def __repr__(self):
        return "MinKey()"
