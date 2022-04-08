import itertools
from .compat import collections_abc
class DirectedGraph(object):
    def __init__(self):
        self._vertices = set()
        self._forwards = {}
        self._backwards = {}
    def __iter__(self):
        return iter(self._vertices)
    def __len__(self):
        return len(self._vertices)
    def __contains__(self, key):
        return key in self._vertices
    def copy(self):
        other = DirectedGraph()
        other._vertices = set(self._vertices)
        other._forwards = {k: set(v) for k, v in self._forwards.items()}
        other._backwards = {k: set(v) for k, v in self._backwards.items()}
        return other
    def add(self, key):
        if key in self._vertices:
            raise ValueError("vertex exists")
        self._vertices.add(key)
        self._forwards[key] = set()
        self._backwards[key] = set()
    def remove(self, key):
        self._vertices.remove(key)
        for f in self._forwards.pop(key):
            self._backwards[f].remove(key)
        for t in self._backwards.pop(key):
            self._forwards[t].remove(key)
    def connected(self, f, t):
        return f in self._backwards[t] and t in self._forwards[f]
    def connect(self, f, t):
        if t not in self._vertices:
            raise KeyError(t)
        self._forwards[f].add(t)
        self._backwards[t].add(f)
    def iter_edges(self):
        for f, children in self._forwards.items():
            for t in children:
                yield f, t
    def iter_children(self, key):
        return iter(self._forwards[key])
    def iter_parents(self, key):
        return iter(self._backwards[key])
class IteratorMapping(collections_abc.Mapping):
    def __init__(self, mapping, accessor, appends=None):
        self._mapping = mapping
        self._accessor = accessor
        self._appends = appends or {}
    def __contains__(self, key):
        return key in self._mapping or key in self._appends
    def __getitem__(self, k):
        try:
            v = self._mapping[k]
        except KeyError:
            return iter(self._appends[k])
        return itertools.chain(self._accessor(v), self._appends.get(k, ()))
    def __iter__(self):
        more = (k for k in self._appends if k not in self._mapping)
        return itertools.chain(self._mapping, more)
    def __len__(self):
        more = len(k for k in self._appends if k not in self._mapping)
        return len(self._mapping) + more
class _FactoryIterableView(object):
    def __init__(self, factory):
        self._factory = factory
    def __repr__(self):
        return "{}({})".format(type(self).__name__, list(self._factory()))
    def __bool__(self):
        try:
            next(self._factory())
        except StopIteration:
            return False
        return True
    __nonzero__ = __bool__
    def __iter__(self):
        return self._factory()
class _SequenceIterableView(object):
    def __init__(self, sequence):
        self._sequence = sequence
    def __repr__(self):
        return "{}({})".format(type(self).__name__, self._sequence)
    def __bool__(self):
        return bool(self._sequence)
    __nonzero__ = __bool__
    def __iter__(self):
        return iter(self._sequence)
def build_iter_view(matches):
    if callable(matches):
        return _FactoryIterableView(matches)
    if not isinstance(matches, collections_abc.Sequence):
        matches = list(matches)
    return _SequenceIterableView(matches)
