
from __future__ import division, absolute_import, print_function
from itertools import chain, starmap
import itertools
from past.types import basestring
from past.utils import PY3
def flatmap(f, items):
    return chain.from_iterable(map(f, items))
if PY3:
    import builtins
    def oldfilter(*args):
        mytype = type(args[1])
        if isinstance(args[1], basestring):
            return mytype().join(builtins.filter(*args))
        elif isinstance(args[1], (tuple, list)):
            return mytype(builtins.filter(*args))
        else:
            return list(builtins.filter(*args))
    def oldmap(func, *iterables):
        zipped = itertools.zip_longest(*iterables)
        l = list(zipped)
        if len(l) == 0:
            return []
        if func is None:
            result = l
        else:
            result = list(starmap(func, l))
        try:
            if max([len(item) for item in result]) == 1:
                return list(chain.from_iterable(result))
        except TypeError as e:
            pass
        return result
    def oldrange(*args, **kwargs):
        return list(builtins.range(*args, **kwargs))
    def oldzip(*args, **kwargs):
        return list(builtins.zip(*args, **kwargs))
    filter = oldfilter
    map = oldmap
    range = oldrange
    from functools import reduce
    zip = oldzip
    __all__ = ['filter', 'map', 'range', 'reduce', 'zip']
else:
    import __builtin__
    filter = __builtin__.filter
    map = __builtin__.map
    range = __builtin__.range
    reduce = __builtin__.reduce
    zip = __builtin__.zip
    __all__ = []
