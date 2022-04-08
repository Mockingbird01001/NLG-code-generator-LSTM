
import warnings
from collections import deque
from itertools import (
    chain,
    combinations,
    count,
    cycle,
    groupby,
    islice,
    repeat,
    starmap,
    tee,
    zip_longest,
)
import operator
from random import randrange, sample, choice
__all__ = [
    'all_equal',
    'consume',
    'convolve',
    'dotproduct',
    'first_true',
    'flatten',
    'grouper',
    'iter_except',
    'ncycles',
    'nth',
    'nth_combination',
    'padnone',
    'pad_none',
    'pairwise',
    'partition',
    'powerset',
    'prepend',
    'quantify',
    'random_combination_with_replacement',
    'random_combination',
    'random_permutation',
    'random_product',
    'repeatfunc',
    'roundrobin',
    'tabulate',
    'tail',
    'take',
    'unique_everseen',
    'unique_justseen',
]
def take(n, iterable):
    return list(islice(iterable, n))
def tabulate(function, start=0):
    return map(function, count(start))
def tail(n, iterable):
    return iter(deque(iterable, maxlen=n))
def consume(iterator, n=None):
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)
def nth(iterable, n, default=None):
    return next(islice(iterable, n, None), default)
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
def quantify(iterable, pred=bool):
    return sum(map(pred, iterable))
def pad_none(iterable):
    return chain(iterable, repeat(None))
padnone = pad_none
def ncycles(iterable, n):
    return chain.from_iterable(repeat(tuple(iterable), n))
def dotproduct(vec1, vec2):
    return sum(map(operator.mul, vec1, vec2))
def flatten(listOfLists):
    return chain.from_iterable(listOfLists)
def repeatfunc(func, times=None, *args):
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))
def _pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    yield from zip(a, b)
try:
    from itertools import pairwise as itertools_pairwise
except ImportError:
    pairwise = _pairwise
else:
    def pairwise(iterable):
        yield from itertools_pairwise(iterable)
    pairwise.__doc__ = _pairwise.__doc__
def grouper(iterable, n, fillvalue=None):
    if isinstance(iterable, int):
        warnings.warn(
            "grouper expects iterable as first parameter", DeprecationWarning
        )
        n, iterable = iterable, n
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
def roundrobin(*iterables):
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))
def partition(pred, iterable):
    if pred is None:
        pred = bool
    evaluations = ((pred(x), x) for x in iterable)
    t1, t2 = tee(evaluations)
    return (
        (x for (cond, x) in t1 if not cond),
        (x for (cond, x) in t2 if cond),
    )
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
def unique_everseen(iterable, key=None):
    seenset = set()
    seenset_add = seenset.add
    seenlist = []
    seenlist_add = seenlist.append
    use_key = key is not None
    for element in iterable:
        k = key(element) if use_key else element
        try:
            if k not in seenset:
                seenset_add(k)
                yield element
        except TypeError:
            if k not in seenlist:
                seenlist_add(k)
                yield element
def unique_justseen(iterable, key=None):
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))
def iter_except(func, exception, first=None):
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass
def first_true(iterable, default=None, pred=None):
    return next(filter(pred, iterable), default)
def random_product(*args, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(choice(pool) for pool in pools)
def random_permutation(iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(sample(pool, r))
def random_combination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return tuple(pool[i] for i in indices)
def random_combination_with_replacement(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)
def nth_combination(iterable, r, index):
    pool = tuple(iterable)
    n = len(pool)
    if (r < 0) or (r > n):
        raise ValueError
    c = 1
    k = min(r, n - r)
    for i in range(1, k + 1):
        c = c * (n - k + i) // i
    if index < 0:
        index += c
    if (index < 0) or (index >= c):
        raise IndexError
    result = []
    while r:
        c, n, r = c * r // n, n - 1, r - 1
        while index >= c:
            index -= c
            c, n = c * (n - r) // n, n - 1
        result.append(pool[-1 - n])
    return tuple(result)
def prepend(value, iterator):
    return chain([value], iterator)
def convolve(signal, kernel):
    kernel = tuple(kernel)[::-1]
    n = len(kernel)
    window = deque([0], maxlen=n) * n
    for x in chain(signal, repeat(0, n - 1)):
        window.append(x)
        yield sum(map(operator.mul, kernel, window))
