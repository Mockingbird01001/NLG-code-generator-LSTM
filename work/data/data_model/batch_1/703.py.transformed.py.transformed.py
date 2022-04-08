import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce, wraps
from heapq import merge, heapify, heapreplace, heappop
from itertools import (
    chain,
    compress,
    count,
    cycle,
    dropwhile,
    groupby,
    islice,
    repeat,
    starmap,
    takewhile,
    tee,
    zip_longest,
)
from math import exp, factorial, floor, log
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
    consume,
    flatten,
    pairwise,
    powerset,
    take,
    unique_everseen,
)
__all__ = [
    'AbortThread',
    'adjacent',
    'always_iterable',
    'always_reversible',
    'bucket',
    'callback_iter',
    'chunked',
    'circular_shifts',
    'collapse',
    'collate',
    'consecutive_groups',
    'consumer',
    'countable',
    'count_cycle',
    'mark_ends',
    'difference',
    'distinct_combinations',
    'distinct_permutations',
    'distribute',
    'divide',
    'exactly_n',
    'filter_except',
    'first',
    'groupby_transform',
    'ilen',
    'interleave_longest',
    'interleave',
    'intersperse',
    'islice_extended',
    'iterate',
    'ichunked',
    'is_sorted',
    'last',
    'locate',
    'lstrip',
    'make_decorator',
    'map_except',
    'map_reduce',
    'nth_or_last',
    'nth_permutation',
    'nth_product',
    'numeric_range',
    'one',
    'only',
    'padded',
    'partitions',
    'set_partitions',
    'peekable',
    'repeat_last',
    'replace',
    'rlocate',
    'rstrip',
    'run_length',
    'sample',
    'seekable',
    'SequenceView',
    'side_effect',
    'sliced',
    'sort_together',
    'split_at',
    'split_after',
    'split_before',
    'split_when',
    'split_into',
    'spy',
    'stagger',
    'strip',
    'substrings',
    'substrings_indexes',
    'time_limited',
    'unique_to_each',
    'unzip',
    'windowed',
    'with_iter',
    'UnequalIterablesError',
    'zip_equal',
    'zip_offset',
    'windowed_complete',
    'all_unique',
    'value_chain',
    'product_index',
    'combination_index',
    'permutation_index',
]
_marker = object()
def chunked(iterable, n, strict=False):
    iterator = iter(partial(take, n, iter(iterable)), [])
    if strict:
        def ret():
            for chunk in iterator:
                if len(chunk) != n:
                    raise ValueError('iterable is not divisible by n.')
                yield chunk
        return iter(ret())
    else:
        return iterator
def first(iterable, default=_marker):
    try:
        return next(iter(iterable))
    except StopIteration as e:
        if default is _marker:
            raise ValueError(
                'first() was called on an empty iterable, and no '
                'default value was provided.'
            ) from e
        return default
def last(iterable, default=_marker):
    try:
        if isinstance(iterable, Sequence):
            return iterable[-1]
        elif hasattr(iterable, '__reversed__') and (hexversion != 0x030800F0):
            return next(reversed(iterable))
        else:
            return deque(iterable, maxlen=1)[-1]
    except (IndexError, TypeError, StopIteration):
        if default is _marker:
            raise ValueError(
                'last() was called on an empty iterable, and no default was '
                'provided.'
            )
        return default
def nth_or_last(iterable, n, default=_marker):
    return last(islice(iterable, n + 1), default=default)
class peekable:
    def __init__(self, iterable):
        self._it = iter(iterable)
        self._cache = deque()
    def __iter__(self):
        return self
    def __bool__(self):
        try:
            self.peek()
        except StopIteration:
            return False
        return True
    def peek(self, default=_marker):
        if not self._cache:
            try:
                self._cache.append(next(self._it))
            except StopIteration:
                if default is _marker:
                    raise
                return default
        return self._cache[0]
    def prepend(self, *items):
        self._cache.extendleft(reversed(items))
    def __next__(self):
        if self._cache:
            return self._cache.popleft()
        return next(self._it)
    def _get_slice(self, index):
        step = 1 if (index.step is None) else index.step
        if step > 0:
            start = 0 if (index.start is None) else index.start
            stop = maxsize if (index.stop is None) else index.stop
        elif step < 0:
            start = -1 if (index.start is None) else index.start
            stop = (-maxsize - 1) if (index.stop is None) else index.stop
        else:
            raise ValueError('slice step cannot be zero')
        if (start < 0) or (stop < 0):
            self._cache.extend(self._it)
        else:
            n = min(max(start, stop) + 1, maxsize)
            cache_len = len(self._cache)
            if n >= cache_len:
                self._cache.extend(islice(self._it, n - cache_len))
        return list(self._cache)[index]
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._get_slice(index)
        cache_len = len(self._cache)
        if index < 0:
            self._cache.extend(self._it)
        elif index >= cache_len:
            self._cache.extend(islice(self._it, index + 1 - cache_len))
        return self._cache[index]
def collate(*iterables, **kwargs):
    warnings.warn(
        "collate is no longer part of more_itertools, use heapq.merge",
        DeprecationWarning,
    )
    return merge(*iterables, **kwargs)
def consumer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return wrapper
def ilen(iterable):
    counter = count()
    deque(zip(iterable, counter), maxlen=0)
    return next(counter)
def iterate(func, start):
    while True:
        yield start
        start = func(start)
def with_iter(context_manager):
    with context_manager as iterable:
        yield from iterable
def one(iterable, too_short=None, too_long=None):
    it = iter(iterable)
    try:
        first_value = next(it)
    except StopIteration as e:
        raise (
            too_short or ValueError('too few items in iterable (expected 1)')
        ) from e
    try:
        second_value = next(it)
    except StopIteration:
        pass
    else:
        msg = (
            'Expected exactly one item in iterable, but got {!r}, {!r}, '
            'and perhaps more.'.format(first_value, second_value)
        )
        raise too_long or ValueError(msg)
    return first_value
def distinct_permutations(iterable, r=None):
    def _full(A):
        while True:
            yield tuple(A)
            for i in range(size - 2, -1, -1):
                if A[i] < A[i + 1]:
                    break
            else:
                return
            for j in range(size - 1, i, -1):
                if A[i] < A[j]:
                    break
            A[i], A[j] = A[j], A[i]
            A[i + 1 :] = A[: i - size : -1]
    def _partial(A, r):
        head, tail = A[:r], A[r:]
        right_head_indexes = range(r - 1, -1, -1)
        left_tail_indexes = range(len(tail))
        while True:
            yield tuple(head)
            pivot = tail[-1]
            for i in right_head_indexes:
                if head[i] < pivot:
                    break
                pivot = head[i]
            else:
                return
            for j in left_tail_indexes:
                if tail[j] > head[i]:
                    head[i], tail[j] = tail[j], head[i]
                    break
            else:
                for j in right_head_indexes:
                    if head[j] > head[i]:
                        head[i], head[j] = head[j], head[i]
                        break
            tail += head[: i - r : -1]
            i += 1
            head[i:], tail[:] = tail[: r - i], tail[r - i :]
    items = sorted(iterable)
    size = len(items)
    if r is None:
        r = size
    if 0 < r <= size:
        return _full(items) if (r == size) else _partial(items, r)
    return iter(() if r else ((),))
def intersperse(e, iterable, n=1):
    if n == 0:
        raise ValueError('n must be > 0')
    elif n == 1:
        return islice(interleave(repeat(e), iterable), 1, None)
    else:
        filler = repeat([e])
        chunks = chunked(iterable, n)
        return flatten(islice(interleave(filler, chunks), 1, None))
def unique_to_each(*iterables):
    pool = [list(it) for it in iterables]
    counts = Counter(chain.from_iterable(map(set, pool)))
    uniques = {element for element in counts if counts[element] == 1}
    return [list(filter(uniques.__contains__, it)) for it in pool]
def windowed(seq, n, fillvalue=None, step=1):
    if n < 0:
        raise ValueError('n must be >= 0')
    if n == 0:
        yield tuple()
        return
    if step < 1:
        raise ValueError('step must be >= 1')
    window = deque(maxlen=n)
    i = n
    for _ in map(window.append, seq):
        i -= 1
        if not i:
            i = step
            yield tuple(window)
    size = len(window)
    if size < n:
        yield tuple(chain(window, repeat(fillvalue, n - size)))
    elif 0 < i < min(step, n):
        window += (fillvalue,) * i
        yield tuple(window)
def substrings(iterable):
    seq = []
    for item in iter(iterable):
        seq.append(item)
        yield (item,)
    seq = tuple(seq)
    item_count = len(seq)
    for n in range(2, item_count + 1):
        for i in range(item_count - n + 1):
            yield seq[i : i + n]
def substrings_indexes(seq, reverse=False):
    r = range(1, len(seq) + 1)
    if reverse:
        r = reversed(r)
    return (
        (seq[i : i + L], i, i + L) for L in r for i in range(len(seq) - L + 1)
    )
class bucket:
    def __init__(self, iterable, key, validator=None):
        self._it = iter(iterable)
        self._key = key
        self._cache = defaultdict(deque)
        self._validator = validator or (lambda x: True)
    def __contains__(self, value):
        if not self._validator(value):
            return False
        try:
            item = next(self[value])
        except StopIteration:
            return False
        else:
            self._cache[value].appendleft(item)
        return True
    def _get_values(self, value):
        while True:
            if self._cache[value]:
                yield self._cache[value].popleft()
            else:
                while True:
                    try:
                        item = next(self._it)
                    except StopIteration:
                        return
                    item_value = self._key(item)
                    if item_value == value:
                        yield item
                        break
                    elif self._validator(item_value):
                        self._cache[item_value].append(item)
    def __iter__(self):
        for item in self._it:
            item_value = self._key(item)
            if self._validator(item_value):
                self._cache[item_value].append(item)
        yield from self._cache.keys()
    def __getitem__(self, value):
        if not self._validator(value):
            return iter(())
        return self._get_values(value)
def spy(iterable, n=1):
    it = iter(iterable)
    head = take(n, it)
    return head.copy(), chain(head, it)
def interleave(*iterables):
    return chain.from_iterable(zip(*iterables))
def interleave_longest(*iterables):
    i = chain.from_iterable(zip_longest(*iterables, fillvalue=_marker))
    return (x for x in i if x is not _marker)
def collapse(iterable, base_type=None, levels=None):
    def walk(node, level):
        if (
            ((levels is not None) and (level > levels))
            or isinstance(node, (str, bytes))
            or ((base_type is not None) and isinstance(node, base_type))
        ):
            yield node
            return
        try:
            tree = iter(node)
        except TypeError:
            yield node
            return
        else:
            for child in tree:
                yield from walk(child, level + 1)
    yield from walk(iterable, 0)
def side_effect(func, iterable, chunk_size=None, before=None, after=None):
    try:
        if before is not None:
            before()
        if chunk_size is None:
            for item in iterable:
                func(item)
                yield item
        else:
            for chunk in chunked(iterable, chunk_size):
                func(chunk)
                yield from chunk
    finally:
        if after is not None:
            after()
def sliced(seq, n, strict=False):
    iterator = takewhile(len, (seq[i : i + n] for i in count(0, n)))
    if strict:
        def ret():
            for _slice in iterator:
                if len(_slice) != n:
                    raise ValueError("seq is not divisible by n.")
                yield _slice
        return iter(ret())
    else:
        return iterator
def split_at(iterable, pred, maxsplit=-1, keep_separator=False):
    if maxsplit == 0:
        yield list(iterable)
        return
    buf = []
    it = iter(iterable)
    for item in it:
        if pred(item):
            yield buf
            if keep_separator:
                yield [item]
            if maxsplit == 1:
                yield list(it)
                return
            buf = []
            maxsplit -= 1
        else:
            buf.append(item)
    yield buf
def split_before(iterable, pred, maxsplit=-1):
    if maxsplit == 0:
        yield list(iterable)
        return
    buf = []
    it = iter(iterable)
    for item in it:
        if pred(item) and buf:
            yield buf
            if maxsplit == 1:
                yield [item] + list(it)
                return
            buf = []
            maxsplit -= 1
        buf.append(item)
    if buf:
        yield buf
def split_after(iterable, pred, maxsplit=-1):
    if maxsplit == 0:
        yield list(iterable)
        return
    buf = []
    it = iter(iterable)
    for item in it:
        buf.append(item)
        if pred(item) and buf:
            yield buf
            if maxsplit == 1:
                yield list(it)
                return
            buf = []
            maxsplit -= 1
    if buf:
        yield buf
def split_when(iterable, pred, maxsplit=-1):
    if maxsplit == 0:
        yield list(iterable)
        return
    it = iter(iterable)
    try:
        cur_item = next(it)
    except StopIteration:
        return
    buf = [cur_item]
    for next_item in it:
        if pred(cur_item, next_item):
            yield buf
            if maxsplit == 1:
                yield [next_item] + list(it)
                return
            buf = []
            maxsplit -= 1
        buf.append(next_item)
        cur_item = next_item
    yield buf
def split_into(iterable, sizes):
    it = iter(iterable)
    for size in sizes:
        if size is None:
            yield list(it)
            return
        else:
            yield list(islice(it, size))
def padded(iterable, fillvalue=None, n=None, next_multiple=False):
    it = iter(iterable)
    if n is None:
        yield from chain(it, repeat(fillvalue))
    elif n < 1:
        raise ValueError('n must be at least 1')
    else:
        item_count = 0
        for item in it:
            yield item
            item_count += 1
        remaining = (n - item_count) % n if next_multiple else n - item_count
        for _ in range(remaining):
            yield fillvalue
def repeat_last(iterable, default=None):
    item = _marker
    for item in iterable:
        yield item
    final = default if item is _marker else item
    yield from repeat(final)
def distribute(n, iterable):
    if n < 1:
        raise ValueError('n must be at least 1')
    children = tee(iterable, n)
    return [islice(it, index, None, n) for index, it in enumerate(children)]
def stagger(iterable, offsets=(-1, 0, 1), longest=False, fillvalue=None):
    children = tee(iterable, len(offsets))
    return zip_offset(
        *children, offsets=offsets, longest=longest, fillvalue=fillvalue
    )
class UnequalIterablesError(ValueError):
    def __init__(self, details=None):
        msg = 'Iterables have different lengths'
        if details is not None:
            msg += (': index 0 has length {}; index {} has length {}').format(
                *details
            )
        super().__init__(msg)
def _zip_equal_generator(iterables):
    for combo in zip_longest(*iterables, fillvalue=_marker):
        for val in combo:
            if val is _marker:
                raise UnequalIterablesError()
        yield combo
def zip_equal(*iterables):
    if hexversion >= 0x30A00A6:
        warnings.warn(
            (
                'zip_equal will be removed in a future version of '
                'more-itertools. Use the builtin zip function with '
                'strict=True instead.'
            ),
            DeprecationWarning,
        )
    try:
        first_size = len(iterables[0])
        for i, it in enumerate(iterables[1:], 1):
            size = len(it)
            if size != first_size:
                break
        else:
            return zip(*iterables)
        raise UnequalIterablesError(details=(first_size, i, size))
    except TypeError:
        return _zip_equal_generator(iterables)
def zip_offset(*iterables, offsets, longest=False, fillvalue=None):
    if len(iterables) != len(offsets):
        raise ValueError("Number of iterables and offsets didn't match")
    staggered = []
    for it, n in zip(iterables, offsets):
        if n < 0:
            staggered.append(chain(repeat(fillvalue, -n), it))
        elif n > 0:
            staggered.append(islice(it, n, None))
        else:
            staggered.append(it)
    if longest:
        return zip_longest(*staggered, fillvalue=fillvalue)
    return zip(*staggered)
def sort_together(iterables, key_list=(0,), key=None, reverse=False):
    if key is None:
        key_argument = itemgetter(*key_list)
    else:
        key_list = list(key_list)
        if len(key_list) == 1:
            key_offset = key_list[0]
            key_argument = lambda zipped_items: key(zipped_items[key_offset])
        else:
            get_key_items = itemgetter(*key_list)
            key_argument = lambda zipped_items: key(
                *get_key_items(zipped_items)
            )
    return list(
        zip(*sorted(zip(*iterables), key=key_argument, reverse=reverse))
    )
def unzip(iterable):
    head, iterable = spy(iter(iterable))
    if not head:
        return ()
    head = head[0]
    iterables = tee(iterable, len(head))
    def itemgetter(i):
        def getter(obj):
            try:
                return obj[i]
            except IndexError:
                raise StopIteration
        return getter
    return tuple(map(itemgetter(i), it) for i, it in enumerate(iterables))
def divide(n, iterable):
    if n < 1:
        raise ValueError('n must be at least 1')
    try:
        iterable[:0]
    except TypeError:
        seq = tuple(iterable)
    else:
        seq = iterable
    q, r = divmod(len(seq), n)
    ret = []
    stop = 0
    for i in range(1, n + 1):
        start = stop
        stop += q + 1 if i <= r else q
        ret.append(iter(seq[start:stop]))
    return ret
def always_iterable(obj, base_type=(str, bytes)):
    if obj is None:
        return iter(())
    if (base_type is not None) and isinstance(obj, base_type):
        return iter((obj,))
    try:
        return iter(obj)
    except TypeError:
        return iter((obj,))
def adjacent(predicate, iterable, distance=1):
    if distance < 0:
        raise ValueError('distance must be at least 0')
    i1, i2 = tee(iterable)
    padding = [False] * distance
    selected = chain(padding, map(predicate, i1), padding)
    adjacent_to_selected = map(any, windowed(selected, 2 * distance + 1))
    return zip(adjacent_to_selected, i2)
def groupby_transform(iterable, keyfunc=None, valuefunc=None, reducefunc=None):
    ret = groupby(iterable, keyfunc)
    if valuefunc:
        ret = ((k, map(valuefunc, g)) for k, g in ret)
    if reducefunc:
        ret = ((k, reducefunc(g)) for k, g in ret)
    return ret
class numeric_range(abc.Sequence, abc.Hashable):
    _EMPTY_HASH = hash(range(0, 0))
    def __init__(self, *args):
        argc = len(args)
        if argc == 1:
            (self._stop,) = args
            self._start = type(self._stop)(0)
            self._step = type(self._stop - self._start)(1)
        elif argc == 2:
            self._start, self._stop = args
            self._step = type(self._stop - self._start)(1)
        elif argc == 3:
            self._start, self._stop, self._step = args
        elif argc == 0:
            raise TypeError(
                'numeric_range expected at least '
                '1 argument, got {}'.format(argc)
            )
        else:
            raise TypeError(
                'numeric_range expected at most '
                '3 arguments, got {}'.format(argc)
            )
        self._zero = type(self._step)(0)
        if self._step == self._zero:
            raise ValueError('numeric_range() arg 3 must not be zero')
        self._growing = self._step > self._zero
        self._init_len()
    def __bool__(self):
        if self._growing:
            return self._start < self._stop
        else:
            return self._start > self._stop
    def __contains__(self, elem):
        if self._growing:
            if self._start <= elem < self._stop:
                return (elem - self._start) % self._step == self._zero
        else:
            if self._start >= elem > self._stop:
                return (self._start - elem) % (-self._step) == self._zero
        return False
    def __eq__(self, other):
        if isinstance(other, numeric_range):
            empty_self = not bool(self)
            empty_other = not bool(other)
            if empty_self or empty_other:
                return empty_self and empty_other
            else:
                return (
                    self._start == other._start
                    and self._step == other._step
                    and self._get_by_index(-1) == other._get_by_index(-1)
                )
        else:
            return False
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_by_index(key)
        elif isinstance(key, slice):
            step = self._step if key.step is None else key.step * self._step
            if key.start is None or key.start <= -self._len:
                start = self._start
            elif key.start >= self._len:
                start = self._stop
            else:
                start = self._get_by_index(key.start)
            if key.stop is None or key.stop >= self._len:
                stop = self._stop
            elif key.stop <= -self._len:
                stop = self._start
            else:
                stop = self._get_by_index(key.stop)
            return numeric_range(start, stop, step)
        else:
            raise TypeError(
                'numeric range indices must be '
                'integers or slices, not {}'.format(type(key).__name__)
            )
    def __hash__(self):
        if self:
            return hash((self._start, self._get_by_index(-1), self._step))
        else:
            return self._EMPTY_HASH
    def __iter__(self):
        values = (self._start + (n * self._step) for n in count())
        if self._growing:
            return takewhile(partial(gt, self._stop), values)
        else:
            return takewhile(partial(lt, self._stop), values)
    def __len__(self):
        return self._len
    def _init_len(self):
        if self._growing:
            start = self._start
            stop = self._stop
            step = self._step
        else:
            start = self._stop
            stop = self._start
            step = -self._step
        distance = stop - start
        if distance <= self._zero:
            self._len = 0
        else:
            q, r = divmod(distance, step)
            self._len = int(q) + int(r != self._zero)
    def __reduce__(self):
        return numeric_range, (self._start, self._stop, self._step)
    def __repr__(self):
        if self._step == 1:
            return "numeric_range({}, {})".format(
                repr(self._start), repr(self._stop)
            )
        else:
            return "numeric_range({}, {}, {})".format(
                repr(self._start), repr(self._stop), repr(self._step)
            )
    def __reversed__(self):
        return iter(
            numeric_range(
                self._get_by_index(-1), self._start - self._step, -self._step
            )
        )
    def count(self, value):
        return int(value in self)
    def index(self, value):
        if self._growing:
            if self._start <= value < self._stop:
                q, r = divmod(value - self._start, self._step)
                if r == self._zero:
                    return int(q)
        else:
            if self._start >= value > self._stop:
                q, r = divmod(self._start - value, -self._step)
                if r == self._zero:
                    return int(q)
        raise ValueError("{} is not in numeric range".format(value))
    def _get_by_index(self, i):
        if i < 0:
            i += self._len
        if i < 0 or i >= self._len:
            raise IndexError("numeric range object index out of range")
        return self._start + i * self._step
def count_cycle(iterable, n=None):
    iterable = tuple(iterable)
    if not iterable:
        return iter(())
    counter = count() if n is None else range(n)
    return ((i, item) for i in counter for item in iterable)
def mark_ends(iterable):
    it = iter(iterable)
    try:
        b = next(it)
    except StopIteration:
        return
    try:
        for i in count():
            a = b
            b = next(it)
            yield i == 0, False, a
    except StopIteration:
        yield i == 0, True, a
def locate(iterable, pred=bool, window_size=None):
    if window_size is None:
        return compress(count(), map(pred, iterable))
    if window_size < 1:
        raise ValueError('window size must be at least 1')
    it = windowed(iterable, window_size, fillvalue=_marker)
    return compress(count(), starmap(pred, it))
def lstrip(iterable, pred):
    return dropwhile(pred, iterable)
def rstrip(iterable, pred):
    cache = []
    cache_append = cache.append
    cache_clear = cache.clear
    for x in iterable:
        if pred(x):
            cache_append(x)
        else:
            yield from cache
            cache_clear()
            yield x
def strip(iterable, pred):
    return rstrip(lstrip(iterable, pred), pred)
class islice_extended:
    def __init__(self, iterable, *args):
        it = iter(iterable)
        if args:
            self._iterable = _islice_helper(it, slice(*args))
        else:
            self._iterable = it
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._iterable)
    def __getitem__(self, key):
        if isinstance(key, slice):
            return islice_extended(_islice_helper(self._iterable, key))
        raise TypeError('islice_extended.__getitem__ argument must be a slice')
def _islice_helper(it, s):
    start = s.start
    stop = s.stop
    if s.step == 0:
        raise ValueError('step argument must be a non-zero integer or None.')
    step = s.step or 1
    if step > 0:
        start = 0 if (start is None) else start
        if start < 0:
            cache = deque(enumerate(it, 1), maxlen=-start)
            len_iter = cache[-1][0] if cache else 0
            i = max(len_iter + start, 0)
            if stop is None:
                j = len_iter
            elif stop >= 0:
                j = min(stop, len_iter)
            else:
                j = max(len_iter + stop, 0)
            n = j - i
            if n <= 0:
                return
            for index, item in islice(cache, 0, n, step):
                yield item
        elif (stop is not None) and (stop < 0):
            next(islice(it, start, start), None)
            cache = deque(islice(it, -stop), maxlen=-stop)
            for index, item in enumerate(it):
                cached_item = cache.popleft()
                if index % step == 0:
                    yield cached_item
                cache.append(item)
        else:
            yield from islice(it, start, stop, step)
    else:
        start = -1 if (start is None) else start
        if (stop is not None) and (stop < 0):
            n = -stop - 1
            cache = deque(enumerate(it, 1), maxlen=n)
            len_iter = cache[-1][0] if cache else 0
            if start < 0:
                i, j = start, stop
            else:
                i, j = min(start - len_iter, -1), None
            for index, item in list(cache)[i:j:step]:
                yield item
        else:
            if stop is not None:
                m = stop + 1
                next(islice(it, m, m), None)
            if start < 0:
                i = start
                n = None
            elif stop is None:
                i = None
                n = start + 1
            else:
                i = None
                n = start - stop
                if n <= 0:
                    return
            cache = list(islice(it, n))
            yield from cache[i::step]
def always_reversible(iterable):
    try:
        return reversed(iterable)
    except TypeError:
        return reversed(list(iterable))
def consecutive_groups(iterable, ordering=lambda x: x):
    for k, g in groupby(
        enumerate(iterable), key=lambda x: x[0] - ordering(x[1])
    ):
        yield map(itemgetter(1), g)
def difference(iterable, func=sub, *, initial=None):
    a, b = tee(iterable)
    try:
        first = [next(b)]
    except StopIteration:
        return iter([])
    if initial is not None:
        first = []
    return chain(first, starmap(func, zip(b, a)))
class SequenceView(Sequence):
    def __init__(self, target):
        if not isinstance(target, Sequence):
            raise TypeError
        self._target = target
    def __getitem__(self, index):
        return self._target[index]
    def __len__(self):
        return len(self._target)
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self._target))
class seekable:
    def __init__(self, iterable, maxlen=None):
        self._source = iter(iterable)
        if maxlen is None:
            self._cache = []
        else:
            self._cache = deque([], maxlen)
        self._index = None
    def __iter__(self):
        return self
    def __next__(self):
        if self._index is not None:
            try:
                item = self._cache[self._index]
            except IndexError:
                self._index = None
            else:
                self._index += 1
                return item
        item = next(self._source)
        self._cache.append(item)
        return item
    def __bool__(self):
        try:
            self.peek()
        except StopIteration:
            return False
        return True
    def peek(self, default=_marker):
        try:
            peeked = next(self)
        except StopIteration:
            if default is _marker:
                raise
            return default
        if self._index is None:
            self._index = len(self._cache)
        self._index -= 1
        return peeked
    def elements(self):
        return SequenceView(self._cache)
    def seek(self, index):
        self._index = index
        remainder = index - len(self._cache)
        if remainder > 0:
            consume(self, remainder)
class run_length:
    @staticmethod
    def encode(iterable):
        return ((k, ilen(g)) for k, g in groupby(iterable))
    @staticmethod
    def decode(iterable):
        return chain.from_iterable(repeat(k, n) for k, n in iterable)
def exactly_n(iterable, n, predicate=bool):
    return len(take(n + 1, filter(predicate, iterable))) == n
def circular_shifts(iterable):
    lst = list(iterable)
    return take(len(lst), windowed(cycle(lst), len(lst)))
def make_decorator(wrapping_func, result_index=0):
    def decorator(*wrapping_args, **wrapping_kwargs):
        def outer_wrapper(f):
            def inner_wrapper(*args, **kwargs):
                result = f(*args, **kwargs)
                wrapping_args_ = list(wrapping_args)
                wrapping_args_.insert(result_index, result)
                return wrapping_func(*wrapping_args_, **wrapping_kwargs)
            return inner_wrapper
        return outer_wrapper
    return decorator
def map_reduce(iterable, keyfunc, valuefunc=None, reducefunc=None):
    valuefunc = (lambda x: x) if (valuefunc is None) else valuefunc
    ret = defaultdict(list)
    for item in iterable:
        key = keyfunc(item)
        value = valuefunc(item)
        ret[key].append(value)
    if reducefunc is not None:
        for key, value_list in ret.items():
            ret[key] = reducefunc(value_list)
    ret.default_factory = None
    return ret
def rlocate(iterable, pred=bool, window_size=None):
    if window_size is None:
        try:
            len_iter = len(iterable)
            return (len_iter - i - 1 for i in locate(reversed(iterable), pred))
        except TypeError:
            pass
    return reversed(list(locate(iterable, pred, window_size)))
def replace(iterable, pred, substitutes, count=None, window_size=1):
    if window_size < 1:
        raise ValueError('window_size must be at least 1')
    substitutes = tuple(substitutes)
    it = chain(iterable, [_marker] * (window_size - 1))
    windows = windowed(it, window_size)
    n = 0
    for w in windows:
        if pred(*w):
            if (count is None) or (n < count):
                n += 1
                yield from substitutes
                consume(windows, window_size - 1)
                continue
        if w and (w[0] is not _marker):
            yield w[0]
def partitions(iterable):
    sequence = list(iterable)
    n = len(sequence)
    for i in powerset(range(1, n)):
        yield [sequence[i:j] for i, j in zip((0,) + i, i + (n,))]
def set_partitions(iterable, k=None):
    L = list(iterable)
    n = len(L)
    if k is not None:
        if k < 1:
            raise ValueError(
                "Can't partition in a negative or zero number of groups"
            )
        elif k > n:
            return
    def set_partitions_helper(L, k):
        n = len(L)
        if k == 1:
            yield [L]
        elif n == k:
            yield [[s] for s in L]
        else:
            e, *M = L
            for p in set_partitions_helper(M, k - 1):
                yield [[e], *p]
            for p in set_partitions_helper(M, k):
                for i in range(len(p)):
                    yield p[:i] + [[e] + p[i]] + p[i + 1 :]
    if k is None:
        for k in range(1, n + 1):
            yield from set_partitions_helper(L, k)
    else:
        yield from set_partitions_helper(L, k)
class time_limited:
    def __init__(self, limit_seconds, iterable):
        if limit_seconds < 0:
            raise ValueError('limit_seconds must be positive')
        self.limit_seconds = limit_seconds
        self._iterable = iter(iterable)
        self._start_time = monotonic()
        self.timed_out = False
    def __iter__(self):
        return self
    def __next__(self):
        item = next(self._iterable)
        if monotonic() - self._start_time > self.limit_seconds:
            self.timed_out = True
            raise StopIteration
        return item
def only(iterable, default=None, too_long=None):
    it = iter(iterable)
    first_value = next(it, default)
    try:
        second_value = next(it)
    except StopIteration:
        pass
    else:
        msg = (
            'Expected exactly one item in iterable, but got {!r}, {!r}, '
            'and perhaps more.'.format(first_value, second_value)
        )
        raise too_long or ValueError(msg)
    return first_value
def ichunked(iterable, n):
    source = iter(iterable)
    while True:
        item = next(source, _marker)
        if item is _marker:
            return
        source, it = tee(chain([item], source))
        yield islice(it, n)
        consume(source, n)
def distinct_combinations(iterable, r):
    if r < 0:
        raise ValueError('r must be non-negative')
    elif r == 0:
        yield ()
        return
    pool = tuple(iterable)
    generators = [unique_everseen(enumerate(pool), key=itemgetter(1))]
    current_combo = [None] * r
    level = 0
    while generators:
        try:
            cur_idx, p = next(generators[-1])
        except StopIteration:
            generators.pop()
            level -= 1
            continue
        current_combo[level] = p
        if level + 1 == r:
            yield tuple(current_combo)
        else:
            generators.append(
                unique_everseen(
                    enumerate(pool[cur_idx + 1 :], cur_idx + 1),
                    key=itemgetter(1),
                )
            )
            level += 1
def filter_except(validator, iterable, *exceptions):
    for item in iterable:
        try:
            validator(item)
        except exceptions:
            pass
        else:
            yield item
def map_except(function, iterable, *exceptions):
    for item in iterable:
        try:
            yield function(item)
        except exceptions:
            pass
def _sample_unweighted(iterable, k):
    reservoir = take(k, iterable)
    W = exp(log(random()) / k)
    next_index = k + floor(log(random()) / log(1 - W))
    for index, element in enumerate(iterable, k):
        if index == next_index:
            reservoir[randrange(k)] = element
            W *= exp(log(random()) / k)
            next_index += floor(log(random()) / log(1 - W)) + 1
    return reservoir
def _sample_weighted(iterable, k, weights):
    weight_keys = (log(random()) / weight for weight in weights)
    reservoir = take(k, zip(weight_keys, iterable))
    heapify(reservoir)
    smallest_weight_key, _ = reservoir[0]
    weights_to_skip = log(random()) / smallest_weight_key
    for weight, element in zip(weights, iterable):
        if weight >= weights_to_skip:
            smallest_weight_key, _ = reservoir[0]
            t_w = exp(weight * smallest_weight_key)
            r_2 = uniform(t_w, 1)
            weight_key = log(r_2) / weight
            heapreplace(reservoir, (weight_key, element))
            smallest_weight_key, _ = reservoir[0]
            weights_to_skip = log(random()) / smallest_weight_key
        else:
            weights_to_skip -= weight
    return [heappop(reservoir)[1] for _ in range(k)]
def sample(iterable, k, weights=None):
    if k == 0:
        return []
    iterable = iter(iterable)
    if weights is None:
        return _sample_unweighted(iterable, k)
    else:
        weights = iter(weights)
        return _sample_weighted(iterable, k, weights)
def is_sorted(iterable, key=None, reverse=False):
    compare = lt if reverse else gt
    it = iterable if (key is None) else map(key, iterable)
    return not any(starmap(compare, pairwise(it)))
class AbortThread(BaseException):
    pass
class callback_iter:
    def __init__(self, func, callback_kwd='callback', wait_seconds=0.1):
        self._func = func
        self._callback_kwd = callback_kwd
        self._aborted = False
        self._future = None
        self._wait_seconds = wait_seconds
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._iterator = self._reader()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self._aborted = True
        self._executor.shutdown()
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._iterator)
    @property
    def done(self):
        if self._future is None:
            return False
        return self._future.done()
    @property
    def result(self):
        if not self.done:
            raise RuntimeError('Function has not yet completed')
        return self._future.result()
    def _reader(self):
        q = Queue()
        def callback(*args, **kwargs):
            if self._aborted:
                raise AbortThread('canceled by user')
            q.put((args, kwargs))
        self._future = self._executor.submit(
            self._func, **{self._callback_kwd: callback}
        )
        while True:
            try:
                item = q.get(timeout=self._wait_seconds)
            except Empty:
                pass
            else:
                q.task_done()
                yield item
            if self._future.done():
                break
        remaining = []
        while True:
            try:
                item = q.get_nowait()
            except Empty:
                break
            else:
                q.task_done()
                remaining.append(item)
        q.join()
        yield from remaining
def windowed_complete(iterable, n):
    if n < 0:
        raise ValueError('n must be >= 0')
    seq = tuple(iterable)
    size = len(seq)
    if n > size:
        raise ValueError('n must be <= len(seq)')
    for i in range(size - n + 1):
        beginning = seq[:i]
        middle = seq[i : i + n]
        end = seq[i + n :]
        yield beginning, middle, end
def all_unique(iterable, key=None):
    seenset = set()
    seenset_add = seenset.add
    seenlist = []
    seenlist_add = seenlist.append
    for element in map(key, iterable) if key else iterable:
        try:
            if element in seenset:
                return False
            seenset_add(element)
        except TypeError:
            if element in seenlist:
                return False
            seenlist_add(element)
    return True
def nth_product(index, *args):
    pools = list(map(tuple, reversed(args)))
    ns = list(map(len, pools))
    c = reduce(mul, ns)
    if index < 0:
        index += c
    if not 0 <= index < c:
        raise IndexError
    result = []
    for pool, n in zip(pools, ns):
        result.append(pool[index % n])
        index //= n
    return tuple(reversed(result))
def nth_permutation(iterable, r, index):
    pool = list(iterable)
    n = len(pool)
    if r is None or r == n:
        r, c = n, factorial(n)
    elif not 0 <= r < n:
        raise ValueError
    else:
        c = factorial(n) // factorial(n - r)
    if index < 0:
        index += c
    if not 0 <= index < c:
        raise IndexError
    if c == 0:
        return tuple()
    result = [0] * r
    q = index * factorial(n) // c if r < n else index
    for d in range(1, n + 1):
        q, i = divmod(q, d)
        if 0 <= n - d < r:
            result[n - d] = i
        if q == 0:
            break
    return tuple(map(pool.pop, result))
def value_chain(*args):
    for value in args:
        if isinstance(value, (str, bytes)):
            yield value
            continue
        try:
            yield from value
        except TypeError:
            yield value
def product_index(element, *args):
    index = 0
    for x, pool in zip_longest(element, args, fillvalue=_marker):
        if x is _marker or pool is _marker:
            raise ValueError('element is not a product of args')
        pool = tuple(pool)
        index = index * len(pool) + pool.index(x)
    return index
def combination_index(element, iterable):
    element = enumerate(element)
    k, y = next(element, (None, None))
    if k is None:
        return 0
    indexes = []
    pool = enumerate(iterable)
    for n, x in pool:
        if x == y:
            indexes.append(n)
            tmp, y = next(element, (None, None))
            if tmp is None:
                break
            else:
                k = tmp
    else:
        raise ValueError('element is not a combination of iterable')
    n, _ = last(pool, default=(n, None))
    index = 1
    for i, j in enumerate(reversed(indexes), start=1):
        j = n - j
        if i <= j:
            index += factorial(j) // (factorial(i) * factorial(j - i))
    return factorial(n + 1) // (factorial(k + 1) * factorial(n - k)) - index
def permutation_index(element, iterable):
    index = 0
    pool = list(iterable)
    for i, x in zip(range(len(pool), -1, -1), element):
        r = pool.index(x)
        index = index * i + r
        del pool[r]
    return index
class countable:
    def __init__(self, iterable):
        self._it = iter(iterable)
        self.items_seen = 0
    def __iter__(self):
        return self
    def __next__(self):
        item = next(self._it)
        self.items_seen += 1
        return item
