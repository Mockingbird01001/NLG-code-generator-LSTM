import six
from mongoengine.errors import OperationError
from mongoengine.queryset.base import (BaseQuerySet, CASCADE, DENY, DO_NOTHING,
                                       NULLIFY, PULL)
__all__ = ('QuerySet', 'QuerySetNoCache', 'DO_NOTHING', 'NULLIFY', 'CASCADE',
           'DENY', 'PULL')
REPR_OUTPUT_SIZE = 20
ITER_CHUNK_SIZE = 100
class QuerySet(BaseQuerySet):
    _has_more = True
    _len = None
    _result_cache = None
    def __iter__(self):
        self._iter = True
        if self._has_more:
            return self._iter_results()
        return iter(self._result_cache)
    def __len__(self):
        if self._len is not None:
            return self._len
        if self._has_more:
            list(self._iter_results())
        self._len = len(self._result_cache)
        return self._len
    def __repr__(self):
        if self._iter:
            return '.. queryset mid-iteration ..'
        self._populate_cache()
        data = self._result_cache[:REPR_OUTPUT_SIZE + 1]
        if len(data) > REPR_OUTPUT_SIZE:
            data[-1] = '...(remaining elements truncated)...'
        return repr(data)
    def _iter_results(self):
        if self._result_cache is None:
            self._result_cache = []
        pos = 0
        while True:
            while pos < len(self._result_cache):
                yield self._result_cache[pos]
                pos += 1
            if not self._has_more:
                raise StopIteration
            if len(self._result_cache) <= pos:
                self._populate_cache()
    def _populate_cache(self):
        if self._result_cache is None:
            self._result_cache = []
        if not self._has_more:
            return
        try:
            for _ in six.moves.range(ITER_CHUNK_SIZE):
                self._result_cache.append(self.next())
        except StopIteration:
            self._has_more = False
    def count(self, with_limit_and_skip=False):
        if with_limit_and_skip is False:
            return super(QuerySet, self).count(with_limit_and_skip)
        if self._len is None:
            self._len = super(QuerySet, self).count(with_limit_and_skip)
        return self._len
    def no_cache(self):
        if self._result_cache is not None:
            raise OperationError('QuerySet already cached')
        return self._clone_into(QuerySetNoCache(self._document,
                                                self._collection))
class QuerySetNoCache(BaseQuerySet):
    def cache(self):
        return self._clone_into(QuerySet(self._document, self._collection))
    def __repr__(self):
        if self._iter:
            return '.. queryset mid-iteration ..'
        data = []
        for _ in six.moves.range(REPR_OUTPUT_SIZE + 1):
            try:
                data.append(self.next())
            except StopIteration:
                break
        if len(data) > REPR_OUTPUT_SIZE:
            data[-1] = '...(remaining elements truncated)...'
        self.rewind()
        return repr(data)
    def __iter__(self):
        queryset = self
        if queryset._iter:
            queryset = self.clone()
        queryset.rewind()
        return queryset
class QuerySetNoDeRef(QuerySet):
    def __dereference(items, max_depth=1, instance=None, name=None):
        return items
