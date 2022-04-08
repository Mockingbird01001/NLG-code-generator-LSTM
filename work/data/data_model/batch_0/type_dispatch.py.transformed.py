
import collections
from typing import Optional, Iterable
from tensorflow.python.types import trace
_MAX_DISPATCH_CACHE = 1024
class TypeDispatchTable:
  """Type dispatch table implementation.
  A type dispatch table is a list, L, of target types. Given a request type, R,
  the table selects a target type, T, according to the following dispatch rules:
    1. R == T or R is subtype of T
    2. There does not exist O in L such that R is subtype of O and O is a
       subtype of T (in other words, T is the closest to R, within list L).
    3. If the above two rules are satisfied by multiple targets, the earliest
       inserted one is chosen.
  """
  def __init__(self):
    self._dispatch_table = collections.OrderedDict()
    self._dispatch_cache = collections.OrderedDict()
  def add_target(self, target: trace.TraceType) -> None:
    self._dispatch_table[target] = None
    for request in self._dispatch_cache:
      if target.is_subtype_of(self._dispatch_cache[request]):
        self._dispatch_cache[request] = target
  @property
  def targets(self) -> Iterable[trace.TraceType]:
    return self._dispatch_table.keys()
  def delete(self, target: trace.TraceType) -> None:
    if target in self._dispatch_table:
      del self._dispatch_table[target]
      for request in list(self._dispatch_cache.keys()):
        if self._dispatch_cache[request] == target:
          del self._dispatch_cache[request]
  def clear(self) -> None:
    self._dispatch_table.clear()
    self._dispatch_cache.clear()
  def dispatch(self, request: trace.TraceType) -> Optional[trace.TraceType]:
    if request in self._dispatch_table:
      return request
    if request in self._dispatch_cache:
      result = self._dispatch_cache.pop(request)
      self._dispatch_cache[request] = result
      return result
    most_specific_subtype = None
    for other in self._dispatch_table:
      if request.is_subtype_of(other):
        if most_specific_subtype is None or other.is_subtype_of(
            most_specific_subtype):
          most_specific_subtype = other
    self._cache_dispatch(request, most_specific_subtype)
    return most_specific_subtype
  def _cache_dispatch(self, request, target):
    if target is not None:
      if len(self._dispatch_cache) > _MAX_DISPATCH_CACHE:
        self._dispatch_cache.popitem(last=False)
      self._dispatch_cache[request] = target
  def try_generalizing_trace_type(self,
                                  target: trace.TraceType) -> trace.TraceType:
    relaxed = target
    for other in self._dispatch_table:
      supertype = relaxed.most_specific_common_supertype([other])
      if supertype is not None:
        relaxed = supertype
    return relaxed
