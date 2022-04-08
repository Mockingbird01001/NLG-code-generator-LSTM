
import collections
from typing import Optional, Sequence, Any, NamedTuple
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.types import trace
DELETE_WITH_WEAKREF = False
class FunctionContext(NamedTuple):
  context: Any
class FunctionCacheKey(trace.TraceType):
  def __init__(self, function_signature: trace.TraceType,
               call_context: FunctionContext):
    self.function_signature = function_signature
    self.call_context = call_context
  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not isinstance(other, FunctionCacheKey):
      return False
    if self.call_context != other.call_context:
      return False
    return self.function_signature.is_subtype_of(other.function_signature)
  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["FunctionCacheKey"]:
    if not all(
        isinstance(other, FunctionCacheKey) and
        self.call_context == other.call_context for other in others):
      return None
    common = self.function_signature.most_specific_common_supertype(
        [other.function_signature for other in others])
    if common is None:
      return None
    return FunctionCacheKey(common, self.call_context)
  def _placeholder_value(self) -> Any:
  def __hash__(self) -> int:
    return hash((self.call_context, self.function_signature))
  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented
    if not isinstance(other, FunctionCacheKey):
      return False
    return (self.call_context == other.call_context and
            self.function_signature == other.function_signature)
  def __repr__(self) -> str:
    return (
        f"{type(self).__name__}(function_signature={repr(self.function_signature)},"
        f" call_context={repr(self.call_context)})")
class FunctionCache:
  __slots__ = [
      "_primary", "_dispatch_table", "_garbage_collectors"
  ]
  def __init__(self):
    self._primary = collections.OrderedDict()
    self._dispatch_table = type_dispatch.TypeDispatchTable()
  def lookup(self, key: FunctionCacheKey, use_function_subtyping: bool):
    if not use_function_subtyping:
      return self._primary.get(key, None)
    dispatch_key = self._dispatch_table.dispatch(key)
    if dispatch_key is not None:
      return self._primary[dispatch_key]
    return None
  def delete(self, key: FunctionCacheKey):
    if key not in self._primary:
      return False
    del self._primary[key]
    self._dispatch_table.delete(key)
    return True
  def add(self, key: FunctionCacheKey,
          deletion_observer: trace_type.WeakrefDeletionObserver,
          concrete):
    self._primary[key] = concrete
    self._dispatch_table.add_target(key)
    deletion_observer.add_listener(
        lambda: self.delete(key) if DELETE_WITH_WEAKREF else None)
  def generalize(self, key: FunctionCacheKey) -> FunctionCacheKey:
  def clear(self):
    self._primary.clear()
    self._dispatch_table.clear()
  def values(self):
    return list(self._primary.values())
