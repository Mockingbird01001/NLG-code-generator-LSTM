
import collections
import functools
import weakref
from tensorflow.python.util import object_identity
try:
except ImportError:
  pass
def is_layer(obj):
  return hasattr(obj, "_is_layer") and not isinstance(obj, type)
def has_weights(obj):
  has_weight = (hasattr(type(obj), "trainable_weights")
                and hasattr(type(obj), "non_trainable_weights"))
  return has_weight and not isinstance(obj, type)
def invalidate_recursive_cache(key):
  def outer(f):
    @functools.wraps(f)
    def wrapped(self, value):
      sentinel.invalidate(key)
      return f(self, value)
    return wrapped
  return outer
class MutationSentinel(object):
  _in_cached_state = False
    may_affect_upstream = (value != self._in_cached_state)
    self._in_cached_state = value
    return may_affect_upstream
  @property
  def in_cached_state(self):
    return self._in_cached_state
class AttributeSentinel(object):
  """Container for managing attribute cache state within a Layer.
  The cache can be invalidated either on an individual basis (for instance when
  an attribute is mutated) or a layer-wide basis (such as when a new dependency
  is added).
  """
  def __init__(self, always_propagate=False):
    self._parents = weakref.WeakSet()
    self.attributes = collections.defaultdict(MutationSentinel)
    self.always_propagate = always_propagate
  def __repr__(self):
    return "{}\n  {}".format(
        super(AttributeSentinel, self).__repr__(),
        {k: v.in_cached_state for k, v in self.attributes.items()})
  def add_parent(self, node):
    self._parents.add(node)
    node.invalidate_all()
  def get(self, key):
    return self.attributes[key].in_cached_state
  def _set(self, key, value):
    may_affect_upstream = self.attributes[key].mark_as(value)
    if may_affect_upstream or self.always_propagate:
        node.invalidate(key)
  def mark_cached(self, key):
    self._set(key, True)
  def invalidate(self, key):
    self._set(key, False)
  def invalidate_all(self):
    for key in self.attributes.keys():
      self.attributes[key].mark_as(False)
    for node in self._parents:
      node.invalidate_all()
def filter_empty_layer_containers(layer_list):
  existing = object_identity.ObjectIdentitySet()
  to_visit = layer_list[::-1]
  while to_visit:
    obj = to_visit.pop()
    if obj in existing:
      continue
    existing.add(obj)
    if is_layer(obj):
      yield obj
    else:
      sub_layers = getattr(obj, "layers", None) or []
      to_visit.extend(sub_layers[::-1])
