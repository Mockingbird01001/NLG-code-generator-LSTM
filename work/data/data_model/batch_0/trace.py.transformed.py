
"""tf.function tracing types.
See `core.GenericFunction` and `core.ConcreteFunction`.
`GenericFunction` assigns types to call arguments, forming a signature.
Function signatures are used to match arguments to `ConcreteFunction`s.
For example, when a new `ConcreteFunction` is traced, it is assigned a
the signature of the arguments it was traced with. Subsequent call arguments
which match its signature will be dispatched to the same `ConcreteFunction`.
If no `ConcreteFunction` with a matching signature is found, a new one may be
traced (a process known as retracing).
"""
import abc
from typing import Optional, Sequence
from typing_extensions import Protocol
from typing_extensions import runtime_checkable
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export("types.experimental.TraceType", v1=[])
class TraceType(metaclass=abc.ABCMeta):
  """Represents the type of object(s) for tf.function tracing purposes.
  `TraceType` is an abstract class that other classes might inherit from to
  provide information regarding associated class(es) for the purposes of
  tf.function tracing. The typing logic provided through this mechanism will be
  used to make decisions regarding usage of cached concrete functions and
  retracing.
  For example, if we have the following tf.function and classes:
  ```python
  @tf.function
  def get_mixed_flavor(fruit_a, fruit_b):
    return fruit_a.flavor + fruit_b.flavor
  class Fruit:
    flavor = tf.constant([0, 0])
  class Apple(Fruit):
    flavor = tf.constant([1, 2])
  class Mango(Fruit):
    flavor = tf.constant([3, 4])
  ```
  tf.function does not know when to re-use an existing concrete function in
  regards to the `Fruit` class so naively it retraces for every new instance.
  ```python
  ```
  However, we, as the designers of the `Fruit` class, know that each subclass
  has a fixed flavor and we can reuse an existing traced concrete function if
  it was the same subclass. Avoiding such unnecessary tracing of concrete
  functions can have significant performance benefits.
  ```python
  class FruitTraceType(tf.types.experimental.TraceType):
    def __init__(self, fruit_type):
      self.fruit_type = fruit_type
    def is_subtype_of(self, other):
       return (type(other) is FruitTraceType and
               self.fruit_type is other.fruit_type)
    def most_specific_common_supertype(self, others):
       return self if all(self == other for other in others) else None
  class Fruit:
   def __tf_tracing_type__(self, context):
     return FruitTraceType(type(self))
  ```
  Now if we try calling it again:
  ```python
  ```
  """
  @abc.abstractmethod
  def is_subtype_of(self, other: "TraceType") -> bool:
    """Returns True if `self` is a subtype of `other`.
    For example, `tf.function` uses subtyping for dispatch:
    if `a.is_subtype_of(b)` is True, then an argument of `TraceType`
    `a` can be used as argument to a `ConcreteFunction` traced with an
    a `TraceType` `b`.
    Args:
     other: A TraceType object to be compared against.
    Example:
    ```python
    class Dimension(TraceType):
      def __init__(self, value: Optional[int]):
        self.value = value
      def is_subtype_of(self, other):
        return (self.value == other.value) or (other.value is None)
    ```
    """
  @abc.abstractmethod
  def most_specific_common_supertype(
      self, others: Sequence["TraceType"]) -> Optional["TraceType"]:
    """Returns the most specific supertype of `self` and `others`, if exists.
    The returned `TraceType` is a supertype of `self` and `others`, that is,
    they are all subtypes (see `is_subtype_of`) of it.
    It is also most specific, that is, there it has no subtype that is also
    a common supertype of `self` and `others`.
    If `self` and `others` have no common supertype, this returns `None`.
    Args:
     others: A sequence of TraceTypes.
    Example:
    ```python
     class Dimension(TraceType):
       def __init__(self, value: Optional[int]):
         self.value = value
       def most_specific_common_supertype(self, other):
          if self.value == other.value:
            return self.value
          else:
            return Dimension(None)
    ```
    """
  @doc_controls.do_not_doc_inheritable
  def _placeholder_value(self):
    """Creates a placeholder for tracing.
    Often it is more useful to trace with a placeholder value than an actual
    one. For example, a placeholder value can represent multiple different
    actual values. This means that the trace generated with that placeholder
    value is more general and reusable which saves expensive retracing.
    For the `Fruit` example shared above, implementing:
    ```python
    class FruitTraceType:
      def _placeholder_value():
        return Fruit()
    ```
    instructs tf.function to trace with the `Fruit()` objects
    instead of the actual `Apple()` and `Mango()` objects when it receives a
    call to `get_mixed_flavor(Apple(), Mango())`. For example, Tensor arguments
    are replaced with Tensors of similar shape and dtype, output from
    a tf.Placeholder op.
    More generally, placeholder values are the arguments of a tf.function,
    as seen from the function's body:
    ```python
    @tf.function
    def foo(x):
      ...
    ```
    """
    raise NotImplementedError
  @abc.abstractmethod
  def __hash__(self) -> int:
    pass
  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    pass
@tf_export("types.experimental.TracingContext", v1=[])
class TracingContext(metaclass=abc.ABCMeta):
  pass
@runtime_checkable
class SupportsTracingProtocol(Protocol):
  @abc.abstractmethod
  def __tf_tracing_type__(self, context: TracingContext) -> TraceType:
tf_export(
    "types.experimental.SupportsTracingProtocol",
    v1=[]).export_constant(__name__, "SupportsTracingProtocol")
