
import functools
import operator
from typing import Optional, Sequence
import six
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
_TENSORSHAPE_V2_OVERRIDE = None
_api_usage_gauge = monitoring.BoolGauge(
    "/tensorflow/api/v2_tensorshape",
    "Whether tensor_shape.enable_v2_tensorshape() is called.")
@tf_export(v1=["enable_v2_tensorshape"])
def enable_v2_tensorshape():
  """In TensorFlow 2.0, iterating over a TensorShape instance returns values.
  This enables the new behavior.
  Concretely, `tensor_shape[i]` returned a Dimension instance in V1, but
  it V2 it returns either an integer, or None.
  Examples:
  ```
  value = tensor_shape[i].value
  value = tensor_shape[i]
  for dim in tensor_shape:
    value = dim.value
    print(value)
  for value in tensor_shape:
    print(value)
  dim = tensor_shape[i]
  if tensor_shape.rank is None:
    dim = Dimension(None)
  else:
    dim = tensor_shape.dims[i]
  ```
  """
  _TENSORSHAPE_V2_OVERRIDE = True
  logging.vlog(1, "Enabling v2 tensorshape")
  _api_usage_gauge.get_cell().set(True)
@tf_export(v1=["disable_v2_tensorshape"])
def disable_v2_tensorshape():
  _TENSORSHAPE_V2_OVERRIDE = False
  logging.vlog(1, "Disabling v2 tensorshape")
  _api_usage_gauge.get_cell().set(False)
@tf_export(
    "compat.dimension_value", v1=["dimension_value", "compat.dimension_value"])
def dimension_value(dimension):
  """Compatibility utility required to allow for both V1 and V2 behavior in TF.
  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.
  When accessing the value of a TensorShape dimension,
  use this utility, like this:
  ```
  value = tensor_shape[i].value
  value = dimension_value(tensor_shape[i])
  ```
  Args:
    dimension: Either a `Dimension` instance, an integer, or None.
  Returns:
    A plain value, i.e. an integer or None.
  """
  if isinstance(dimension, Dimension):
    return dimension.value
  return dimension
@tf_export(
    "compat.dimension_at_index",
    v1=["dimension_at_index", "compat.dimension_at_index"])
def dimension_at_index(shape, index):
  """Compatibility utility required to allow for both V1 and V2 behavior in TF.
  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.
  If you want to retrieve the Dimension instance corresponding to a certain
  index in a TensorShape instance, use this utility, like this:
  ```
  dim = tensor_shape[i]
  dim = dimension_at_index(tensor_shape, i)
  if tensor_shape.rank is None:
    dim = Dimension(None)
  else:
    dim = tensor_shape.dims[i]
  ```
  Args:
    shape: A TensorShape instance.
    index: An integer index.
  Returns:
    A dimension object.
  """
  assert isinstance(shape, TensorShape)
  if shape.rank is None:
    return Dimension(None)
  else:
    return shape.dims[index]
@tf_export(v1=["Dimension"])
class Dimension(object):
  """Represents the value of one dimension in a TensorShape.
  @compatibility(TF2)
  In TF2, members of a `TensorShape` object are integers. The `Dimension` class
  is not part of TF2's data model.
  Please refer to the [TensorShape section of the migration guide]
  patterns adapting Dimension objects to a TF2 syntax.
  @end_compatibility
  """
  __slots__ = ["_value"]
  def __init__(self, value):
      if value < 0:
        raise ValueError("Dimension %d must be >= 0" % value)
      self._value = value
    elif value is None:
      self._value = None
    elif isinstance(value, Dimension):
      self._value = value._value
    else:
      try:
        self._value = int(value.__index__())
      except AttributeError:
        six.raise_from(
            TypeError("Dimension value must be integer or None or have "
                      "an __index__ method, got value '{0!r}' with type '{1!r}'"
                      .format(value, type(value))), None)
      if self._value < 0:
        raise ValueError("Dimension %d must be >= 0" % self._value)
  def __repr__(self):
    return "Dimension(%s)" % repr(self._value)
  def __str__(self):
    value = self._value
    return "?" if value is None else str(value)
  def __eq__(self, other):
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value == other.value
  def __ne__(self, other):
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value != other.value
  def __bool__(self):
    return bool(self._value)
  def __int__(self):
    return self._value
  def __long__(self):
    return self._value
  def __index__(self):
    return self._value
  @property
  def value(self):
    return self._value
  def is_compatible_with(self, other):
    other = as_dimension(other)
    return (self._value is None or other.value is None or
            self._value == other.value)
  def assert_is_compatible_with(self, other):
    """Raises an exception if `other` is not compatible with this Dimension.
    Args:
      other: Another Dimension.
    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    if not self.is_compatible_with(other):
      raise ValueError("Dimensions %s and %s are not compatible" %
                       (self, other))
  def merge_with(self, other):
    """Returns a Dimension that combines the information in `self` and `other`.
    Dimensions are combined as follows:
    ```python
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(None))  ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(None))
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(m))
    ```
    Args:
      other: Another Dimension.
    Returns:
      A Dimension containing the combined information of `self` and
      `other`.
    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    other = as_dimension(other)
    self.assert_is_compatible_with(other)
    if self._value is None:
      return Dimension(other.value)
    else:
      return Dimension(self._value)
  def __add__(self, other):
    """Returns the sum of `self` and `other`.
    Dimensions are summed as follows:
    ```python
    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m + n)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value + other.value)
  def __radd__(self, other):
    return self + other
  def __sub__(self, other):
    """Returns the subtraction of `other` from `self`.
    Dimensions are subtracted as follows:
    ```python
    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m - n)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the subtraction of `other` from `self`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value - other.value)
  def __rsub__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value - self._value)
  def __mul__(self, other):
    """Returns the product of `self` and `other`.
    Dimensions are summed as follows:
    ```python
    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m * n)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the product of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value * other.value)
  def __rmul__(self, other):
    return self * other
  def __floordiv__(self, other):
    """Returns the quotient of `self` and `other` rounded down.
    Dimensions are divided as follows:
    ```python
    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m // n)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value // other.value)
  def __rfloordiv__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value // self._value)
  def __div__(self, other):
    return self // other
  def __rdiv__(self, other):
    """Use `__floordiv__` via `x // y` instead.
    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
    this function will explicitly call for usage of `//` instead.
    Args:
      other: Another `Dimension`.
    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', "
                    "please use // instead".format(type(other).__name__))
  def __truediv__(self, other):
    """Use `__floordiv__` via `x // y` instead.
    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'Dimension' and 'int'`,
    this function will explicitly call for usage of `//` instead.
    Args:
      other: Another `Dimension`.
    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: 'Dimension' and '{}', "
                    "please use // instead".format(type(other).__name__))
  def __rtruediv__(self, other):
    """Use `__floordiv__` via `x // y` instead.
    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
    this function will explicitly call for usage of `//` instead.
    Args:
      other: Another `Dimension`.
    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', "
                    "please use // instead".format(type(other).__name__))
  def __mod__(self, other):
    """Returns `self` modulo `other`.
    Dimension modulo are computed as follows:
    ```python
    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m % n)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is `self` modulo `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value % other.value)
  def __rmod__(self, other):
    other = as_dimension(other)
    return other % self
  def __lt__(self, other):
    """Returns True if `self` is known to be less than `other`.
    Dimensions are compared as follows:
    ```python
    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(n))    == (m < n)
    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(None)) == None
    ```
    Args:
      other: Another Dimension.
    Returns:
      The value of `self.value < other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value < other.value
  def __le__(self, other):
    """Returns True if `self` is known to be less than or equal to `other`.
    Dimensions are compared as follows:
    ```python
    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(n))    == (m <= n)
    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(None)) == None
    ```
    Args:
      other: Another Dimension.
    Returns:
      The value of `self.value <= other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value <= other.value
  def __gt__(self, other):
    """Returns True if `self` is known to be greater than `other`.
    Dimensions are compared as follows:
    ```python
    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(n))    == (m > n)
    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(None)) == None
    ```
    Args:
      other: Another Dimension.
    Returns:
      The value of `self.value > other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value > other.value
  def __ge__(self, other):
    """Returns True if `self` is known to be greater than or equal to `other`.
    Dimensions are compared as follows:
    ```python
    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(n))    == (m >= n)
    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(None)) == None
    ```
    Args:
      other: Another Dimension.
    Returns:
      The value of `self.value >= other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value >= other.value
  def __reduce__(self):
    return Dimension, (self._value,)
def as_dimension(value):
  if isinstance(value, Dimension):
    return value
  else:
    return Dimension(value)
@tf_export("TensorShape")
class TensorShape(trace.TraceType):
  """Represents the shape of a `Tensor`.
  A `TensorShape` represents a possibly-partial shape specification for a
  `Tensor`. It may be one of the following:
  * *Fully-known shape:* has a known number of dimensions and a known size
    for each dimension. e.g. `TensorShape([16, 256])`
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension. e.g. `TensorShape([None, 256])`
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions. e.g. `TensorShape(None)`
  If a tensor is produced by an operation of type `"Foo"`, its shape
  may be inferred if there is a registered shape function for
  `"Foo"`. See [Shape
  for details of shape functions and how to register them. Alternatively,
  you may set the shape explicitly using `tf.Tensor.set_shape`.
  """
  __slots__ = ["_dims"]
  def __init__(self, dims):
      self._dims = tuple(as_dimension(d).value for d in dims)
    elif dims is None:
      self._dims = None
    elif isinstance(dims, tensor_shape_pb2.TensorShapeProto):
      if dims.unknown_rank:
        self._dims = None
      else:
        self._dims = tuple(
            dim.size if dim.size != -1 else None
            for dim in dims.dim
            )
    elif isinstance(dims, TensorShape):
      self._dims = dims._dims
    else:
      try:
        dims_iter = iter(dims)
      except TypeError:
        self._dims = (as_dimension(dims).value,)
      else:
        self._dims = []
        for d in dims_iter:
          try:
            self._dims.append(as_dimension(d).value)
          except TypeError as e:
            six.raise_from(
                TypeError(
                    "Failed to convert '{0!r}' to a shape: '{1!r}'"
                    "could not be converted to a dimension. A shape should "
                    "either be single dimension (e.g. 10), or an iterable of "
                    "dimensions (e.g. [1, 10, None])."
                    .format(dims, d)), e)
        self._dims = tuple(self._dims)
  @property
  def _v2_behavior(self):
    if _TENSORSHAPE_V2_OVERRIDE is None:
      return tf2.enabled()
    return _TENSORSHAPE_V2_OVERRIDE
  def __repr__(self):
    if self._v2_behavior:
      if self._dims is not None:
        return f"TensorShape({list(self._dims)})"
      else:
        return "TensorShape(None)"
    else:
      return f"TensorShape({self.dims})"
  def __str__(self):
    if self.rank is None:
      return "<unknown>"
    elif self.rank == 1:
      if self._v2_behavior:
        return "(%s,)" % self._dims[0]
      else:
        return "(%s,)" % self.dims[0]
    else:
      if self._v2_behavior:
        return "(%s)" % ", ".join(str(d) for d in self._dims)
      else:
        return "(%s)" % ", ".join(str(d) for d in self.dims)
  @property
  def rank(self):
    if self._dims is not None:
      return len(self._dims)
    return None
  @property
  def dims(self):
    if self._dims is None:
      return None
    return [as_dimension(d) for d in self._dims]
  @property
  def ndims(self):
    return self.rank
  def __len__(self):
    if self._dims is None:
      raise ValueError("Cannot take the length of shape with unknown rank.")
    return len(self._dims)
  def __bool__(self):
    return self._dims is not None
  __nonzero__ = __bool__
  def __iter__(self):
    if self._dims is None:
      raise ValueError("Cannot iterate over a shape with unknown rank.")
    else:
      if self._v2_behavior:
        return iter(d for d in self._dims)
      else:
        return iter(d for d in self.dims)
  def __getitem__(self, key):
    if self._dims is not None:
      if isinstance(key, slice):
        return TensorShape(self._dims[key])
      else:
        if self._v2_behavior:
          return self._dims[key]
        else:
          return self.dims[key]
    else:
      if isinstance(key, slice):
        start = key.start if key.start is not None else 0
        stop = key.stop
        if key.step is not None:
          raise ValueError("Steps are not yet handled")
        if stop is None:
          return unknown_shape()
        elif start < 0 or stop < 0:
          return unknown_shape()
        else:
          return unknown_shape(rank=stop - start)
      else:
        if self._v2_behavior:
          return None
        else:
          return Dimension(None)
  def num_elements(self):
    if self.is_fully_defined():
      return functools.reduce(operator.mul, self.as_list(), 1)
    else:
      return None
  def merge_with(self, other):
    """Returns a `TensorShape` combining the information in `self` and `other`.
    The dimensions in `self` and `other` are merged element-wise,
    according to the rules below:
    ```python
    Dimension(n).merge_with(Dimension(None)) == Dimension(n)
    Dimension(None).merge_with(Dimension(n)) == Dimension(n)
    Dimension(None).merge_with(Dimension(None)) == Dimension(None)
    Dimension(n).merge_with(Dimension(m))
    ```
    >> ts = tf.TensorShape([1,2])
    >> ot1 = tf.TensorShape([1,2])
    >> ts.merge_with(ot).as_list()
    [1,2]
    >> ot2 = tf.TensorShape([1,None])
    >> ts.merge_with(ot2).as_list()
    [1,2]
    >> ot3 = tf.TensorShape([None, None])
    >> ot3.merge_with(ot2).as_list()
    [1, None]
    Args:
      other: Another `TensorShape`.
    Returns:
      A `TensorShape` containing the combined information of `self` and
      `other`.
    Raises:
      ValueError: If `self` and `other` are not compatible.
    """
    other = as_shape(other)
    if self.dims is None:
      return other
    if other.dims is None:
      return self
    else:
      try:
        self.assert_same_rank(other)
        new_dims = [
            dim.merge_with(other_dim)
            for dim, other_dim in zip(self.dims, other.dims)
        ]
        return TensorShape(new_dims)
      except ValueError:
        raise ValueError("Shapes %s and %s are not compatible" % (self, other))
  def __add__(self, other):
    return self.concatenate(other)
  def __radd__(self, other):
    if not isinstance(other, TensorShape):
      other = TensorShape(other)
    return other.concatenate(self)
  def concatenate(self, other):
    other = as_shape(other)
    if self.dims is None or other.dims is None:
      return unknown_shape()
    else:
      return TensorShape(self.dims + other.dims)
  def assert_same_rank(self, other):
    other = as_shape(other)
    if self.rank is not None and other.rank is not None:
      if self.rank != other.rank:
        raise ValueError("Shapes %s and %s must have the same rank" %
                         (self, other))
  def assert_has_rank(self, rank):
    if self.rank not in (None, rank):
      raise ValueError("Shape %s must have rank %d" % (self, rank))
  def with_rank(self, rank):
    try:
      return self.merge_with(unknown_shape(rank=rank))
    except ValueError:
      raise ValueError("Shape %s must have rank %d" % (self, rank))
  def with_rank_at_least(self, rank):
    if self.rank is not None and self.rank < rank:
      raise ValueError("Shape %s must have rank at least %d" % (self, rank))
    else:
      return self
  def with_rank_at_most(self, rank):
    if self.rank is not None and self.rank > rank:
      raise ValueError("Shape %s must have rank at most %d" % (self, rank))
    else:
      return self
  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """Returns True iff `self` is subtype of `other`.
    Shape A is a subtype of shape B if shape B can successfully represent it:
    * A `TensorShape` of any rank is a subtype of `TensorShape(None)`.
    *  TensorShapes of equal ranks are covariant, i.e.
      `TensorShape([A1, A2, ..])` is a subtype of
      `TensorShape([B1, B2, ..])` iff An is a subtype of Bn.
      An is subtype of Bn iff An == Bn or Bn is None.
    * TensorShapes of different defined ranks have no subtyping relation.
    The subtyping relation is reflexive and transitive, but not symmetric.
    Some examples:
    * `TensorShape([32, 784])` is a subtype of `TensorShape(None)`, and
      `TensorShape([4, 4])` is also a subtype of `TensorShape(None)` but
      `TensorShape([32, 784])` and `TensorShape([4, 4])` are not subtypes of
      each other.
    * All two-dimensional shapes are subtypes of `TensorShape([None, None])`,
      such as `TensorShape([32, 784])`. There is no subtype relationship with,
      for example, `TensorShape([None])` or `TensorShape([None, None, None])`.
    * `TensorShape([32, None])` is also a subtype of `TensorShape([None, None])`
      and `TensorShape(None)`. It is not a subtype of, for example,
      `TensorShape([32])`, `TensorShape([32, None, 1])`,
      `TensorShape([64, None])` or `TensorShape([None, 32])`.
    * `TensorShape([32, 784])` is a subtype of itself, and also
      `TensorShape([32, None])`, `TensorShape([None, 784])`,
      `TensorShape([None, None])` and `TensorShape(None)`.
      It has no subtype relation with, for example, `TensorShape([32, 1, 784])`
      or `TensorShape([None])`.
    Args:
      other: Another `TensorShape`.
    Returns:
      True iff `self` is subtype of `other`.
    """
    if not isinstance(other, TensorShape):
      return False
    if other.rank is None:
      return True
    if self.rank != other.rank:
      return False
  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["TensorShape"]:
    """Returns the most specific supertype `TensorShape` of self and others.
    * `TensorShape([None, 1])` is the most specific `TensorShape` supertyping
      both `TensorShape([2, 1])` and `TensorShape([5, 1])`. Note that
      `TensorShape(None)` is also a supertype but it is not "most specific".
    * `TensorShape([1, 2, 3])` is the most specific `TensorShape` supertyping
      both `TensorShape([1, 2, 3])` and `TensorShape([1, 2, 3]`). There are
      other less specific TensorShapes that supertype above mentioned
      TensorShapes, e.g. `TensorShape([1, 2, None])`, `TensorShape(None)`.
     * `TensorShape([None, None])` is the most specific `TensorShape`
       supertyping both `TensorShape([2, None])` and `TensorShape([None, 3])`.
       As always, `TensorShape(None)` is also a supertype but not the most
       specific one.
     * `TensorShape(None`) is the only `TensorShape` supertyping both
       `TensorShape([1, 2, 3])` and `TensorShape([1, 2])`. In general, any two
       shapes that have different ranks will only have `TensorShape(None)`
       as a common supertype.
     * `TensorShape(None)` is the only `TensorShape` supertyping both
       `TensorShape([1, 2, 3])` and `TensorShape(None)`. In general, the common
       supertype of any shape with `TensorShape(None)` is `TensorShape(None)`.
    Args:
      others: Sequence of `TensorShape`.
    Returns:
      A `TensorShape` which is the most specific supertype shape of `self`
      and `others`. None if it does not exist.
    """
    if any(not isinstance(other, TensorShape) for other in others):
      return None
    if self.rank is None:
      return unknown_shape()
    if any(other.dims is None or self.rank != other.rank for other in others):
      return unknown_shape()
    dims = [
        dim if all(dim == other._dims[i]
                   for other in others) else None
        for i, dim in enumerate(self._dims)
    ]
    return TensorShape(dims)
  def is_compatible_with(self, other):
    """Returns True iff `self` is compatible with `other`.
    Two possibly-partially-defined shapes are compatible if there
    exists a fully-defined shape that both shapes can represent. Thus,
    compatibility allows the shape inference code to reason about
    partially-defined shapes. For example:
    * TensorShape(None) is compatible with all shapes.
    * TensorShape([None, None]) is compatible with all two-dimensional
      shapes, such as TensorShape([32, 784]), and also TensorShape(None). It is
      not compatible with, for example, TensorShape([None]) or
      TensorShape([None, None, None]).
    * TensorShape([32, None]) is compatible with all two-dimensional shapes
      with size 32 in the 0th dimension, and also TensorShape([None, None])
      and TensorShape(None). It is not compatible with, for example,
      TensorShape([32]), TensorShape([32, None, 1]) or TensorShape([64, None]).
    * TensorShape([32, 784]) is compatible with itself, and also
      TensorShape([32, None]), TensorShape([None, 784]), TensorShape([None,
      None]) and TensorShape(None). It is not compatible with, for example,
      TensorShape([32, 1, 784]) or TensorShape([None]).
    The compatibility relation is reflexive and symmetric, but not
    transitive. For example, TensorShape([32, 784]) is compatible with
    TensorShape(None), and TensorShape(None) is compatible with
    TensorShape([4, 4]), but TensorShape([32, 784]) is not compatible with
    TensorShape([4, 4]).
    Args:
      other: Another TensorShape.
    Returns:
      True iff `self` is compatible with `other`.
    """
    other = as_shape(other)
    if self.dims is not None and other.dims is not None:
      if self.rank != other.rank:
        return False
      for x_dim, y_dim in zip(self.dims, other.dims):
        if not x_dim.is_compatible_with(y_dim):
          return False
    return True
  def assert_is_compatible_with(self, other):
    if not self.is_compatible_with(other):
      raise ValueError("Shapes %s and %s are incompatible" % (self, other))
  def most_specific_compatible_shape(self, other):
    """Returns the most specific TensorShape compatible with `self` and `other`.
    * TensorShape([None, 1]) is the most specific TensorShape compatible with
      both TensorShape([2, 1]) and TensorShape([5, 1]). Note that
      TensorShape(None) is also compatible with above mentioned TensorShapes.
    * TensorShape([1, 2, 3]) is the most specific TensorShape compatible with
      both TensorShape([1, 2, 3]) and TensorShape([1, 2, 3]). There are more
      less specific TensorShapes compatible with above mentioned TensorShapes,
      e.g. TensorShape([1, 2, None]), TensorShape(None).
    Args:
      other: Another `TensorShape`.
    Returns:
      A `TensorShape` which is the most specific compatible shape of `self`
      and `other`.
    """
    other = as_shape(other)
    if self.dims is None or other.dims is None or self.rank != other.rank:
      return unknown_shape()
    dims = [
        d1 if d1 is not None and d2 is not None and d1 == d2 else None
        for d1, d2 in zip(self.dims, other.dims)
    ]
    return TensorShape(dims)
  def is_fully_defined(self):
    return (self._dims is not None and
            all(dim is not None for dim in self._dims))
  def assert_is_fully_defined(self):
    if not self.is_fully_defined():
      raise ValueError("Shape %s is not fully defined" % self)
  def as_list(self):
    if self._dims is None:
      raise ValueError("as_list() is not defined on an unknown TensorShape.")
    return list(self._dims)
  def as_proto(self):
    if self._dims is None:
      return tensor_shape_pb2.TensorShapeProto(unknown_rank=True)
    else:
      return tensor_shape_pb2.TensorShapeProto(dim=[
          tensor_shape_pb2.TensorShapeProto.Dim(
              size=-1 if d is None else d) for d in self._dims
      ])
  def __eq__(self, other):
    """Returns True if `self` is equivalent to `other`.
    It first tries to convert `other` to `TensorShape`. `TypeError` is thrown
    when the conversion fails. Otherwise, it compares each element in the
    TensorShape dimensions.
    * Two *Fully known* shapes, return True iff each element is equal.
    >>> t_a = tf.TensorShape([1,2])
    >>> a = [1, 2]
    >>> t_b = tf.TensorShape([1,2])
    >>> t_c = tf.TensorShape([1,2,3])
    >>> t_a.__eq__(a)
    True
    >>> t_a.__eq__(t_b)
    True
    >>> t_a.__eq__(t_c)
    False
    * Two *Partially-known* shapes, return True iff each element is equal.
    >>> p_a = tf.TensorShape([1,None])
    >>> p_b = tf.TensorShape([1,None])
    >>> p_c = tf.TensorShape([2,None])
    >>> p_a.__eq__(p_b)
    True
    >>> t_a.__eq__(p_a)
    False
    >>> p_a.__eq__(p_c)
    False
    * Two *Unknown shape*, return True.
    >>> unk_a = tf.TensorShape(None)
    >>> unk_b = tf.TensorShape(None)
    >>> unk_a.__eq__(unk_b)
    True
    >>> unk_a.__eq__(t_a)
    False
    Args:
      other: A `TensorShape` or type that can be converted to `TensorShape`.
    Returns:
      True if the dimensions are all equal.
    Raises:
      TypeError if `other` can not be converted to `TensorShape`.
    """
    try:
      other = as_shape(other)
    except TypeError:
      return NotImplemented
    return self._dims == other._dims
  def __hash__(self):
    return hash(self._dims)
  def __reduce__(self):
    return TensorShape, (self.dims,)
  def __concat__(self, other):
    return self.concatenate(other)
def as_shape(shape):
  if isinstance(shape, TensorShape):
    return shape
  else:
    return TensorShape(shape)
def unknown_shape(rank=None, **kwargs):
  """Returns an unknown TensorShape, optionally with a known rank.
  Args:
    rank: (Optional) If specified, the number of dimensions in the shape.
    **kwargs: For backwards compatibility.
  Returns:
    An unknown TensorShape.
  Raises:
    TypeError: In case of invalid arguments.
  """
  if rank is None and "ndims" in kwargs:
    rank = kwargs.pop("ndims")
  if kwargs:
    raise TypeError("Unknown argument: %s" % kwargs)
  if rank is None:
    return TensorShape(None)
  else:
    return TensorShape([Dimension(None)] * rank)
