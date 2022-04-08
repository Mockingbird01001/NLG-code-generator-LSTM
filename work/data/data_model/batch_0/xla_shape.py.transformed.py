
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python_api import types
class Shape(object):
  def __init__(self, element_type, dimensions, layout=None):
    """Creates a new XLA Shape.
    Args:
      element_type: element type from xla_data_pb2.
      dimensions: sequence of dimensions sizes (integers), or sequence
        of Shapes in the case of a tuple, i.e. when element_type is
        TUPLE.
      layout: optional minor_to_major sequence for layout. If not given, the
        default major-to-minor layout is used.
    Raises:
      ValueError: if element_type is TUPLE but dimensions are not Shape objects.
    """
    self.message = xla_data_pb2.ShapeProto()
    self.message.element_type = element_type
    if element_type == xla_data_pb2.TUPLE:
      if not all(isinstance(subshape, Shape) for subshape in dimensions):
        raise ValueError(
            'XLA tuple requires sequence of Shape objects as dimensions')
      self._tuple_shapes = tuple(dimensions)
      for component_shape in self._tuple_shapes:
        component_message = self.message.tuple_shapes.add()
        component_message.CopyFrom(component_shape.message)
    else:
      self.message.dimensions.extend(dimensions)
      if layout is None:
        layout = list(reversed(range(len(dimensions))))
      self.message.layout.format = xla_data_pb2.DENSE
      self.message.layout.minor_to_major.extend(layout)
  def element_type(self):
    return self.message.element_type
  def is_tuple(self):
    return self.element_type() == xla_data_pb2.TUPLE
  def dimensions(self):
    if self.is_tuple():
      raise ValueError('Tuple shape has no dimensions. Try tuple_shapes()?')
    return self.message.dimensions
  def tuple_shapes(self):
    if not self.is_tuple():
      raise ValueError('tuple_shapes() called on a non-tuple shape')
    return self._tuple_shapes
  def layout(self):
    return self.message.layout
  @staticmethod
  def from_pyval(pyval):
    return CreateShapeFromNumpy(pyval)
  element_type = types.MAP_DTYPE_TO_RECORD[str(ndarray.dtype)].primitive_type
  dimensions = ndarray.shape
  if _np.isfortran(ndarray):
    layout = range(ndarray.ndim)
  else:
    layout = list(reversed(range(ndarray.ndim)))
  return Shape(element_type, dimensions, layout)
  """Create a Shape from a Numpy array or a nested tuple structure thereof.
  Args:
    value: Numpy array or (possibly nested) tuple structure that bottoms out in
      Numpy arrays.
  Returns:
    A Shape object.
  """
  if isinstance(value, tuple):
    return Shape(
        xla_data_pb2.TUPLE,
        [CreateShapeFromNumpy(component) for component in value])
  else:
    return _CreateShapeFromNumpy(value)
  """Create a shape from a Numpy dtype and a sequence of nonnegative integers.
  Args:
    dtype: a numpy dtype, e.g. np.dtype('int32').
    shape_tuple: a sequence of nonnegative integers.
  Returns:
    A Shape object.
  """
  element_type = types.MAP_DTYPE_TO_RECORD[str(dtype)].primitive_type
  return Shape(element_type, shape_tuple)
