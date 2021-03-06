
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python_api import types
from tensorflow.compiler.xla.python_api import xla_shape
def ConvertLiteralToNumpyArray(literal):
  element_type = literal.shape.element_type
  if element_type == xla_data_pb2.TUPLE:
    return tuple(
        ConvertLiteralToNumpyArray(subliteral)
        for subliteral in literal.tuple_literals)
  type_record = types.MAP_XLA_TYPE_TO_RECORD[element_type]
  if not literal.shape.dimensions:
    return _np.array(
        getattr(literal, type_record.literal_field_name)[0],
        type_record.numpy_dtype)
  else:
    layout_order = literal.shape.layout.minor_to_major
    numpy_shape = tuple(literal.shape.dimensions)
    if layout_order == list(range(len(literal.shape.dimensions))):
      numpy_reshaper = lambda arr: arr.reshape(numpy_shape, order='F')
    elif layout_order == list(range(len(literal.shape.dimensions) - 1, -1, -1)):
      numpy_reshaper = lambda arr: arr.reshape(numpy_shape, order='C')
    else:
      raise NotImplementedError('Unsupported layout: {0}'.format(layout_order))
    ndarray = _np.array(
        getattr(literal, type_record.literal_field_name),
        copy=False,
        dtype=type_record.numpy_dtype)
    return numpy_reshaper(ndarray)
def _ConvertNumpyArrayToLiteral(ndarray):
  type_record = types.MAP_DTYPE_TO_RECORD[str(ndarray.dtype)]
  literal = xla_data_pb2.LiteralProto()
  literal.shape.CopyFrom(xla_shape.CreateShapeFromNumpy(ndarray).message)
  if ndarray.ndim == 0:
    getattr(literal, type_record.literal_field_name).append(
        ndarray.astype(type_record.literal_field_type).item())
  else:
    if ndarray.dtype in {_np.bool_, _np.dtype('bool')}:
      for element in _np.nditer(ndarray):
        getattr(literal, type_record.literal_field_name).append(
            type_record.literal_field_type(element))
    else:
      ndarray_flat = ndarray.ravel(order='A')
      getattr(literal, type_record.literal_field_name).extend(ndarray_flat)
  return literal
def ConvertNumpyArrayToLiteral(value):
  if isinstance(value, tuple):
    literal = xla_data_pb2.LiteralProto()
    literal.shape.CopyFrom(xla_shape.CreateShapeFromNumpy(value).message)
    for component in value:
      component_literal = literal.tuple_literals.add()
      component_literal.CopyFrom(ConvertNumpyArrayToLiteral(component))
    return literal
  else:
    return _ConvertNumpyArrayToLiteral(value)
