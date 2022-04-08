
from tensorflow.core.protobuf import composite_tensor_variant_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_composite_tensor_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def composite_tensor_to_variants(value, type_spec=None, name=None):
  """Encodes `value` as a scalar variant tensor.
  Args:
    value: The `ExtensionType` value to encode.
    type_spec: Information about the value's type that should be included in the
      encoding.
    name: Optional name for the operation.
  Returns:
    A Tensor with shape=`()` and dtype=`tf.variant`.
  Raises:
    ValueError: If `type_spec` is not compatible with `value`.
  """
  if not isinstance(value, composite_tensor.CompositeTensor):
    raise TypeError("Expected `value` to be a CompositeTensor. "
                    f"Received {type(value)}.")
  if type_spec is None:
  if not type_spec.is_compatible_with(value):
    raise ValueError(f"`type_spec` {type_spec} is not compatible with `value` "
                     f"{value!r}.")
  metadata = composite_tensor_variant_pb2.CompositeTensorVariantMetadata()
  metadata.type_spec_proto.CopyFrom(
      nested_structure_coder.encode_structure(type_spec).type_spec_value)
  return gen_composite_tensor_ops.CompositeTensorVariantFromComponents(
      components=nest.flatten(value, expand_composites=True),
      metadata=metadata.SerializeToString(),
      name=name)
def composite_tensor_from_variant(encoded, type_spec, name=None):
  if not isinstance(encoded, ops.Tensor):
    raise TypeError(f"Expected `encoded` to be a Tensor, got {encoded!r}.")
  if encoded.dtype != dtypes.variant:
    raise TypeError("Expected `encoded` to have dtype=variant, got "
                    f"{encoded!r}.")
  encoded.shape.assert_is_compatible_with(())
  metadata = composite_tensor_variant_pb2.CompositeTensorVariantMetadata()
  metadata.type_spec_proto.CopyFrom(
      nested_structure_coder.encode_structure(type_spec).type_spec_value)
  component_dtypes = [
      t.dtype for t in nest.flatten(type_spec, expand_composites=True)
  ]
  components = gen_composite_tensor_ops.CompositeTensorVariantToComponents(
      encoded=encoded,
      metadata=metadata.SerializeToString(),
      Tcomponents=component_dtypes,
      name=name)
  return nest.pack_sequence_as(type_spec, components, expand_composites=True)
@ops.RegisterGradient("CompositeTensorVariantFromComponents")
def _composite_tensor_to_variants_grad(op, grad):
  return gen_composite_tensor_ops.CompositeTensorVariantToComponents(
      encoded=grad,
      metadata=op.get_attr("metadata"),
      Tcomponents=op.get_attr("Tcomponents"))
@ops.RegisterGradient("CompositeTensorVariantToComponents")
def _composite_tensor_from_variant_grad(op, *grad):
  assert len(grad) == len(op.outputs)
  components = [
      op.outputs[i] if grad[i] is None else grad[i] for i in range(len(grad))
  ]
  return gen_composite_tensor_ops.CompositeTensorVariantFromComponents(
      components=components, metadata=op.get_attr("metadata"))
