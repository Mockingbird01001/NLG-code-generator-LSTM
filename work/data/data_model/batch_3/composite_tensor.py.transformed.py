
import abc
import six
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export("__internal__.CompositeTensor", v1=[])
@six.add_metaclass(abc.ABCMeta)
class CompositeTensor(object):
  """Abstract base class for Tensor-like objects that are composed from Tensors.
  Each `CompositeTensor` can be decomposed into a structured collection of
  component `tf.Tensor`s, and reconstructed from those components.
  The `tensorflow.python.util.nest` module has support for treating composite
  tensors as structure, which makes it easy to flatten and reconstruct
  composite tensors (or larger structures that contain composite tensors).
  E.g.:
  ```python
  flat_list_of_tensors = nest.flatten(ct, expand_composites=True)
  result = nest.pack_sequence_as(ct, transformed_list_of_tensors,
                                 expand_composites=True)
  ```
  """
  @abc.abstractproperty
  def _type_spec(self):
    raise NotImplementedError(f"{type(self).__name__}._type_spec()")
  def _shape_invariant_to_type_spec(self, shape):
    """Returns a TypeSpec given a shape invariant (used by `tf.while_loop`).
    Args:
      shape: A `tf.TensorShape` object.  The shape invariant for this
        `CompositeTensor`, or `None` if a default shape invariant should be used
        (based on the value of this `CompositeTensor`).
    Returns:
      A nested structure whose values are `tf.TensorShape` objects, specifying
      the shape invariants for the tensors that comprise this `CompositeTensor`.
    """
    raise NotImplementedError(
        f"{type(self).__name__}._shape_invariant_to_type_spec")
  def _consumers(self):
    consumers = nest.flatten([
        component.consumers()
        for component in nest.flatten(self, expand_composites=True)
        if getattr(component, "graph", None) is not None
    ])
    return list(set(consumers))
  def __tf_tracing_type__(self, context):
    return self._type_spec.__tf_tracing_type__(context)
_pywrap_utils.RegisterType("CompositeTensor", CompositeTensor)
def replace_composites_with_components(structure):
  """Recursively replaces CompositeTensors with their components.
  Args:
    structure: A `nest`-compatible structure, possibly containing composite
      tensors.
  Returns:
    A copy of `structure`, where each composite tensor has been replaced by
    its components.  The result will contain no composite tensors.
    Note that `nest.flatten(replace_composites_with_components(structure))`
    returns the same value as `nest.flatten(structure)`.
  """
  if isinstance(structure, CompositeTensor):
    return replace_composites_with_components(
  elif not nest.is_nested(structure):
    return structure
  else:
    return nest.map_structure(
        replace_composites_with_components, structure, expand_composites=False)
