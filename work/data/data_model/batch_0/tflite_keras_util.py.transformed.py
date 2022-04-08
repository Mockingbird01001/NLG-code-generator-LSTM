
"""Keras functions required by TensorFlow Lite.
The functions defined in this library have been copied over from Keras in order
to remove the dependency from TensorFlow Lite to Keras. The functions which
could not be copied over are accessed using the dependency inversion principle.
(for details, refer to tensorflow/python/util/keras_deps.py).
"""
import copy
from tensorflow.python.eager import def_function
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def _enforce_names_consistency(specs):
  def _has_name(spec):
    return hasattr(spec, 'name') and spec.name is not None
  def _clear_name(spec):
    spec = copy.deepcopy(spec)
    if hasattr(spec, 'name'):
    return spec
  flat_specs = nest.flatten(specs)
  name_inconsistency = (
      any(_has_name(s) for s in flat_specs) and
      not all(_has_name(s) for s in flat_specs))
  if name_inconsistency:
    specs = nest.map_structure(_clear_name, specs)
  return specs
def model_input_signature(model, keep_original_batch_size=False):
  """Inspect model to get its input signature.
  The model's input signature is a list with a single (possibly-nested) object.
  This is due to the Keras-enforced restriction that tensor inputs must be
  passed in as the first argument.
  For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}
  will have input signature: [{'feature1': TensorSpec, 'feature2': TensorSpec}]
  Args:
    model: Keras Model object.
    keep_original_batch_size: A boolean indicating whether we want to keep using
      the original batch size or set it to None. Default is `False`, which means
      that the batch dim of the returned input signature will always be set to
      `None`.
  Returns:
    A list containing either a single TensorSpec or an object with nested
    TensorSpecs. This list does not contain the `training` argument.
  """
  if hasattr(model, 'save_spec'):
    input_specs = model.save_spec(dynamic_batch=not keep_original_batch_size)
    if input_specs is None:
      return None
    input_specs = input_specs[0][0]
  else:
        dynamic_batch=not keep_original_batch_size)
    if input_specs is None:
      return None
  input_specs = _enforce_names_consistency(input_specs)
  if isinstance(input_specs,
                collections_abc.Sequence) and len(input_specs) == 1:
    return input_specs
  else:
    return [input_specs]
def raise_model_input_error(model):
  raise ValueError(
      'Model {} cannot be saved because the input shapes have not been '
      'set. Usually, input shapes are automatically determined from calling'
      ' `.fit()` or `.predict()`. To manually set the shapes, call '
      '`model.build(input_shape)`.'.format(model))
def _create_pseudo_names(tensors, prefix):
  def one_index(ele):
    if isinstance(ele, int):
      return ele + 1
    return ele
  flat_paths = list(nest.yield_flat_paths(tensors))
  flat_paths = nest.map_structure(one_index, flat_paths)
  names = []
  for path in flat_paths:
    if not path:
    else:
      name = '_'.join(str(p) for p in path)
      if isinstance(path[0], int):
        name = prefix + name
    names.append(name)
  return names
def create_pseudo_output_names(outputs):
  return _create_pseudo_names(outputs, prefix='output_')
def trace_model_call(model, input_signature=None):
  if input_signature is None:
    if isinstance(model.call, def_function.Function):
      input_signature = model.call.input_signature
  if input_signature is None:
    input_signature = model_input_signature(model)
  if input_signature is None:
    raise_model_input_error(model)
  @def_function.function(input_signature=input_signature, autograph=False)
  def _wrapped_model(*args):
    inputs = args[0] if len(input_signature) == 1 else list(args)
    with keras_deps.get_call_context_function()().enter(
        model, inputs=inputs, build_graph=False, training=False, saving=True):
      outputs = model(inputs, training=False)
    return outputs
  return _wrapped_model
