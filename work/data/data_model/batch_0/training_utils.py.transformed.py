
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def slice_arrays(arrays, indices, contiguous=True):
  """Slices batches out of provided arrays (workaround for eager tensors).
  Unfortunately eager tensors don't have the same slicing behavior as
  Numpy arrays (they follow the same slicing behavior as symbolic TF tensors),
  hence we cannot use `generic_utils.slice_arrays` directly
  and we have to implement this workaround based on `concat`. This has a
  performance cost.
  Args:
    arrays: Single array or list of arrays.
    indices: List of indices in the array that should be included in the output
      batch.
    contiguous: Boolean flag indicating whether the indices are contiguous.
  Returns:
    Slice of data (either single array or list of arrays).
  """
  converted_to_list = False
  if not isinstance(arrays, list):
    converted_to_list = True
    arrays = [arrays]
  if any(tensor_util.is_tf_type(x) for x in arrays):
    if not contiguous:
      entries = [[x[i:i + 1] for i in indices] for x in arrays]
      slices = [array_ops.concat(x, axis=0) for x in entries]
    else:
      slices = [x[indices[0]:indices[-1] + 1] for x in arrays]
  else:
    slices = generic_utils.slice_arrays(arrays, indices)
  if converted_to_list:
    slices = slices[0]
  return slices
def handle_partial_sample_weights(outputs, sample_weights, sample_weight_modes,
                                  check_all_flat=False):
  any_sample_weight = sample_weights is not None and any(
      w is not None for w in sample_weights)
  partial_sample_weight = any_sample_weight and any(
      w is None for w in sample_weights)
  if not any_sample_weight:
    return None, any_sample_weight, partial_sample_weight
  if not partial_sample_weight:
    return sample_weights, any_sample_weight, partial_sample_weight
  if check_all_flat:
    nest.assert_same_structure(
        list_to_tuple(sample_weights),
        list_to_tuple(nest.flatten(sample_weights)))
    nest.assert_same_structure(
        list_to_tuple(outputs),
        list_to_tuple(nest.flatten(outputs)))
    if sample_weight_modes is not None:
      nest.assert_same_structure(
          sample_weight_modes, nest.flatten(sample_weight_modes))
  new_sample_weights = []
  for i, sw in enumerate(sample_weights):
    if sw is None:
      as_numpy = isinstance(outputs[i], np.ndarray)
      output = outputs[i]
      output_shape = output.shape if as_numpy else array_ops.shape(output)
      is_temporal = (
          sample_weight_modes is not None and
          sample_weight_modes[i] == 'temporal')
      sw_shape = (output_shape[0],
                  output_shape[1]) if is_temporal else (output_shape[0],)
      new_sample_weights.append(
          np.ones(sw_shape) if as_numpy else array_ops.ones(sw_shape))
    else:
      new_sample_weights.append(sw)
  return (list_to_tuple(new_sample_weights),
          any_sample_weight, partial_sample_weight)
class RespectCompiledTrainableState(object):
  def __init__(self, model):
    self._model = model
    self._current_trainable_state = None
    self._compiled_trainable_state = None
    self._should_set_trainable = False
  def __enter__(self):
    for layer, trainable in self._compiled_trainable_state.items():
      if (layer in self._current_trainable_state and
          trainable != self._current_trainable_state[layer]):
        self._should_set_trainable = True
        break
    if self._should_set_trainable:
  def __exit__(self, type_arg, value_arg, traceback_arg):
    if self._should_set_trainable:
def get_input_shape_and_dtype(layer):
  """Retrieves input shape and input dtype of layer if applicable.
  Args:
    layer: Layer (or model) instance.
  Returns:
    Tuple (input_shape, input_dtype). Both could be None if the layer
      does not have a defined input shape.
  Raises:
    ValueError: in case an empty Sequential or Functional model is passed.
  """
  def _is_graph_model(layer):
    return ((hasattr(layer, '_is_graph_network') and layer._is_graph_network) or
            layer.__class__.__name__ == 'Sequential')
  while _is_graph_model(layer):
    if not layer.layers:
      raise ValueError('An empty Model cannot be used as a Layer.')
    layer = layer.layers[0]
  if getattr(layer, '_batch_input_shape', None):
    return layer._batch_input_shape, layer.dtype
  return None, None
def get_static_batch_size(layer):
  batch_input_shape, _ = get_input_shape_and_dtype(layer)
  if batch_input_shape is not None:
    return tensor_shape.Dimension(batch_input_shape[0]).value
  return None
def list_to_tuple(maybe_list):
  if isinstance(maybe_list, list):
    return tuple(maybe_list)
  return maybe_list
