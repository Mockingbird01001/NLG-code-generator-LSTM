
import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
class PartialBatchPaddingHandler(object):
  def __init__(self, output_shape):
    self.padded_batch_size = 0
    self.padding_mask = array_ops.zeros(0)
    self.output_shape = output_shape
  def get_real_batch_size(self, dataset_batch):
    if isinstance(dataset_batch, (tuple, list)):
      dataset_batch = dataset_batch[0]
    assert nest.flatten(dataset_batch)
    def _find_any_tensor(batch_features):
      tensors = [
          x for x in nest.flatten(batch_features) if tensor_util.is_tf_type(x)
      ]
      if not tensors:
        raise ValueError('Cannot find any Tensor in features dict.')
      return tensors[0]
    return backend.cast(backend.shape(_find_any_tensor(dataset_batch))[0],
                        dtype='int64')
  def update_mask(self, padding_mask, dataset_batch):
    original_batch_size = self.get_real_batch_size(dataset_batch)
    missing_count = self.padded_batch_size - original_batch_size
    mask = backend.concatenate([array_ops.ones(original_batch_size),
                                array_ops.zeros(missing_count)], axis=0)
    return backend.concatenate([padding_mask, mask], axis=0)
  def pad_batch(self, *dataset_batch_elements):
    def _pad(batch):
      padded_dict_batch = {}
      if isinstance(batch, dict):
        for key, value in batch.items():
          padded_dict_batch[key] = _pad(value)
        return padded_dict_batch
      rank = len(batch.shape)
      assert rank > 0
      missing_count = (self.padded_batch_size -
                       self.get_real_batch_size(batch))
      padding = backend.stack([[0, missing_count]] + [[0, 0]] * (rank - 1))
      return array_ops.pad(batch, padding, 'constant')
    if len(dataset_batch_elements) == 1:
      return _pad(dataset_batch_elements[0])
    batch_elements = []
    for batch_element in dataset_batch_elements:
      batch_elements.append(_pad(batch_element))
    return tuple(batch_elements)
  def apply_mask(self, prediction_result):
    padding_mask = backend.get_value(self.padding_mask)
    assert len(padding_mask.shape) == 1
    if len(self.output_shape) == 1:
      prediction = np.take(prediction_result,
                           np.nonzero(
                               padding_mask[:len(prediction_result)]),
                           axis=0)
      if prediction.shape[0] == 1:
        prediction = np.squeeze(prediction, axis=0)
      return prediction
    else:
      predictions = []
      for i in range(len(self.output_shape)):
        prediction = prediction_result[i]
        prediction = np.take(prediction, np.nonzero(
            padding_mask[:len(prediction)]), axis=0)
        predictions.append(np.squeeze(prediction))
      return predictions
