
import numpy as np
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=["ragged.RaggedTensorValue"])
@dispatch.register_dispatchable_type
class RaggedTensorValue:
  def __init__(self, values, row_splits):
    if not (isinstance(row_splits, (np.ndarray, np.generic)) and
            row_splits.dtype in (np.int64, np.int32) and row_splits.ndim == 1):
      raise TypeError("row_splits must be a 1D int32 or int64 numpy array")
    if not isinstance(values, (np.ndarray, np.generic, RaggedTensorValue)):
      raise TypeError("values must be a numpy array or a RaggedTensorValue")
    if (isinstance(values, RaggedTensorValue) and
        row_splits.dtype != values.row_splits.dtype):
      raise ValueError("row_splits and values.row_splits must have "
                       "the same dtype")
    self._values = values
    self._row_splits = row_splits
  row_splits = property(
      lambda self: self._row_splits,
      doc=
)
  values = property(
      lambda self: self._values,
      doc=
)
  dtype = property(
      lambda self: self._values.dtype,
      doc=
)
  @property
  def flat_values(self):
    rt_values = self.values
    while isinstance(rt_values, RaggedTensorValue):
      rt_values = rt_values.values
    return rt_values
  @property
  def nested_row_splits(self):
    rt_nested_splits = [self.row_splits]
    rt_values = self.values
    while isinstance(rt_values, RaggedTensorValue):
      rt_nested_splits.append(rt_values.row_splits)
      rt_values = rt_values.values
    return tuple(rt_nested_splits)
  @property
  def ragged_rank(self):
    values_is_ragged = isinstance(self._values, RaggedTensorValue)
    return self._values.ragged_rank + 1 if values_is_ragged else 1
  @property
  def shape(self):
    return (self._row_splits.shape[0] - 1,) + (None,) + self._values.shape[1:]
  @property
  def _nested_row_partitions(self):
    return [RowPartition.from_row_splits(rs) for rs in self.nested_row_splits]
  def __str__(self):
    return "<tf.RaggedTensorValue %s>" % self.to_list()
  def __repr__(self):
    return "tf.RaggedTensorValue(values=%r, row_splits=%r)" % (self._values,
                                                               self._row_splits)
  def to_list(self):
    if isinstance(self._values, RaggedTensorValue):
      values_as_list = self._values.to_list()
    else:
      values_as_list = self._values.tolist()
    return [
        values_as_list[self._row_splits[i]:self._row_splits[i + 1]]
        for i in range(len(self._row_splits) - 1)
    ]
