
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
class MatchingFilesDataset(dataset_ops.DatasetSource):
  def __init__(self, patterns):
    self._patterns = ops.convert_to_tensor(
        patterns, dtype=dtypes.string, name="patterns")
    variant_tensor = ged_ops.matching_files_dataset(self._patterns)
    super(MatchingFilesDataset, self).__init__(variant_tensor)
  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)
