
import functools
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, "Use `tf.data.Dataset.random(...)`.")
@tf_export("data.experimental.RandomDataset", v1=[])
class RandomDatasetV2(dataset_ops.RandomDataset):
@deprecation.deprecated(None, "Use `tf.data.Dataset.random(...)`.")
@tf_export(v1=["data.experimental.RandomDataset"])
class RandomDatasetV1(dataset_ops.DatasetV1Adapter):
  @functools.wraps(RandomDatasetV2.__init__)
  def __init__(self, seed=None):
    wrapped = RandomDatasetV2(seed)
    super(RandomDatasetV1, self).__init__(wrapped)
if tf2.enabled():
  RandomDataset = RandomDatasetV2
else:
  RandomDataset = RandomDatasetV1
