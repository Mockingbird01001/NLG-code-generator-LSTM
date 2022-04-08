
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export("data.experimental.TFRecordWriter")
@deprecation.deprecated(
    None, "To write TFRecords to disk, use `tf.io.TFRecordWriter`. To save "
    "and load the contents of a dataset, use `tf.data.experimental.save` "
    "and `tf.data.experimental.load`")
class TFRecordWriter(object):
  """Writes a dataset to a TFRecord file.
  The elements of the dataset must be scalar strings. To serialize dataset
  elements as strings, you can use the `tf.io.serialize_tensor` function.
  ```python
  dataset = tf.data.Dataset.range(3)
  dataset = dataset.map(tf.io.serialize_tensor)
  writer = tf.data.experimental.TFRecordWriter("/path/to/file.tfrecord")
  writer.write(dataset)
  ```
  To read back the elements, use `TFRecordDataset`.
  ```python
  dataset = tf.data.TFRecordDataset("/path/to/file.tfrecord")
  dataset = dataset.map(lambda x: tf.io.parse_tensor(x, tf.int64))
  ```
  To shard a `dataset` across multiple TFRecord files:
  ```python
  def reduce_func(key, dataset):
    filename = tf.strings.join([PATH_PREFIX, tf.strings.as_string(key)])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)
  dataset = dataset.enumerate()
  dataset = dataset.apply(tf.data.experimental.group_by_window(
    lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max
  ))
  for _ in dataset:
    pass
  ```
  """
  def __init__(self, filename, compression_type=None):
    """Initializes a `TFRecordWriter`.
    Args:
      filename: a string path indicating where to write the TFRecord data.
      compression_type: (Optional.) a string indicating what type of compression
        to use when writing the file. See `tf.io.TFRecordCompressionType` for
        what types of compression are available. Defaults to `None`.
    """
    self._filename = ops.convert_to_tensor(
        filename, dtypes.string, name="filename")
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
  def write(self, dataset):
    if not isinstance(dataset, dataset_ops.DatasetV2):
      raise TypeError(
          f"Invalid `dataset.` Expected a `tf.data.Dataset` object but got "
          f"{type(dataset)}."
      )
    if not dataset_ops.get_structure(dataset).is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.string)):
      raise TypeError(
          f"Invalid `dataset`. Expected a`dataset` that produces scalar "
          f"`tf.string` elements, but got a dataset which produces elements "
          f"with shapes {dataset_ops.get_legacy_output_shapes(dataset)} and "
          f"types {dataset_ops.get_legacy_output_types(dataset)}.")
    dataset = dataset._apply_debug_options()
    return gen_experimental_dataset_ops.dataset_to_tf_record(
        dataset._variant_tensor, self._filename, self._compression_type)
