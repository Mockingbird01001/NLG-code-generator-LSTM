
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, "Use `tf.data.Dataset.unique(...)")
@tf_export("data.experimental.unique")
def unique():
  """Creates a `Dataset` from another `Dataset`, discarding duplicates.
  Use this transformation to produce a dataset that contains one instance of
  each unique element in the input. For example:
  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])
  ```
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.unique()
  return _apply_fn
