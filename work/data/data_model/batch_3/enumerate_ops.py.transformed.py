
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, "Use `tf.data.Dataset.enumerate()`.")
@tf_export("data.experimental.enumerate_dataset")
def enumerate_dataset(start=0):
  """A transformation that enumerates the elements of a dataset.
  It is similar to python's `enumerate`.
  For example:
  ```python
  a = { 1, 2, 3 }
  b = { (7, 8), (9, 10) }
  a.apply(tf.data.experimental.enumerate_dataset(start=5))
  => { (5, 1), (6, 2), (7, 3) }
  b.apply(tf.data.experimental.enumerate_dataset())
  => { (0, (7, 8)), (1, (9, 10)) }
  ```
  Args:
    start: A `tf.int64` scalar `tf.Tensor`, representing the start value for
      enumeration.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.enumerate(start)
  return _apply_fn
