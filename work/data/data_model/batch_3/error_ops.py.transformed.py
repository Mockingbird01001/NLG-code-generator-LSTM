
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export
@tf_export("data.experimental.ignore_errors")
def ignore_errors(log_warning=False):
  """Creates a `Dataset` from another `Dataset` and silently ignores any errors.
  Use this transformation to produce a dataset that contains the same elements
  as the input, but silently drops any elements that caused an error. For
  example:
  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1., 2., 0., 4.])
  InvalidArgumentError.
  dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, "error"))
  dataset =
  ```
  Args:
     log_warning: (Optional.) A 'tf.bool' scalar indicating whether ignored
      errors should be logged to stderr. Defaults to 'False'.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return _IgnoreErrorsDataset(dataset, log_warning)
  return _apply_fn
class _IgnoreErrorsDataset(dataset_ops.UnaryUnchangedStructureDataset):
  def __init__(self, input_dataset, log_warning):
    self._input_dataset = input_dataset
    variant_tensor = (
        gen_experimental_dataset_ops.ignore_errors_dataset(
            log_warning=log_warning,
            **self._flat_structure))
    super(_IgnoreErrorsDataset, self).__init__(input_dataset, variant_tensor)
