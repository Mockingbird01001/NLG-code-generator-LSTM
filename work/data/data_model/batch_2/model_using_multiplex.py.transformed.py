
import tensorflow as tf
def _get_example_tensors():
  cond = tf.constant([True, False, True, False, True], dtype=bool)
  a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
  b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
  return cond, a, b
def save(multiplex_op, path):
  example_cond, example_a, example_b = _get_example_tensors()
  class UseMultiplex(tf.Module):
    @tf.function(input_signature=[
        tf.TensorSpec.from_tensor(example_cond),
        tf.TensorSpec.from_tensor(example_a),
        tf.TensorSpec.from_tensor(example_b)
    ])
    def use_multiplex(self, cond, a, b):
      return multiplex_op(cond, a, b)
  model = UseMultiplex()
  tf.saved_model.save(
      model,
      path,
      signatures=model.use_multiplex.get_concrete_function(
          tf.TensorSpec.from_tensor(example_cond),
          tf.TensorSpec.from_tensor(example_a),
          tf.TensorSpec.from_tensor(example_b)))
def load_and_use(path):
  """Load and used a model that was previously created by `save()`.
  Args:
    path: Directory to load model from, typically the same directory that was
      used by save().
  Returns:
    A tensor that is the result of using the multiplex op that is
    tf.constant([1, 20, 3, 40, 5], dtype=tf.int64).
  """
  example_cond, example_a, example_b = _get_example_tensors()
  restored = tf.saved_model.load(path)
  return restored.use_multiplex(example_cond, example_a, example_b)
