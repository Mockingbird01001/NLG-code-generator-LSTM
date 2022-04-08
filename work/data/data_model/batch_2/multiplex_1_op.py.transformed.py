
import tensorflow as tf
from tensorflow.python.platform import resource_loader
_multiplex_1_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("multiplex_1_kernel.so"))
examples_multiplex_dense = _multiplex_1_module.examples1_multiplex_dense
def multiplex(cond, a, b, name=None):
  """Return elements chosen from `a` or `b` depending on `cond`.
  This is similar to `np.where` and `tf.where`, but simplified to only handle
  the case of dense tensors, no optional parameters, no broadcasting, etc..
  >>> multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)>
  Args:
    cond: tf.Tensor of type bool. Where True, yield `a`, otherwise yield `b`.
    a: tf.Tensor with the same type and shape as `b`.
    b: tf.Tensor with the same type and shape as `a`.
    name: An optional name for the op.
  Returns:
    A tf.Tensor with elements from `a` where `cond` is True, and elements
    from `b` elsewhere.
  """
  return examples_multiplex_dense(
      cond=cond, a_values=a, b_values=b, name=name)
