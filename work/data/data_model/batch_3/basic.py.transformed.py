
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common
class TestModule(tf.Module):
  def __init__(self):
    super(TestModule, self).__init__()
    self.v42 = tf.Variable(42.0)
    self.c43 = tf.constant(43.0)
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def some_function(self, x):
    return x + self.v42 + self.c43
if __name__ == '__main__':
  common.do_test(TestModule)
