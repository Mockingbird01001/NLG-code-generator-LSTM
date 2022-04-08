
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common
class TestModule(tf.Module):
  def __init__(self):
    super(TestModule, self).__init__()
    self.v = tf.Variable(42.0)
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def callee(self, x):
    return x, self.v
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def caller(self, x):
    return self.callee(x)
if __name__ == '__main__':
  common.do_test(TestModule)
