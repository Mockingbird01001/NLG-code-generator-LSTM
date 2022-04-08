
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common
class TestModule(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def some_function(self, x):
    return self.callee(x)
  @tf.function
  def callee(self, x, n={'foo': 42}):
    return x
if __name__ == '__main__':
  common.do_test(TestModule)
