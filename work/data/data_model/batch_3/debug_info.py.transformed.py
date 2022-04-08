
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common
class TestModule(tf.Module):
  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32)
  ])
  def some_function(self, x, y):
    return x + y
if __name__ == '__main__':
  common.do_test(TestModule, show_debug_info=True)
