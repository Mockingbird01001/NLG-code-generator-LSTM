
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common
class TestModule(tf.Module):
  def __init__(self):
    super(TestModule, self).__init__()
    self.v0 = tf.Variable([0.], shape=tf.TensorShape(None))
    self.v1 = tf.Variable([0., 1.], shape=[None])
if __name__ == '__main__':
  common.do_test(TestModule, exported_names=[])
