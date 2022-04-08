
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common
class ReferencesParent(tf.Module):
  def __init__(self, parent):
    super(ReferencesParent, self).__init__()
    self.parent = parent
    self.my_variable = tf.Variable(3.)
class TestModule(tf.Module):
  def __init__(self):
    super(TestModule, self).__init__()
    self.child = ReferencesParent(self)
if __name__ == '__main__':
  common.do_test(TestModule)
