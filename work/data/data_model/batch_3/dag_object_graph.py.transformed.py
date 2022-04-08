
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common
class Child(tf.Module):
  def __init__(self):
    super(Child, self).__init__()
    self.my_variable = tf.Variable(3.)
class TestModule(tf.Module):
  def __init__(self):
    super(TestModule, self).__init__()
    self.child1 = Child()
    self.child2 = self.child1
if __name__ == '__main__':
  common.do_test(TestModule)
