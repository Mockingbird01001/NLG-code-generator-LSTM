
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common
def mnist_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))
  return model
class TestModule(tf.Module):
  def __init__(self):
    super(TestModule, self).__init__()
    self.model = mnist_model()
  @tf.function(input_signature=[
      tf.TensorSpec([1, 28, 28, 1], tf.float32),
  ])
  def my_predict(self, x):
    return self.model(x)
if __name__ == '__main__':
  common.do_test(TestModule, exported_names=['my_predict'])
