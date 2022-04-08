
import tensorflow.compat.v2 as tf
np = tf.experimental.numpy
NUM_CLASSES = 3
INPUT_SIZE = 10
HIDDEN_UNITS = 10
class MLP:
  """MLP model.
  T = Relu(Add(MatMul(A, B), C))
  R = Relu(Add(MatMul(T, D), E))
  """
  def __init__(self, num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
               hidden_units=HIDDEN_UNITS):
    self.w1 = np.random.uniform(size=[input_size, hidden_units]).astype(
        np.float32)
    self.w2 = np.random.uniform(size=[hidden_units, num_classes]).astype(
        np.float32)
    self.b1 = np.random.uniform(size=[1, hidden_units]).astype(
        np.float32)
    self.b2 = np.random.uniform(size=[1, num_classes]).astype(
        np.float32)
  def inference(self, inputs):
    return self._forward(inputs, self.w1, self.w2, self.b1, self.b2)
  def _forward(self, x, w1, w2, b1, b2):
    x = np.maximum(np.matmul(x, w1) + b1, 0.)
    x = np.maximum(np.matmul(x, w2) + b2, 0.)
    return x
