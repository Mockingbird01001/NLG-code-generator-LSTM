
import sys
from absl import app
import tensorflow.compat.v2 as tf
if hasattr(tf, 'enable_v2_behavior'):
  tf.enable_v2_behavior()
class TestGraphDebugInfo(object):
  def testConcreteFunctionDebugInfo(self):
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
    def model(x):
      return y + y
    func = model.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func], model)
    converter.convert()
def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  try:
    TestGraphDebugInfo().testConcreteFunctionDebugInfo()
    sys.stdout.write('testConcreteFunctionDebugInfo')
    sys.stdout.write(str(e))
if __name__ == '__main__':
  app.run(main)
