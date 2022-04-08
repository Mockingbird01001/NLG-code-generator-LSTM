
import pkgutil
import tensorflow as tf
from tensorflow.python import tf2
from tensorflow.python.platform import test
class ModuleTest(test.TestCase):
  def testCanLoadWithPkgutil(self):
    out = pkgutil.find_loader('tensorflow')
    self.assertIsNotNone(out)
  def testDocString(self):
    self.assertIn('TensorFlow', tf.__doc__)
    self.assertNotIn('Wrapper', tf.__doc__)
  def testDict(self):
    tf.nn
    tf.keras
    tf.image
    self.assertIn('nn', tf.__dict__)
    self.assertIn('keras', tf.__dict__)
    self.assertIn('image', tf.__dict__)
  def testName(self):
    self.assertEqual('tensorflow', tf.__name__)
  def testBuiltInName(self):
    if tf2.enabled():
      self.assertEqual(
          'tf.Tensor([1 2 3 4 5 6 7 8 9], shape=(9,), dtype=int32)',
          str(tf.range(1, 10)))
    else:
      self.assertEqual('Tensor("range:0", shape=(9,), dtype=int32)',
                       str(tf.range(1, 10)))
  def testCompatV2HasCompatV1(self):
    tf.compat.v2.compat.v1.keras
  def testSummaryMerged(self):
    tf.summary.image
    if hasattr(tf, '_major_api_version') and tf._major_api_version == 2:
      tf.summary.create_file_writer
    else:
      tf.compat.v1.summary.FileWriter
  def testPythonModuleIsHidden(self):
    self.assertNotIn('python', dir(tf))
if __name__ == '__main__':
  test.main()
