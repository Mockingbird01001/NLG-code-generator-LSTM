
import os
import tempfile
import six
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import tf_upgrade
class TestUpgrade(test_util.TensorFlowTestCase):
  def _upgrade(self, old_file_text):
    in_file = six.StringIO(old_file_text)
    out_file = six.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade.TFAPIChangeSpec())
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()
  def testParseError(self):
    _, report, unused_errors, unused_new_text = self._upgrade(
        "import tensorflow as tf\na + \n")
    self.assertNotEqual(six.ensure_str(report).find("Failed to parse"), -1)
  def testReport(self):
    text = "tf.mul(a, b)\n"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    self.assertTrue(
        six.ensure_str(report).find(
            "Renamed function `tf.mul` to `tf.multiply`"))
  def testRename(self):
    text = "tf.mul(a, tf.sub(b, c))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.multiply(a, tf.subtract(b, c))\n")
  def testRenamePack(self):
    text = "tf.pack(a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.stack(a)\n")
    text = "tf.unpack(a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.unstack(a)\n")
  def testReorder(self):
    text = "tf.concat(a, b)\ntf.split(a, b, c)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(axis=a, values=b)\n"
                     "tf.split(axis=a, num_or_size_splits=b, value=c)\n")
  def testConcatReorderWithKeywordArgs(self):
    text = "tf.concat(concat_dim=a, values=b)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(axis=a, values=b)\n")
    text = "tf.concat(values=b, concat_dim=a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(values=b, axis=a)\n")
    text = "tf.concat(a, values=b)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(axis=a, values=b)\n")
  def testConcatReorderNested(self):
    text = "tf.concat(a, tf.concat(c, d))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text, "tf.concat(axis=a, values=tf.concat(axis=c, values=d))\n")
  def testInitializers(self):
    text = ("tf.zeros_initializer;tf.zeros_initializer ()\n"
            "tf.ones_initializer;tf.ones_initializer ()\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text, "tf.zeros_initializer();tf.zeros_initializer ()\n"
                  "tf.ones_initializer();tf.ones_initializer ()\n")
  def testKeyword(self):
    text = "tf.reduce_any(a, reduction_indices=[1, 2])\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.reduce_any(a, axis=[1, 2])\n")
  def testComplexExpression(self):
    text = "(foo + bar)[a].word()"
    _ = self._upgrade(text)
  def testReverse(self):
    text = "tf.reverse(a, b)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, new_text)
    self.assertIn("tf.reverse requires manual check", errors[0])
  def testListComprehension(self):
      _, unused_report, errors, new_text = self._upgrade(input)
      self.assertEqual(new_text, output)
    _test("tf.concat(0,  \t[x for x in y])\n",
          "tf.concat(axis=0,  \tvalues=[x for x in y])\n")
    _test("tf.concat(0,[x for x in y])\n",
          "tf.concat(axis=0,values=[x for x in y])\n")
    _test("tf.concat(0,[\nx for x in y])\n",
          "tf.concat(axis=0,values=[\nx for x in y])\n")
    _test("tf.concat(0,[\n \tx for x in y])\n",
          "tf.concat(axis=0,values=[\n \tx for x in y])\n")
class TestUpgradeFiles(test_util.TensorFlowTestCase):
  def testInplace(self):
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "tf.mul(a, b)\n"
    upgraded = "tf.multiply(a, b)\n"
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade.TFAPIChangeSpec())
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)
if __name__ == "__main__":
  test_lib.main()
