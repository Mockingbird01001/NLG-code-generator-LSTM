
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import test
class XlaTestCaseTestCase(test.TestCase):
  def testManifestEmptyLineDoesNotCatchAll(self):
    manifest =
    disabled_regex, _ = xla_test.parse_disabled_manifest(manifest)
    self.assertEqual(disabled_regex, "testCaseOne")
  def testManifestWholeLineCommentDoesNotCatchAll(self):
testCaseOne
testCaseTwo
"""
    disabled_regex, _ = xla_test.parse_disabled_manifest(manifest)
    self.assertEqual(disabled_regex, "testCaseOne|testCaseTwo")
if __name__ == "__main__":
  test.main()
