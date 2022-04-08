
from tensorflow.python.framework import versions
from tensorflow.python.platform import test
class VersionTest(test.TestCase):
  def testVersion(self):
    self.assertEqual(type(versions.__version__), str)
    self.assertEqual(type(versions.VERSION), str)
    self.assertRegex(versions.__version__, r'^\d+\.\d+\.(\d+(\-\w+)?|head)$')
    self.assertRegex(versions.VERSION, r'^\d+\.\d+\.(\d+(\-\w+)?|head)$')
  def testGraphDefVersion(self):
    version = versions.GRAPH_DEF_VERSION
    min_consumer = versions.GRAPH_DEF_VERSION_MIN_CONSUMER
    min_producer = versions.GRAPH_DEF_VERSION_MIN_PRODUCER
    for v in version, min_consumer, min_producer:
      self.assertEqual(type(v), int)
    self.assertLessEqual(0, min_consumer)
    self.assertLessEqual(0, min_producer)
    self.assertLessEqual(min_producer, version)
  def testGitAndCompilerVersion(self):
    self.assertEqual(type(versions.__git_version__), str)
    self.assertEqual(type(versions.__compiler_version__), str)
    self.assertEqual(type(versions.GIT_VERSION), str)
    self.assertEqual(type(versions.COMPILER_VERSION), str)
if __name__ == '__main__':
  test.main()
