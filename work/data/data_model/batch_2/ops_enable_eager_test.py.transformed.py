
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
class OpsEnableAndDisableEagerTest(googletest.TestCase):
  def setUp(self):
    ops.enable_eager_execution()
    self.assertTrue(context.executing_eagerly())
    ops.enable_eager_execution()
    self.assertTrue(context.executing_eagerly())
  def tearDown(self):
    ops.disable_eager_execution()
    self.assertFalse(context.executing_eagerly())
    ops.disable_eager_execution()
    self.assertFalse(context.executing_eagerly())
if __name__ == '__main__':
  googletest.main()
