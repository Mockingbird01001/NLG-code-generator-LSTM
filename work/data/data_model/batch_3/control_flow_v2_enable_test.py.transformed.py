
import os
os.environ["TF2_BEHAVIOR"] = "1"
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
class ControlFlowV2EnableTest(test.TestCase):
  def testIsEnabled(self):
    self.assertTrue(tf2.enabled())
    self.assertTrue(control_flow_util.ENABLE_CONTROL_FLOW_V2)
if __name__ == "__main__":
  googletest.main()
