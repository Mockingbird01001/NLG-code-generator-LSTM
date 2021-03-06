
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
class ControlFlowV2TogglesTest(test.TestCase):
  def testOutputAllIntermediates(self):
    self.assertIsNone(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_v2_toggles.output_all_intermediates(True)
    self.assertTrue(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_v2_toggles.output_all_intermediates(False)
    self.assertFalse(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_v2_toggles.output_all_intermediates(None)
    self.assertIsNone(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
if __name__ == '__main__':
  googletest.main()
