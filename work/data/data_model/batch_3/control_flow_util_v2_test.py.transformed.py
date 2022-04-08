
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.platform import test
class ControlFlowUtilV2Test(test.TestCase):
  def setUp(self):
    self._enable_control_flow_v2_old = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
  def tearDown(self):
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = self._enable_control_flow_v2_old
  def _create_control_flow(self, expect_in_defun):
    def body(i):
      def branch():
        self.assertEqual(control_flow_util_v2.in_defun(), expect_in_defun)
        return i + 1
      return control_flow_ops.cond(constant_op.constant(True),
                                   branch, lambda: 0)
    return control_flow_ops.while_loop(lambda i: i < 4, body,
                                       [constant_op.constant(0)])
  @test_util.run_in_graph_and_eager_modes
  def testInDefun(self):
    self._create_control_flow(False)
    @function.defun
    def defun():
      self._create_control_flow(True)
    defun()
    self.assertFalse(control_flow_util_v2.in_defun())
if __name__ == "__main__":
  test.main()
