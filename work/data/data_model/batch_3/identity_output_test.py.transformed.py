
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
class IdentityTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    x1 = math_ops.exp(x)
    x1 = x1 + x
    out1 = array_ops.identity(x1, name='output_0')
    out2 = array_ops.identity(x1, name='output_1')
    iden1 = array_ops.identity(x1)
    out3 = array_ops.identity(iden1, name='output_2')
    return [out1, out2, out3]
  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 32]],
                            [[100, 32]] * 3)
  def ExpectedEnginesToBuild(self, run_params):
    return ['TRTEngineOp_000']
if __name__ == '__main__':
  test.main()
