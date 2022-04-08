
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
class ExcludeUnsupportedInt32Test(trt_test.TfTrtIntegrationTestBase):
  def _ConstOp(self, shape, dtype):
    return constant_op.constant(np.random.randn(*shape), dtype=dtype)
  def GraphFn(self, x):
    dtype = x.dtype
    b = self._ConstOp((4, 10), dtype)
    x = math_ops.matmul(x, b)
    b = self._ConstOp((10,), dtype)
    x = nn.bias_add(x, b)
    return array_ops.identity(x, name='output_0')
  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.int32, [[100, 4]], [[100, 10]])
  def setUp(self):
    self.DisableNonTrtOptimizers()
  def GetMaxBatchSize(self, run_params):
    if run_params.dynamic_engine:
      return None
    return 100
  def ExpectedEnginesToBuild(self, run_params):
    return []
class CalibrationInt32Support(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, inp):
    inp_transposed = array_ops.transpose(inp, [0, 3, 2, 1], name='transpose_0')
    return array_ops.identity(inp_transposed, name='output_0')
  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.int32, [[3, 4, 5, 6]],
                            [[3, 6, 5, 4]])
  def ShouldRunTest(self, run_params):
    return trt_test.IsQuantizationWithCalibration(
        run_params), 'test calibration and INT8'
  def ExpectedEnginesToBuild(self, run_params):
    return ['TRTEngineOp_000']
if __name__ == '__main__':
  test.main()
