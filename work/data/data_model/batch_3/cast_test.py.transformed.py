
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
class CastInt32ToFp32Test(trt_test.TfTrtIntegrationTestBase):
  def _ConstOp(self, shape, dtype):
    return constant_op.constant(np.random.randn(*shape), dtype=dtype)
  def GraphFn(self, x):
    b_f = self._ConstOp((1, 10), dtypes.float32)
    x_f = math_ops.cast(x, dtypes.float32)
    x_f = math_ops.mul(x_f, b_f)
    b_f = self._ConstOp((1, 10), dtypes.float32)
    x_f = math_ops.add(x_f, b_f)
    return array_ops.identity(x_f, name="output_0")
  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.int32, [[1, 10]], [[1, 10]])
  def ExpectedEnginesToBuild(self, run_params):
    if run_params.precision_mode == "FP16":
      return {"TRTEngineOp_000": ["Cast", "Add", "Mul"]}
    else:
      return {"TRTEngineOp_000": ["Add", "Mul"]}
if __name__ == "__main__":
  test.main()
