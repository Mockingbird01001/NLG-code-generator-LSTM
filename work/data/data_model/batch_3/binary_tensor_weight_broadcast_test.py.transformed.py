
import os
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test
class BinaryTensorWeightBroadcastTest(trt_test.TfTrtIntegrationTestBase):
  def _ConstOp(self, shape):
    return constant_op.constant(np.random.randn(*shape), dtype=dtypes.float32)
  def GraphFn(self, x):
    for weights_shape in [
    ]:
      a = self._ConstOp(weights_shape)
      f = x + a
      x = self.trt_incompatible_op(f)
      a = self._ConstOp(weights_shape)
      f = a + x
      x = self.trt_incompatible_op(f)
    return gen_array_ops.reshape(x, [5, -1], name="output_0")
  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[10, 24, 24, 20]],
                            [[5, 23040]])
  def ExpectedEnginesToBuild(self, run_params):
    num_engines = 17 if run_params.dynamic_shape else 16
    return [f"TRTEngineOp_{i:03d}" for i in range(num_engines)]
  def setUp(self):
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"
if __name__ == "__main__":
  test.main()
