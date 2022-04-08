
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
class MultiConnectionNeighborEngineTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    dtype = x.dtype
    e = constant_op.constant(
        np.random.normal(.05, .005, [3, 2, 3, 4]), name="weights", dtype=dtype)
    conv = nn.conv2d(
        input=x,
        filter=e,
        data_format="NCHW",
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    b = constant_op.constant(
        np.random.normal(2.0, 1.0, [1, 4, 1, 1]), name="bias", dtype=dtype)
    t = conv + b
    b = constant_op.constant(
        np.random.normal(5.0, 1.0, [1, 4, 1, 1]), name="bias", dtype=dtype)
    q = conv - b
    edge = self.trt_incompatible_op(q)
    b = constant_op.constant(
        np.random.normal(5.0, 1.0, [1, 4, 1, 1]), name="bias", dtype=dtype)
    d = b + conv
    edge3 = self.trt_incompatible_op(d)
    edge1 = self.trt_incompatible_op(conv)
    t = t - edge1
    q = q + edge
    t = t + q
    t = t + d
    t = t - edge3
    return array_ops.squeeze(t, name="output_0")
  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 3, 7, 5]],
                            [[2, 4, 5, 4]])
  def ExpectedEnginesToBuild(self, run_params):
    return ["TRTEngineOp_000", "TRTEngineOp_001"]
if __name__ == "__main__":
  test.main()
