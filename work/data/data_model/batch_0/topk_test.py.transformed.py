
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
class TopKTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    k = 5
    k_tensor = constant_op.constant(k, dtype=dtypes.int32, name="Const")
    values, indices = nn_ops.top_k(x, k_tensor, name="TopK")
    values = array_ops.identity(values, name="output_0")
    indices = array_ops.identity(indices, name="output_1")
    return values, indices
  def GetParams(self):
    k = 5
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 100]],
                            [[100, k], [100, k]])
  def ExpectedEnginesToBuild(self, run_params):
    return {"TRTEngineOp_000": ["Const", "TopK"]}
class TopKOutputTypeTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    k = 5
    k_tensor = constant_op.constant(k, dtype=dtypes.int32, name="Const")
    values, indices = nn_ops.top_k(x, k_tensor, name="TopK")
    indices = array_ops.reshape(indices, [100, 1, 5], name="Reshape")
    values = array_ops.identity(values, name="output_0")
    indices = array_ops.identity(indices, name="output_1")
    return values, indices
  def GetParams(self):
    k = 5
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 100]],
                            [[100, k], [100, 1, k]])
  def ExpectedEnginesToBuild(self, run_params):
    return {"TRTEngineOp_000": ["Const", "TopK", "Reshape", "Reshape/shape"]}
if __name__ == "__main__":
  test.main()
