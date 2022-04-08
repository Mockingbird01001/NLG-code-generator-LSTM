
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
class LRUCacheTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    bias = constant_op.constant(
        np.random.randn(1, 10, 10, 1), dtype=dtypes.float32)
    x = math_ops.add(x, bias)
    x = nn.relu(x)
    return array_ops.identity(x, name="output")
  def GetParams(self):
    dtype = dtypes.float32
    input_dims = [[[1, 10, 10, 2]], [[2, 10, 10, 2]], [[4, 10, 10, 2]],
                  [[2, 10, 10, 2]]]
    expected_output_dims = [[[1, 10, 10, 2]], [[2, 10, 10, 2]], [[4, 10, 10,
                                                                  2]],
                            [[2, 10, 10, 2]]]
    return trt_test.TfTrtIntegrationTestParams(
        graph_fn=self.GraphFn,
        input_specs=[
            tensor_spec.TensorSpec([None, 10, 10, 2], dtypes.float32, "input")
        ],
        output_specs=[
            tensor_spec.TensorSpec([None, 10, 10, 1], dtypes.float32, "output")
        ],
        input_dims=input_dims,
        expected_output_dims=expected_output_dims)
  def ExpectedEnginesToBuild(self, run_params):
    return ["TRTEngineOp_000"]
  def ShouldRunTest(self, run_params):
    return (run_params.dynamic_engine and not trt_test.IsQuantizationMode(
        run_params.precision_mode)), "test dynamic engine and non-INT8"
if __name__ == "__main__":
  test.main()
