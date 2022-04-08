
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
def _GraphFn(x, add_quantization_nodes):
  def _Quantize(x, r):
    if add_quantization_nodes:
      x = gen_array_ops.fake_quant_with_min_max_vars(x, -r, r)
    return x
  x = _Quantize(x, 10.0)
  x = x + 5
  x = _Quantize(x, 15.0)
  x = x - 5
  x = _Quantize(x, 10.0)
  x = x * 0.1
  x = _Quantize(x, 1.0)
  w = constant_op.constant(np.ones((8, 1)), dtype=dtypes.float32)
  x = math_ops.matmul(x, w)
  x = _Quantize(x, 10.0)
  return array_ops.identity(x, name="output_0")
def _GetParams(self):
  return self.BuildParams(self.GraphFn, dtypes.float32, [[8, 8]], [[8, 1]])
class QuantizationMissingAllRangesTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    return _GraphFn(x, add_quantization_nodes=False)
  def GetParams(self):
    return _GetParams(self)
  def ShouldRunTest(self, run_params):
    return (trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.convert_online and not run_params.dynamic_engine
           ), "test static engine, offline conversion and INT8"
  def ExpectedEnginesToBuild(self, run_params):
    return ["TRTEngineOp_000"]
class QuantizationWithRangesTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    return _GraphFn(x, add_quantization_nodes=True)
  def GetParams(self):
    return _GetParams(self)
  def ShouldRunTest(self, run_params):
    return (trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.convert_online), "test offline conversion and INT8"
  def ExpectedEnginesToBuild(self, run_params):
    return ["TRTEngineOp_000"]
  def ExpectedAbsoluteTolerance(self, run_params):
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01
  def ExpectedRelativeTolerance(self, run_params):
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01
class NonQuantizedPrecisionsWithRangesTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    return _GraphFn(x, add_quantization_nodes=True)
  def GetParams(self):
    return _GetParams(self)
  def ShouldRunTest(self, run_params):
    return not trt_test.IsQuantizationMode(
        run_params.precision_mode), "test non-INT8"
  def ExpectedEnginesToBuild(self, run_params):
    return ["TRTEngineOp_000", "TRTEngineOp_001"]
  def ExpectedAbsoluteTolerance(self, run_params):
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01
  def ExpectedRelativeTolerance(self, run_params):
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01
if __name__ == "__main__":
  test.main()
