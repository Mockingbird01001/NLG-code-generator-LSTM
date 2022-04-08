
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
class UnaryTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x1, x2):
    x = x1
    q = math_ops.abs(x)
    q = q + 1.0
    q = gen_math_ops.exp(q)
    q = gen_math_ops.log(q)
    q = array_ops.squeeze(q, axis=-2)
    q = math_ops.abs(q)
    q = q + 2.2
    q = gen_math_ops.sqrt(q)
    q = gen_math_ops.rsqrt(q)
    q = math_ops.negative(q)
    q = array_ops.squeeze(q, axis=3)
    q = math_ops.abs(q)
    q = q + 3.0
    a = gen_math_ops.reciprocal(q)
    x = constant_op.constant(np.random.randn(5, 8, 12), dtype=x.dtype)
    q = math_ops.abs(x)
    q = q + 2.0
    q = gen_math_ops.exp(q)
    q = gen_math_ops.log(q)
    q = math_ops.abs(q)
    q = q + 2.1
    q = gen_math_ops.sqrt(q)
    q = gen_math_ops.rsqrt(q)
    q = math_ops.negative(q)
    q = math_ops.abs(q)
    q = q + 4.0
    b = gen_math_ops.reciprocal(q)
    x = x2
    q = math_ops.abs(x)
    q = q + 5.0
    q = gen_math_ops.exp(q)
    q = array_ops.squeeze(q, axis=[-1, -2, 3])
    q = gen_math_ops.log(q)
    q = math_ops.abs(q)
    q = q + 5.1
    q = gen_array_ops.reshape(q, [12, 5, 1, 1, 8, 1, 12])
    q = array_ops.squeeze(q, axis=[5, 2, 3])
    q = gen_math_ops.sqrt(q)
    q = math_ops.abs(q)
    q = q + 5.2
    q = gen_math_ops.rsqrt(q)
    q = math_ops.negative(q)
    q = math_ops.abs(q)
    q = q + 5.3
    c = gen_math_ops.reciprocal(q)
    q = a * b
    q = q / c
    return array_ops.squeeze(q, name="output_0")
  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [[12, 5, 8, 1, 1, 12], [12, 5, 8, 1, 12, 1, 1]],
                            [[12, 5, 8, 12]])
  def ExpectedEnginesToBuild(self, run_params):
    if run_params.dynamic_shape:
      return ["TRTEngineOp_000"]
    else:
      return ["TRTEngineOp_000", "TRTEngineOp_001"]
if __name__ == "__main__":
  test.main()
