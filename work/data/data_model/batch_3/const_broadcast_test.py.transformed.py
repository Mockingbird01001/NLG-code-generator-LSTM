
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
class ConstBroadcastTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, x):
    dtype = x.dtype
    filt1 = constant_op.constant(
        0.3, shape=(3, 3, 2, 1), dtype=dtype, name='filt1')
    y1 = nn.conv2d(x, filt1, strides=[1, 1, 1, 1], padding='SAME', name='y1')
    z1 = nn.relu(y1, name='z1')
    filt2 = constant_op.constant(
        0.3, shape=(3, 3, 1, 1), dtype=dtype, name='filt2')
    y2 = nn.conv2d(z1, filt2, strides=[1, 1, 1, 1], padding='SAME', name='y2')
    z2 = nn.relu(y2, name='z')
    filt3 = constant_op.constant(
        0.3, shape=(3, 3, 1, 1), dtype=dtype, name='filt3')
    y3 = nn.conv2d(z2, filt3, strides=[1, 1, 1, 1], padding='SAME', name='y3')
    return nn.relu(y3, name='output_0')
  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[5, 12, 12, 2]],
                            [[5, 12, 12, 1]])
  def ExpectedEnginesToBuild(self, run_params):
    return ['TRTEngineOp_000']
  def ExpectedAbsoluteTolerance(self, run_params):
    return 1.e-04 if run_params.precision_mode == 'FP32' else 1.e-02
  def ExpectedRelativeTolerance(self, run_params):
    return 1.e-04 if run_params.precision_mode == 'FP32' else 1.e-02
if __name__ == '__main__':
  test.main()
