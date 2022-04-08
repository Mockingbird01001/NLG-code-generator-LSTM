
import unittest
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
class MaxBatchSizesTestBase(trt_test.TfTrtIntegrationTestBase):
  @classmethod
  def setUpClass(cls):
    if cls is MaxBatchSizesTestBase:
      raise unittest.SkipTest(
          'MaxBatchSizesTestBase defines base class for other tests.')
    super(MaxBatchSizesTestBase, cls).setUpClass()
  @property
  def tensor_shapes(self):
    return [[1, 512, 1, 1], [64, 2, 2, 2], [32, 4, 2, 2], [16, 8, 2, 2]]
  @property
  def max_batch_sizes(self):
    return [shape[0] for shape in self.tensor_shapes]
  def GetParams(self):
    return self.BuildParams(
        self.GraphFn,
        dtype=dtypes.float32,
        input_shapes=[self.tensor_shapes[0]],
        output_shapes=[self.tensor_shapes[-1]])
  def ShouldRunTest(self, run_params):
    return (not run_params.dynamic_engine, 'test static engine only.')
  def GetMaxBatchSize(self, run_params):
    if run_params.dynamic_engine:
      return None
    return min(self.max_batch_sizes)
  def ExpectedEnginesToBuild(self, run_params):
    return [
        f'TRTEngineOp_{seq_id:03d}'
        for seq_id in range(len(self.max_batch_sizes))
    ]
  def ExpectedMaxBatchSizes(self, run_params):
    return self.max_batch_sizes
class AnnotateMaxBatchSizesTest(MaxBatchSizesTestBase):
  def GraphFn(self, inp):
    tensor = inp * 2.0
    tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[1][1:])
    with ops.get_default_graph()._attr_scope({
        '_tftrt_op_max_batch_size':
            attr_value_pb2.AttrValue(i=self.max_batch_sizes[1])
    }):
      tensor = tensor + 3.0
    tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[2][1:])
    with ops.get_default_graph()._attr_scope({
        '_tftrt_op_max_batch_size':
            attr_value_pb2.AttrValue(i=self.max_batch_sizes[2])
    }):
      tensor = tensor * 4.0
    tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[3][1:])
    with ops.get_default_graph()._attr_scope({
        '_tftrt_op_max_batch_size':
            attr_value_pb2.AttrValue(i=self.max_batch_sizes[3])
    }):
      tensor += tensor + 5.0
    return array_ops.identity(tensor, name='output_0')
class StaticBatchSizeTest(MaxBatchSizesTestBase):
  def GraphFn(self, inp):
    tensor = inp * 2.0
    tensor = array_ops.reshape(tensor, self.tensor_shapes[1])
    tensor = tensor + 3.0
    tensor = array_ops.reshape(tensor, self.tensor_shapes[2])
    tensor = tensor * 4.0
    tensor = array_ops.reshape(tensor, self.tensor_shapes[3])
    tensor += tensor + 5.0
    return array_ops.identity(tensor, name='output_0')
if __name__ == '__main__':
  test.main()
