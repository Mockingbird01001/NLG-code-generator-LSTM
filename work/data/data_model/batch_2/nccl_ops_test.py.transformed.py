
from functools import partial
import numpy as np
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import nccl_ops
from tensorflow.python.platform import test
def _DeviceTensors(tensors, devices):
  res = []
  for t, d in zip(tensors, devices):
    with ops.device(d):
      res.append(array_ops.identity(t))
  return res
def _NcclAllReduce(nccl_fun, tensors, devices):
  return nccl_fun(_DeviceTensors(tensors, devices))
def _NcclReduce(nccl_fun, tensors, devices):
  receiver = np.random.randint(0, len(devices))
  with ops.device(devices[receiver]):
    return [nccl_fun(_DeviceTensors(tensors, devices))]
def _NcclBroadcast(tensors, devices):
  sender = np.random.randint(0, len(devices))
  with ops.device(devices[sender]):
    tensor = array_ops.identity(tensors[0])
    broadcast = nccl_ops.broadcast(tensor)
  return _DeviceTensors([broadcast] * len(devices), devices)
class NcclTestCase(test.TestCase):
  def _Test(self,
            nccl_reduce,
            numpy_fn,
            device_sets=(['/device:GPU:1', '/device:GPU:2', '/device:GPU:0'],
                         ['/device:GPU:1', '/device:GPU:0'])):
    for dtype in [np.float16, np.float32, np.int32, np.int64, np.float64]:
      with self.test_session():
        for devices in device_sets:
          shape = (3, 4)
          random = (np.random.random_sample(shape) - .5) * 1024
          tensors = []
          for _ in devices:
            tensors.append(random.astype(dtype))
          np_ans = tensors[0]
          for t in tensors[1:]:
            np_ans = numpy_fn(np_ans, t)
          reduce_tensors = nccl_reduce(tensors, devices)
          self.assertNotEmpty(reduce_tensors)
          for r in reduce_tensors:
            self.assertEqual(shape, r.get_shape())
          result_tensors = [array_ops.identity(t) for t in reduce_tensors]
          if not test.is_gpu_available():
            continue
          for t in self.evaluate(result_tensors):
            self.assertAllClose(t, np_ans)
  def _TestGradient(self, nccl_reduce, numpy_fn):
    def _Gradient(tensors, devices):
      inputs = [array_ops.placeholder(t.dtype, t.shape) for t in tensors]
      reduce_tensors = nccl_reduce(inputs, devices)
      losses = _DeviceTensors(tensors, [t.device for t in reduce_tensors])
      grads = gradients.gradients(
          reduce_tensors, inputs, losses, colocate_gradients_with_ops=True)
      return [g for g in grads if g is not None]
    self._Test(_Gradient, numpy_fn)
class AllReduceTest(NcclTestCase):
  def testAllReduce(self):
    self._Test(partial(_NcclAllReduce, nccl_ops.all_sum), lambda x, y: x + y)
    self._Test(partial(_NcclAllReduce, nccl_ops.all_prod), lambda x, y: x * y)
    self._Test(partial(_NcclAllReduce, nccl_ops.all_min), np.minimum)
    self._Test(partial(_NcclAllReduce, nccl_ops.all_max), np.maximum)
  def testAllSumGrad(self):
    self._TestGradient(
        partial(_NcclAllReduce, nccl_ops.all_sum), lambda x, y: x + y)
  def testErrors(self):
    with self.assertRaisesRegex(ValueError, 'Device assignment .* required'):
      nccl_ops.all_sum([array_ops.identity(np.random.random_sample((3, 4)))])
    with self.assertRaisesRegex(ValueError, 'Must pass >0 tensors'):
      nccl_ops.all_sum([])
class SingleReduceTest(NcclTestCase):
  def testSum(self):
    self._Test(partial(_NcclReduce, nccl_ops.reduce_sum), lambda x, y: x + y)
  def testSumGrad(self):
    self._TestGradient(partial(_NcclReduce, nccl_ops.reduce_sum),
                       lambda x, y: x)
class BroadcastTest(NcclTestCase):
  def testBroadcast(self):
    self._Test(_NcclBroadcast, lambda x, y: x)
  def testBroadcastSingleDevice(self):
    self._Test(_NcclBroadcast, lambda x, y: x,
               (['/device:GPU:0', '/device:GPU:0'],))
  def testBroadcastToCpuError(self):
    try:
      self._Test(_NcclBroadcast, lambda x, y: x,
                 (['/device:GPU:0', '/device:CPU:0'],))
    except errors.NotFoundError as e:
      self.assertRegex(
          str(e), "No registered '_NcclBroadcastRecv' OpKernel for CPU devices")
    else:
      if test.is_gpu_available():
        self.fail("Didn't raise NotFoundError trying to broadcast to CPU")
class CombinedTest(NcclTestCase):
  def _Combined(self, tensors, devices):
    all_reduce_tensors = _NcclAllReduce(nccl_ops.all_sum, tensors, devices)
    single_reduce_tensors = _NcclReduce(nccl_ops.reduce_sum, tensors, devices)
    broadcast_tensors = _NcclBroadcast(single_reduce_tensors, devices)
    return all_reduce_tensors + broadcast_tensors
  def testCombined(self):
    self._Test(self._Combined, lambda x, y: x + y)
if __name__ == '__main__':
  test.main()
