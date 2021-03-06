
from absl.testing import parameterized
import numpy as np
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
JOB_NAME = "remote_device"
ALT_JOB_NAME = "alt_remote_device"
def get_server_def(job_name, local_server_port, remote_server_addresses,
                   task_index):
  cluster_def = cluster_pb2.ClusterDef()
  job_def = cluster_def.job.add()
  job_def.name = job_name
  job_def.tasks[0] = "localhost:%d" % local_server_port
  for i, remote_server_address in enumerate(remote_server_addresses, start=1):
    job_def.tasks[i] = remote_server_address
  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_def,
      job_name=job_name,
      task_index=task_index,
      protocol="grpc")
  return server_def
class RemoteExecutionTest(test.TestCase, parameterized.TestCase):
    super(RemoteExecutionTest, self).__init__(methodName)
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server2 = server_lib.Server.create_local_server()
    self._cached_server1_target = self._cached_server1.target[len("grpc://"):]
    self._cached_server2_target = self._cached_server2.target[len("grpc://"):]
  def setUp(self):
    super(RemoteExecutionTest, self).setUp()
    local_port = pywrap_tfe.TF_PickUnusedPortOrDie()
    context.set_server_def(
        server_def=get_server_def(
            JOB_NAME,
            local_server_port=local_port,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))
  def tearDown(self):
    super(RemoteExecutionTest, self).tearDown()
    ops.device(None).__enter__()
    context._reset_context()
  @test_util.run_in_async_and_sync_mode
  @test_util.run_gpu_only
  def testGpuToRemoteCopy(self):
    x1 = array_ops.ones([2, 2]).gpu()
    with ops.device("/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
    np.testing.assert_array_equal(x1.numpy(), x2.numpy())
  @test_util.run_in_async_and_sync_mode
  @test_util.run_gpu_only
  def testGpuToRemoteOp(self):
    with ops.device("gpu:0"):
      x = array_ops.ones([2, 2])
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      y = math_ops.matmul(x, x)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())
  @test_util.run_in_async_and_sync_mode
  def testDefunMatmul(self):
    mm_defun = function.defun(math_ops.matmul)
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    with ops.device("job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME):
      x2 = array_ops.ones([2, 2])
      y = mm_defun(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())
  @test_util.run_in_async_and_sync_mode
  def testSimpleMatmul(self):
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    with ops.device("job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME):
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())
  def testEagerPyFuncPlacement(self):
    if not ops.executing_eagerly_outside_functions():
      return
    def f(x):
      return math_ops.square(x)
    with ops.device("/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      const_op = constant_op.constant(3.0, dtype=dtypes.float32)
      py_func_op = script_ops.eager_py_func(
          func=f, inp=[const_op], Tout=dtypes.float32)
      self.assertEqual(py_func_op.device,
                       "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME)
      self.assertEqual(self.evaluate(py_func_op), 9.0)
  @test_util.run_in_async_and_sync_mode
  def testSimpleWeightRead(self):
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      w = resource_variable_ops.ResourceVariable([[2.0]])
      loss = w * w
    np.testing.assert_array_equal([[4.0]], loss.numpy())
  @test_util.run_in_async_and_sync_mode
  def testTapeWeightRead(self):
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      w = resource_variable_ops.ResourceVariable([[3.0]])
      with backprop.GradientTape() as tape:
        loss = w * w
      grad = tape.gradient(loss, w)
    np.testing.assert_array_equal([[9.0]], loss.numpy())
    np.testing.assert_array_equal([[6.0]], grad.numpy())
  @test_util.run_in_async_and_sync_mode
  def testServerDefChanged(self):
    context.set_server_def(
        server_def=get_server_def(
            ALT_JOB_NAME,
            local_server_port=0,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % ALT_JOB_NAME):
      x1 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())
    context.set_server_def(
        server_def=get_server_def(
            JOB_NAME,
            local_server_port=0,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())
  @test_util.run_in_async_and_sync_mode
  def testConnectToRemoteServer(self):
    context._reset_context()
    remote.connect_to_remote_host(self._cached_server1_target)
    with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
      x1 = array_ops.ones([2, 2])
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())
  @test_util.run_in_async_and_sync_mode
  def testContextDeviceUpdated(self):
    with ops.device("cpu:0"):
      x1 = array_ops.ones([2, 2])
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())
    self.assertEqual(y.device,
                     "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME)
if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
