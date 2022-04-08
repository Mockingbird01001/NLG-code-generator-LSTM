
import functools
from absl.testing import parameterized
from tensorflow.compiler.xla.service import hlo_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_strategy_util
class RecomputeGradMemoryTest(test.TestCase, parameterized.TestCase):
  def _get_device_type(self):
    for accelerator in ["GPU", "TPU"]:
      if config.list_physical_devices(accelerator):
        return accelerator
    return "CPU"
  def _grad(self, test_func, argnums=0):
    def _f(*params):
      with backprop.GradientTape() as tape:
        tape.watch(params)
        output = test_func(*params)
      return tape.gradient(output, params[argnums])
    return _f
  @parameterized.named_parameters(
      (f"_{mode}", mode) for mode in ["eager", "graph"])
  @test_util.run_v2_only
  def testRecomputeGradNonXla(self, mode):
    device_type = self._get_device_type()
    device_name = f"{device_type}:0"
    if device_type == "TPU":
      self.skipTest("XLA is required for TPU.")
    if device_type == "CPU":
      self.skipTest("b/185371422: get_memory_info does't support CPU yet.")
    config.reset_memory_stats(device_name)
    base_memory = config.get_memory_info(device_name)["current"]
    n = 500
    with ops.device(device_name):
      a = array_ops.ones((n, n), dtype=dtypes.float16)
    def f(x):
      for _ in range(5):
        x = math_ops.matmul(x, x)
      return x
    def g(f, x):
      for _ in range(5):
        x = f(x)
      return x[0][0]
    def run(test_func):
      with ops.device(device_name):
        if mode == "eager":
          return self._grad(test_func)(a)
        else:
          return def_function.function(self._grad(test_func))(a)
    f_no_recompute = functools.partial(g, f)
    f_recompute = functools.partial(g, custom_gradient.recompute_grad(f))
    run(f_no_recompute)
    peak_memory_no_recompute = (
        config.get_memory_info(device_name)["peak"] - base_memory)
    config.reset_memory_stats(device_name)
    run(f_recompute)
    peak_memory_recompute = (
        config.get_memory_info(device_name)["peak"] - base_memory)
    self.assertGreaterEqual(peak_memory_no_recompute, 2 * n * n * 5 * 5)
    self.assertGreaterEqual(peak_memory_recompute, 2 * n * n * 5 * 2)
    self.assertLess(peak_memory_recompute, 2 * n * n * 5 * 3)
    res_no_recompute = run(f_no_recompute)
    res_recompute = run(f_recompute)
    self.assertAllClose(res_no_recompute, res_recompute)
  @test_util.run_v2_only
  def testRecomputeGradXla(self):
    device_type = self._get_device_type()
    device_name = f"{device_type}:0"
    if device_type == "TPU":
      tpu_strategy_util.initialize_tpu_system()
    n = 500
    with ops.device(device_name):
    def f(x):
      for _ in range(5):
        x = math_ops.matmul(x, x)
      return x
    def g(f, x):
      for _ in range(5):
        x = f(x)
      return x[0][0]
    def get_peak_memory(test_func):
      test_func = def_function.function(self._grad(test_func), jit_compile=True)
      hlo_proto_serialized = test_func.experimental_get_compiler_ir(a)(
          stage="optimized_hlo_proto_serialized")
      hlo_proto = hlo_pb2.HloProto.FromString(hlo_proto_serialized)
      allocations = hlo_proto.buffer_assignment.buffer_allocations
      return sum(getattr(allocation, "size") for allocation in allocations)
    f_no_recompute = functools.partial(g, f)
    f_recompute = functools.partial(g, custom_gradient.recompute_grad(f))
    peak_memory_no_recompute = get_peak_memory(f_no_recompute)
    peak_memory_recompute = get_peak_memory(f_recompute)
    self.assertGreaterEqual(peak_memory_no_recompute, 2 * n * n * 5 * 5)
    self.assertGreaterEqual(peak_memory_recompute, 2 * n * n * 5 * 2)
    self.assertLess(peak_memory_recompute, 2 * n * n * 5 * 3)
    with ops.device(device_name):
      res_recompute = f_recompute(a)
      res_no_recompute = f_no_recompute(a)
    self.assertAllClose(res_recompute, res_no_recompute)
if __name__ == "__main__":
  googletest.main()
