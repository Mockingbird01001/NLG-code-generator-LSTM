
import gc
import time
import tensorflow as tf
from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.eager import context
from tensorflow.python.profiler import trace
NUM_ITERATIONS = 30000
def _run_benchmark(func, num_iters, execution_mode=None):
  ctx = context.context()
  with context.execution_mode(execution_mode):
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in range(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()
    return end - start
class KpiBenchmarks(benchmarks_test_base.MicroBenchmarksBase):
  def _get_benchmark_name(self):
    return self._get_name()
  def _run(self, func, num_iters):
    gc.disable()
    gc.collect()
    self.run_report(_run_benchmark, func, num_iters)
    gc.enable()
  def benchmark_tf_constant_2x2(self):
    x = [[1., 2.], [3., 4.]]
    def fn():
      with trace.Trace("tf.constant-2x2"):
        tf.constant(x)
    self._run(fn, NUM_ITERATIONS)
  def benchmark_tf_convert_to_tensor_2x2(self):
    x = [[1., 2.], [3., 4.]]
    def fn():
      with trace.Trace("tf.convert_to_tensor-2x2"):
        tf.convert_to_tensor(x)
    self._run(fn, NUM_ITERATIONS)
  def benchmark_tf_nn_relu_2x2(self):
    x = tf.constant([[1., 2.], [3., 4.]])
    def fn():
      with trace.Trace("tf.nn.relu-2x2"):
        tf.nn.relu(x)
    self._run(fn, NUM_ITERATIONS)
  def benchmark_tf_function_invocation_identity(self):
    x = tf.constant([[1., 2.], [3., 4.]])
    @tf.function
    def identity(x):
      return x
    def fn():
      with trace.Trace("tf.function-identity"):
        identity(x)
    self._run(fn, NUM_ITERATIONS)
if __name__ == "__main__":
  tf.test.main()
