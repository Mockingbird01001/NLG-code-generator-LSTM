
import time
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import test
def run_benchmark(func, num_iters):
  start = time.time()
  for _ in range(num_iters):
    func()
  end = time.time()
  return end - start
class SingleOpBenchmarks(test.Benchmark):
  def _run_and_report(self, func, num_iters):
    total_time = run_benchmark(func, num_iters)
    mean_us = total_time * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={
            "examples_per_sec": float("{0:.3f}".format(num_iters / total_time)),
        })
  def benchmarkAddScalars(self):
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(shape=[], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(shape=[], dtype=dtypes.float32, name="y")
      def bench():
        return gen_math_ops.add(x, y)
      self._run_and_report(bench, 1000)
  def benchmarkAddBatchedMatrices(self):
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(
          shape=[32, 784, 1000], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(
          shape=[32, 784, 1000], dtype=dtypes.float32, name="y")
      def bench():
        return gen_math_ops.add(x, y)
      self._run_and_report(bench, 1000)
  def benchmarkMatMul(self):
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(
          shape=[784, 1000], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(
          shape=[1000, 1000], dtype=dtypes.float32, name="y")
      def bench():
        return gen_math_ops.mat_mul(x, y)
      self._run_and_report(bench, 1000)
if __name__ == "__main__":
  test.main()
