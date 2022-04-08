
import time
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
class ReduceBenchmarks(test.Benchmark):
  def _run(self, func, num_iters):
    func()
    start = time.time()
    for _ in range(num_iters):
      func()
    end = time.time()
    mean_us = (end - start) * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={"examples_per_sec": num_iters / (end - start)})
  def benchmark_reduce_sum_grad_eager(self):
    with context.eager_mode():
      tensor = array_ops.zeros([100, 1000])
      def fn():
        backprop.gradients_function(math_ops.reduce_sum, [0])(tensor)
      self._run(fn, 10000)
  def benchmark_reduce_sum_grad_eager_cpu(self):
    with context.eager_mode(), ops.device("/cpu:0"):
      tensor = array_ops.zeros([100, 1000])
      def fn():
        backprop.gradients_function(math_ops.reduce_sum, [0])(tensor)
      self._run(fn, 10000)
  def benchmark_reduce_sum_grad_graph(self):
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0)))
    with ops.Graph().as_default(), session.Session(config=config) as sess:
      tensor = constant_op.constant(np.zeros([100, 1000], dtype=np.float32))
      reduction = math_ops.reduce_sum(tensor)
      grad, = gradients_impl.gradients(reduction, tensor)
      def fn():
        self.evaluate(grad.op)
      self._run(fn, 10000)
  def benchmark_reduce_sum_grad_graph_cpu(self):
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0)))
    with ops.Graph().as_default(), session.Session(config=config) as sess:
      with ops.device("/cpu:0"):
        tensor = constant_op.constant(np.zeros([100, 1000], dtype=np.float32))
        reduction = math_ops.reduce_sum(tensor)
        grad, = gradients_impl.gradients(reduction, tensor)
      def fn():
        self.evaluate(grad.op)
      self._run(fn, 10000)
if __name__ == "__main__":
  test.main()
