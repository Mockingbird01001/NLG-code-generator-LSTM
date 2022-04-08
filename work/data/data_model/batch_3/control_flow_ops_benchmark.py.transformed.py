
import time
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
class CondWithManyIntermediatesBenchmark(test.Benchmark):
  NUM_INTERMEDIATES = 1000
  NUM_ITERS = 500
  NUM_WARM_UP_ITERS = 50
  def _create_cond(self, x):
    def branch_fn():
      return x + sum(random_ops.random_normal([])
                     for _ in range(self.NUM_INTERMEDIATES))
    return control_flow_ops.cond(math_ops.not_equal(x, -1),
                                 branch_fn, lambda: 0.0)
  def _benchmark_defun(self):
    @function.defun
    def cond_fn(x):
      return self._create_cond(x)
    for _ in range(self.NUM_WARM_UP_ITERS):
      cond_fn(0.0)
    start_time = time.time()
    for _ in range(self.NUM_ITERS):
      cond_fn(0.0)
    self.report_benchmark(
        wall_time=time.time() - start_time,
        iters=self.NUM_ITERS)
  def _benchmark_graph(self):
    with context.graph_mode():
      with ops.Graph().as_default():
        x = array_ops.placeholder(dtypes.float32)
        cond_val = self._create_cond(x)
        with session.Session() as sess:
          cond_fn = sess.make_callable(cond_val, [x])
          for _ in range(self.NUM_WARM_UP_ITERS):
            cond_fn(0.0)
          start_time = time.time()
          for _ in range(self.NUM_ITERS):
            cond_fn(0.0)
          self.report_benchmark(
              wall_time=time.time() - start_time,
              iters=self.NUM_ITERS)
  def benchmark_cond_v1_defun(self):
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False
    self._benchmark_defun()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val
  def benchmark_cond_v2_defun(self):
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
    self._benchmark_defun()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val
  def benchmark_cond_v1_graph(self):
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False
    self._benchmark_graph()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val
  def benchmark_cond_v2_graph(self):
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
    self._benchmark_graph()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val
if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
