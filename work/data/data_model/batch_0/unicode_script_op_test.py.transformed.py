
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test
class UnicodeScriptOpTest(test.TestCase):
  @test_util.run_deprecated_v1
  def testValidScripts(self):
    inputs = [
        ord("a"),
        ord(",")
    ]
    with self.cached_session():
      input_vector = constant_op.constant(inputs, dtypes.int32)
      outputs = string_ops.unicode_script(input_vector).eval()
      self.assertAllEqual(
          outputs,
          [
          ])
  @test_util.run_deprecated_v1
  def testInvalidScript(self):
    inputs = [-100, 0xffffff]
    with self.cached_session():
      input_vector = constant_op.constant(inputs, dtypes.int32)
      outputs = string_ops.unicode_script(input_vector).eval()
      self.assertAllEqual(outputs, [-1, -1])
class UnicodeScriptBenchmarks(test.Benchmark):
  def _generateBenchmarkInput(self, size):
    chars = []
    i = 0
    offset = 0
    continuity_size = 20
    while i < size:
      chars.append(ord("a") + offset)
      i += 1
      offset += 1
      if i % continuity_size == 0:
        offset += 100
        if offset > 0x1F940:
          offset = 0
    return chars
  def benchmark_unicode_script(self):
    with session.Session(config=benchmark.benchmark_config()) as sess:
      chars = self._generateBenchmarkInput(1000000)
      script = string_ops.unicode_script(chars)
      self.run_op_benchmark(sess, script.op, min_iters=100)
if __name__ == "__main__":
  test.main()
