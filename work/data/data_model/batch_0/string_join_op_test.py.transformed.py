
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
class StringJoinOpTest(test.TestCase):
  @test_util.run_deprecated_v1
  def testStringJoin(self):
    input0 = ["a", "b"]
    input1 = "a"
    input2 = [["b"], ["c"]]
    with self.cached_session():
      output = string_ops.string_join([input0, input1])
      self.assertAllEqual(output, [b"aa", b"ba"])
      output = string_ops.string_join([input0, input1], separator="--")
      self.assertAllEqual(output, [b"a--a", b"b--a"])
      output = string_ops.string_join([input0, input1, input0], separator="--")
      self.assertAllEqual(output, [b"a--a--a", b"b--a--b"])
      output = string_ops.string_join([input1] * 4, separator="!")
      self.assertEqual(self.evaluate(output), b"a!a!a!a")
      output = string_ops.string_join([input2] * 2, separator="")
      self.assertAllEqual(output, [[b"bb"], [b"cc"]])
        string_ops.string_join([input0, input2]).eval()
if __name__ == "__main__":
  test.main()
