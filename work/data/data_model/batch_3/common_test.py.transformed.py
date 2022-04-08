
import json
from tensorflow.python.debug.lib import common
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
class CommonTest(test_util.TensorFlowTestCase):
  @test_util.run_v1_only("Relies on tensor name, which is unavailable in TF2")
  def testOnFeedOneFetch(self):
    a = constant_op.constant(10.0, name="a")
    b = constant_op.constant(20.0, name="b")
    run_key = common.get_run_key({"a": a}, [b])
    loaded = json.loads(run_key)
    self.assertItemsEqual(["a:0"], loaded[0])
    self.assertItemsEqual(["b:0"], loaded[1])
  @test_util.run_v1_only("Relies on tensor name, which is unavailable in TF2")
  def testGetRunKeyFlat(self):
    a = constant_op.constant(10.0, name="a")
    b = constant_op.constant(20.0, name="b")
    run_key = common.get_run_key({"a": a}, [a, b])
    loaded = json.loads(run_key)
    self.assertItemsEqual(["a:0"], loaded[0])
    self.assertItemsEqual(["a:0", "b:0"], loaded[1])
  @test_util.run_v1_only("Relies on tensor name, which is unavailable in TF2")
  def testGetRunKeyNestedFetches(self):
    a = constant_op.constant(10.0, name="a")
    b = constant_op.constant(20.0, name="b")
    c = constant_op.constant(30.0, name="c")
    d = constant_op.constant(30.0, name="d")
    run_key = common.get_run_key(
        {}, {"set1": [a, b], "set2": {"c": c, "d": d}})
    loaded = json.loads(run_key)
    self.assertItemsEqual([], loaded[0])
    self.assertItemsEqual(["a:0", "b:0", "c:0", "d:0"], loaded[1])
if __name__ == "__main__":
  googletest.main()
