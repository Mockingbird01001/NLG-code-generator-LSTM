
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
class CounterTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(start=3, step=4, expected_output=[[3, 7, 11]]) +
          combinations.combine(start=0, step=-1, expected_output=[[0, -1, -2]]))
  )
  def testCounter(self, start, step, expected_output):
    dataset = counter.Counter(start, step)
    self.assertEqual(
        [], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    self.assertEqual(dtypes.int64, dataset_ops.get_legacy_output_types(dataset))
    get_next = self.getNext(dataset)
    for expected in expected_output:
      self.assertEqual(expected, self.evaluate(get_next()))
if __name__ == "__main__":
  test.main()
