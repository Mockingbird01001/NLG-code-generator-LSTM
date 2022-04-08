
import functools
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import test
def _test_combinations():
  def make_range():
    return dataset_ops.Dataset.range(10)
  def fn_with_side_effect(arg):
    logging_ops.print_v2(arg)
    return arg
  def apply_map_with_capture(ds):
    const = constant_op.constant(-1, dtype=dtypes.int64)
    return ds.map(lambda x: (x, const))
  def apply_map_with_multiple_components(ds):
  parallel_map_name = "ParallelMap"
  cases = [
      ("Skip0", lambda ds: ds.skip(0), None),
      ("SkipN", lambda ds: ds.skip(5), "FiniteSkip"),
      ("Repeat1", lambda ds: ds.repeat(1), None),
      ("RepeatN", lambda ds: ds.repeat(10), "FiniteRepeat"),
      ("Prefetch0", lambda ds: ds.prefetch(0), None),
      ("PrefetchN", lambda ds: ds.prefetch(1), "Prefetch"),
      ("Take-1", lambda ds: ds.take(-1), None),
      ("TakeN", lambda ds: ds.take(2), "FiniteTake"),
      ("MapIdentity", lambda ds: ds.map(lambda x: x), None),
      ("MapNonIdentity", lambda ds: ds.map(lambda x: x * 2), "Map"),
      ("MapWithSideEffect", lambda ds: ds.map(fn_with_side_effect), "Map"),
      ("MapWithCapture", apply_map_with_capture, "Map"),
      ("MapWithMultipleComponents", apply_map_with_multiple_components,
       parallel_map_name),
      ("MapRestructure", lambda ds: ds.map(lambda x: {"value": x}), ""),
      ("PMapIdentity", lambda ds: ds.map(lambda x: x, num_parallel_calls=2),
       None),
      ("PMapNonIdentity",
       lambda ds: ds.map(lambda x: x * 2, num_parallel_calls=2),
       parallel_map_name),
      ("Shard1", lambda ds: ds.shard(1, 0), None),
      ("ShardN", lambda ds: ds.shard(2, 0), "Shard"),
  ]
  def reduce_fn(result, case):
    name, transformation, expected = case
    return result + combinations.combine(
        init_dataset_fn=make_range,
        transformation=combinations.NamedObject(name, transformation),
        expected_name=expected)
  test_combinations = functools.reduce(reduce_fn, cases, [])
  return test_combinations
class NoopEliminationTest(test_base.DatasetTestBase, parameterized.TestCase):
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_combinations()))
  def testNoopElimination(self, init_dataset_fn, transformation, expected_name):
    dataset = init_dataset_fn()
    if expected_name:
      dataset = dataset.apply(
          testing.assert_next([expected_name, "FiniteTake"]))
    else:
      dataset = dataset.apply(testing.assert_next(["FiniteTake"]))
    dataset = dataset.apply(transformation)
    dataset = dataset.take(1)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset)
    self.evaluate(get_next())
if __name__ == "__main__":
  test.main()
