
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test
def _test_combinations():
  def filter_fn(dataset, predicate):
    return dataset.filter(predicate)
  def legacy_filter_fn(dataset, predicate):
    return dataset.filter_with_legacy_function(predicate)
  filter_combinations = combinations.combine(
      tf_api_version=[1, 2],
      mode=["eager", "graph"],
      apply_filter=combinations.NamedObject("filter_fn", filter_fn))
  legacy_filter_combinations = combinations.combine(
      tf_api_version=1,
      mode=["eager", "graph"],
      apply_filter=combinations.NamedObject("legacy_filter_fn",
                                            legacy_filter_fn))
  return filter_combinations + legacy_filter_combinations
class FilterParallelizationTest(test_base.DatasetTestBase,
                                parameterized.TestCase):
  def enableFilterParallelization(self, dataset):
    options = options_lib.Options()
    options.experimental_optimization.filter_parallelization = True
    return dataset.with_options(options)
  @combinations.generate(_test_combinations())
  def testFilterDataset(self, apply_filter):
    components = (np.arange(7, dtype=np.int64),
                  np.array([[1, 2, 3]], dtype=np.int64) *
                  np.arange(7, dtype=np.int64)[:, np.newaxis],
                  np.array(37.0, dtype=np.float64) * np.arange(7))
    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)
      dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
          _map_fn).repeat(count)
      dataset = self.enableFilterParallelization(dataset)
      dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
      dataset = apply_filter(
          dataset,
          lambda x, _y, _z: math_ops.equal(math_ops.mod(x, modulus), 0))
      self.assertEqual(
          [c.shape[1:] for c in components],
          [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
      get_next = self.getNext(dataset)
      for _ in range(count):
        for i in [x for x in range(7) if x**2 % modulus == 0]:
          result = self.evaluate(get_next())
          for component, result_component in zip(components, result):
            self.assertAllEqual(component[i]**2, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())
    do_test(14, 2)
    do_test(4, 18)
    do_test(0, 1)
  @combinations.generate(_test_combinations())
  def testFilterRange(self, apply_filter):
    dataset = dataset_ops.Dataset.range(4)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset,
                           lambda x: math_ops.not_equal(math_ops.mod(x, 3), 2))
    self.assertDatasetProduces(dataset, expected_output=[0, 1, 3])
  @combinations.generate(_test_combinations())
  def testFilterDict(self, apply_filter):
    dataset = dataset_ops.Dataset.range(10).map(
        lambda x: {"foo": x * 2, "bar": x**2})
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset, lambda d: math_ops.equal(d["bar"] % 2, 0))
    dataset = dataset.map(lambda d: d["foo"] + d["bar"])
    self.assertDatasetProduces(
        dataset,
        expected_output=[(i * 2 + i**2) for i in range(10) if not (i**2) % 2])
  @combinations.generate(_test_combinations())
  def testUseStepContainerInFilter(self, apply_filter):
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    def _predicate(xs):
      squared_xs = map_fn.map_fn(lambda x: x * x, xs)
      summed = math_ops.reduce_sum(squared_xs)
      return math_ops.equal(summed, 1 + 4 + 9)
    dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
    dataset = self.enableFilterParallelization(dataset)
    if repr(apply_filter) != "legacy_filter_fn":
      dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    else:
      dataset = dataset.apply(testing.assert_next(["Filter"]))
    dataset = apply_filter(dataset, _predicate)
    self.assertDatasetProduces(dataset, expected_output=[input_data[0]])
  @combinations.generate(_test_combinations())
  def testSparse(self, apply_filter):
    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1])), i
    def _filter_fn(_, i):
      return math_ops.equal(i % 2, 0)
    dataset = dataset_ops.Dataset.range(10).map(_map_fn)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset, _filter_fn)
    dataset = dataset.map(lambda x, i: x)
    self.assertDatasetProduces(
        dataset, expected_output=[_map_fn(i * 2)[0] for i in range(5)])
  @combinations.generate(_test_combinations())
  def testShortCircuit(self, apply_filter):
    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(10),
         dataset_ops.Dataset.from_tensors(True).repeat(None)))
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset, lambda x, y: y)
    self.assertDatasetProduces(
        dataset, expected_output=[(i, True) for i in range(10)])
  @combinations.generate(_test_combinations())
  def testParallelFilters(self, apply_filter):
    dataset = dataset_ops.Dataset.range(10)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset, lambda x: math_ops.equal(x % 2, 0))
    next_elements = [self.getNext(dataset) for _ in range(10)]
    self.assertEqual([0 for _ in range(10)],
                     self.evaluate(
                         [next_element() for next_element in next_elements]))
  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).filter(
        lambda x: True, name="filter")
    self.assertDatasetProduces(dataset, [42])
  @combinations.generate(test_base.default_test_combinations())
  def testInputOutOfRange(self):
    def py_fn(_):
      raise StopIteration()
    dataset = dataset_ops.Dataset.range(5)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = dataset.filter(
        lambda x: script_ops.py_func(py_fn, [x], dtypes.bool, stateful=False))
    get_next = self.getNext(dataset)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(autotune=[False, True])))
  def testAutotuneSetting(self, autotune):
    dataset = dataset_ops.Dataset.range(4)
    options = options_lib.Options()
    options.experimental_optimization.filter_parallelization = True
    options.autotune.enabled = autotune
    dataset = dataset.with_options(options)
    if autotune:
      dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    else:
      dataset = dataset.apply(testing.assert_next(["Filter"]))
    dataset = dataset.filter(
        lambda x: math_ops.not_equal(math_ops.mod(x, 3), 2))
    self.assertDatasetProduces(dataset, expected_output=[0, 1, 3])
class FilterCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                           parameterized.TestCase):
  def enableFilterParallelization(self, dataset):
    options = options_lib.Options()
    options.experimental_optimization.filter_parallelization = True
    return dataset.with_options(options)
  def _build_filter_range_graph(self, div):
    dataset = dataset_ops.Dataset.range(100)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    return dataset.filter(
        lambda x: math_ops.not_equal(math_ops.mod(x, div), 2))
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    div = 3
    num_outputs = sum(x % 3 != 2 for x in range(100))
    verify_fn(self, lambda: self._build_filter_range_graph(div), num_outputs)
  def _build_filter_dict_graph(self):
    dataset = dataset_ops.Dataset.range(10).map(lambda x: {
        "foo": x * 2,
        "bar": x**2
    })
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    return dataset.filter(lambda d: math_ops.equal(d["bar"] % 2, 0)).map(
        lambda d: d["foo"] + d["bar"])
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testDict(self, verify_fn):
    num_outputs = sum((x**2) % 2 == 0 for x in range(10))
    verify_fn(self, lambda: self._build_filter_dict_graph(), num_outputs)
  def _build_sparse_filter(self):
    def _map_fn(i):
      return sparse_tensor.SparseTensor(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[1, 1]), i
    def _filter_fn(_, i):
      return math_ops.equal(i % 2, 0)
    dataset = dataset_ops.Dataset.range(10).map(_map_fn)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    return dataset.filter(_filter_fn).map(lambda x, i: x)
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testSparse(self, verify_fn):
    verify_fn(self, lambda: self._build_sparse_filter(), num_outputs=5)
if __name__ == "__main__":
  test.main()
