
import numpy as np
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import get_single_element
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.eager import def_function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_dataset_ops
class SingleThreadedFlatMapDataset(dataset_ops.UnaryDataset):
  def __init__(self, input_dataset, map_func):
    self._input_dataset = input_dataset
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        defun_kwargs={"_executor": "SINGLE_THREADED_EXECUTOR"})
    variant_tensor = gen_dataset_ops.flat_map_dataset(
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **self._flat_structure)
    super(SingleThreadedFlatMapDataset, self).__init__(input_dataset,
                                                       variant_tensor)
  def _functions(self):
    return [self._map_func]
  @property
  def element_spec(self):
    return self._structure
  def _transformation_name(self):
    return "SingleThreadedFlatMapDataset"
class FromTensorSlicesBenchmark(benchmark_base.DatasetBenchmarkBase):
  def benchmark_slice_repeat_batch(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100
    num_elements = input_size * num_epochs // batch_size
    input_data = np.random.randn(input_size)
    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "from_tensor_slices.benchmark.1",
            "parameters": "%d.%d" % (input_size, batch_size),
        },
        name="slice_repeat_batch_input_%d_batch_%d" % (input_size, batch_size))
  def benchmark_reshape_slice_repeat(self):
    input_size = 10000
    reshape_dim = [100, 100]
    num_epochs = 100
    num_elements = num_epochs * reshape_dim[0]
    data = np.random.randn(input_size).reshape(*reshape_dim)
    dataset = dataset_ops.Dataset.from_tensor_slices(data).repeat(num_epochs)
    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "from_tensor_slices.benchmark.2",
            "parameters": "%d" % input_size,
        },
        name="reshape_slice_repeat_input_%d" % input_size,
    )
  def benchmark_slice_repeat_sparse(self):
    non_zeros_per_row_values = [0, 1, 5, 10, 100]
    num_rows_values = [32, 64, 128, 1024]
    for non_zeros_per_row in non_zeros_per_row_values:
      tensor = sparse_tensor.SparseTensor(
          indices=np.arange(non_zeros_per_row, dtype=np.int64)[:, np.newaxis],
          values=np.arange(non_zeros_per_row, dtype=np.int64),
          dense_shape=[1000])
      for num_rows in num_rows_values:
        @def_function.function
        def make_dataset():
          dataset = dataset_ops.Dataset.from_tensors(tensor)
          dataset = dataset.repeat(num_rows).batch(num_rows)
          batched_tensor = get_single_element.get_single_element(dataset)
          dataset = dataset_ops.Dataset.from_tensors(batched_tensor).repeat()
          return SingleThreadedFlatMapDataset(
              dataset, dataset_ops.Dataset.from_tensor_slices)
        self.run_and_report_benchmark(
            make_dataset(),
            num_elements=100000,
            iters=5,
            extras={
                "model_name": "from_tensor_slices.benchmark.3",
                "parameters": "%d.%d" % (non_zeros_per_row, num_rows),
            },
            name="slice_repeat_sparse_elements_per_row_%d_num_rows_%d" %
            (non_zeros_per_row, num_rows))
  def benchmark_slice_batch_cache_repeat(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100
    num_elements = input_size * num_epochs // batch_size
    input_data = np.random.randn(input_size)
    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data).batch(
            batch_size).cache().repeat(num_epochs))
    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "from_tensor_slices.benchmark.4",
            "parameters": "%d.%d" % (input_size, batch_size),
        },
        name="slice_batch_cache_repeat_input_%d_batch_%d" %
        (input_size, batch_size))
if __name__ == "__main__":
  benchmark_base.test.main()
