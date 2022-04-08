
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
class UnbatchBenchmark(benchmark_base.DatasetBenchmarkBase):
  def benchmark_native_unbatch(self):
    batch_sizes = [1, 2, 5, 10, 20, 50]
    num_elements = 10000
    for batch_size in batch_sizes:
      dataset = dataset_ops.Dataset.from_tensors("element").repeat(None)
      dataset = dataset.batch(batch_size)
      dataset = dataset.unbatch()
      self.run_and_report_benchmark(
          dataset=dataset,
          num_elements=num_elements,
          iters=5,
          extras={
              "model_name": "unbatch.benchmark.1",
              "parameters": "%d" % batch_size,
          },
          name="native_batch_size_%d" % batch_size)
  def benchmark_old_unbatch_implementation(self):
    batch_sizes = [1, 2, 5, 10, 20, 50]
    num_elements = 10000
    for batch_size in batch_sizes:
      dataset = dataset_ops.Dataset.from_tensors("element").repeat(None)
      dataset = dataset.batch(batch_size)
      dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
      self.run_and_report_benchmark(
          dataset=dataset,
          num_elements=num_elements,
          iters=5,
          extras={
              "model_name": "unbatch.benchmark.2",
              "parameters": "%d" % batch_size,
          },
          name="unfused_batch_size_%d" % batch_size)
if __name__ == "__main__":
  benchmark_base.test.main()
