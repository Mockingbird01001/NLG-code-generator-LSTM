
from absl.testing import parameterized
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
def get_dataset_from_tensor_slices(inp_array):
  dataset = dataset_ops.DatasetV2.from_tensor_slices(inp_array)
  if not tf2.enabled():
    dataset = dataset_ops.Dataset.from_tensor_slices(inp_array)
  return dataset
class AssertFlattenedMixin(object):
  def assert_equal_flattened(self, expected_results, actual_results):
    self.assertEqual(len(expected_results), len(actual_results))
    for i, expected_result in enumerate(expected_results):
      final_result = []
      actual_result = actual_results[i]
      for val in actual_result:
        final_result.extend(val.numpy())
      self.assertAllEqual(expected_result, final_result)
class GradientTapeTest(test.TestCase, parameterized.TestCase,
                       AssertFlattenedMixin):
  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testStepInFunctionGradient(self, distribution):
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)
    @def_function.function
    def train_step(x):
      def computation(x):
        return math_ops.square(x)
      with backprop.GradientTape() as tape:
        y = computation(x)
      grads = tape.gradient(y, x)
      return grads
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = distribution.experimental_local_results(
          distribution.run(train_step, args=(x,)))
      results.append(output)
    self.assert_equal_flattened([[10., 12.], [14., 16.]], results)
  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testRunInFunctionGradient(self, distribution):
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)
    @def_function.function
    def run(x):
      def train_step(x):
        def computation(x):
          return math_ops.square(x)
        with backprop.GradientTape() as tape:
          y = computation(x)
        grads = tape.gradient(y, x)
        return grads
      return distribution.experimental_local_results(
          distribution.run(train_step, args=(x,)))
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = run(x)
      results.append(output)
    self.assert_equal_flattened([[10., 12.], [14., 16.]], results)
  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          model_in_tf_function=[True, False]
      ))
  def testNestedFunction(self, distribution, model_in_tf_function):
    def model(x):
      return x * x
    if model_in_tf_function:
      model = def_function.function(model)
    with distribution.scope():
      x = variables.Variable(1.0)
      @def_function.function
      def train_step():
        def replica_step():
          with backprop.GradientTape() as tape:
            y = model(x)
          return tape.gradient(y, x)
        return distribution.run(replica_step)
      grads = distribution.experimental_local_results(train_step())
      self.assertLen(grads, distribution.num_replicas_in_sync)
      self.assertTrue(all(g is not None for g in grads))
if __name__ == "__main__":
  test.main()
