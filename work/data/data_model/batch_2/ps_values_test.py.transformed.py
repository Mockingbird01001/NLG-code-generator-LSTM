
from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.central_storage_strategy_with_two_gpus
        ],
        mode=["graph", "eager"]))
class AggregatingVariableTest(test.TestCase, parameterized.TestCase):
  def testAssignOutOfScope(self, distribution):
    with distribution.scope():
      aggregating = variables_lib.Variable(1.)
    self.assertIsInstance(aggregating, ps_values.AggregatingVariable)
    self.evaluate(aggregating.assign(3.))
    self.assertEqual(self.evaluate(aggregating.read_value()), 3.)
    self.assertEqual(self.evaluate(aggregating._v.read_value()), 3.)
  def testAssignAdd(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          1, aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(variables_lib.global_variables_initializer())
    @def_function.function
    def assign():
      return v.assign_add(2)
    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(assign)))
    self.assertAllEqual([3], per_replica_results)
if __name__ == "__main__":
  test.main()
