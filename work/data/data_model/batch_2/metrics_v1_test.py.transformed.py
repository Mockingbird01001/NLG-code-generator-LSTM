
from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import variables
def _labeled_dataset_fn():
  return dataset_ops.Dataset.range(1000).map(
      lambda x: {"labels": x % 5, "predictions": x % 3}).batch(
          4, drop_remainder=True)
def _boolean_dataset_fn():
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [True, False, True, False],
      "predictions": [True, True, False, False]}).repeat().batch(
          3, drop_remainder=True)
def _threshold_dataset_fn():
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [True, False, True, False],
      "predictions": [1.0, 0.75, 0.25, 0.]}).repeat().batch(
          3, drop_remainder=True)
def _regression_dataset_fn():
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [1., .5, 1., 0.],
      "predictions": [1., .75, .25, 0.]}).repeat()
def all_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.one_device_strategy,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      mode=["graph"])
def tpu_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.tpu_strategy_one_step,
          strategy_combinations.tpu_strategy
      ],
      mode=["graph"])
class MetricsV1Test(test.TestCase, parameterized.TestCase):
  def _test_metric(self, distribution, dataset_fn, metric_fn, expected_fn):
    with ops.Graph().as_default(), distribution.scope():
      iterator = distribution.make_input_fn_iterator(lambda _: dataset_fn())
      if strategy_test_lib.is_tpu_strategy(distribution):
        def step_fn(ctx, inputs):
          value, update = distribution.extended.call_for_each_replica(
              metric_fn, args=(inputs,))
          ctx.set_non_tensor_output(name="value", output=value)
          return distribution.group(update)
        ctx = distribution.extended.experimental_run_steps_on_iterator(
            step_fn, iterator, iterations=distribution.extended.steps_per_run)
        update = ctx.run_op
        value = ctx.non_tensor_outputs["value"]
        batches_per_update = (
            distribution.num_replicas_in_sync *
            distribution.extended.steps_per_run)
      else:
        value, update = distribution.extended.call_for_each_replica(
            metric_fn, args=(iterator.get_next(),))
        update = distribution.group(update)
        batches_per_update = distribution.num_replicas_in_sync
      self.evaluate(iterator.initializer)
      self.evaluate(variables.local_variables_initializer())
      batches_consumed = 0
      for i in range(4):
        self.evaluate(update)
        batches_consumed += batches_per_update
        self.assertAllClose(expected_fn(batches_consumed),
                            self.evaluate(value),
                            0.001,
          break
  @combinations.generate(all_combinations() + tpu_combinations())
  def testMean(self, distribution):
    def _dataset_fn():
      return dataset_ops.Dataset.range(1000).map(math_ops.to_float).batch(
          4, drop_remainder=True)
    def _expected_fn(num_batches):
      return num_batches * 2 - 0.5
    self._test_metric(distribution, _dataset_fn, metrics.mean, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testAccuracy(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.accuracy(labels, predictions)
    def _expected_fn(num_batches):
      return [3./4, 3./8, 3./12, 4./16][num_batches - 1]
    self._test_metric(
        distribution, _labeled_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations())
  def testMeanPerClassAccuracy(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.mean_per_class_accuracy(
          labels, predictions, num_classes=5)
    def _expected_fn(num_batches):
      mean = lambda x: sum(x) / len(x)
      return [mean([1., 1., 1., 0., 0.]),
              mean([0.5, 0.5, 0.5, 0., 0.]),
              mean([1./3, 1./3, 0.5, 0., 0.]),
              mean([0.5, 1./3, 1./3, 0., 0.])][num_batches - 1]
    self._test_metric(
        distribution, _labeled_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations())
  def testMeanIOU(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.mean_iou(
          labels, predictions, num_classes=5)
    def _expected_fn(num_batches):
      mean = lambda x: sum(x) / len(x)
              mean([1./4, 1./4, 1./3, 0., 0.]),
              mean([1./6, 1./6, 1./5, 0., 0.]),
              mean([2./8, 1./7, 1./7, 0., 0.])][num_batches - 1]
    self._test_metric(
        distribution, _labeled_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testMeanTensor(self, distribution):
    def _dataset_fn():
      dataset = dataset_ops.Dataset.range(1000).map(math_ops.to_float)
      dataset = dataset.batch(4, drop_remainder=True)
      return dataset
    def _expected_fn(num_batches):
      first = 2. * num_batches - 2.
      return [first, first + 1., first + 2., first + 3.]
    self._test_metric(
        distribution, _dataset_fn, metrics.mean_tensor, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testAUCROC(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.auc(labels, predictions, num_thresholds=8, curve="ROC",
                         summation_method="careful_interpolation")
    def _expected_fn(num_batches):
      return [0.5, 7./9, 0.8, 0.75][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testAUCPR(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.auc(labels, predictions, num_thresholds=8, curve="PR",
                         summation_method="careful_interpolation")
    def _expected_fn(num_batches):
      return [0.797267, 0.851238, 0.865411, 0.797267][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testFalseNegatives(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.false_negatives(labels, predictions)
    def _expected_fn(num_batches):
      return [1., 1., 2., 3.][num_batches - 1]
    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testFalseNegativesAtThresholds(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.false_negatives_at_thresholds(labels, predictions, [.5])
    def _expected_fn(num_batches):
      return [[1.], [1.], [2.], [3.]][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testTrueNegatives(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.true_negatives(labels, predictions)
    def _expected_fn(num_batches):
      return [0., 1., 2., 3.][num_batches - 1]
    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testTrueNegativesAtThresholds(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.true_negatives_at_thresholds(labels, predictions, [.5])
    def _expected_fn(num_batches):
      return [[0.], [1.], [2.], [3.]][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testFalsePositives(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.false_positives(labels, predictions)
    def _expected_fn(num_batches):
      return [1., 2., 2., 3.][num_batches - 1]
    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testFalsePositivesAtThresholds(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.false_positives_at_thresholds(labels, predictions, [.5])
    def _expected_fn(num_batches):
      return [[1.], [2.], [2.], [3.]][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testTruePositives(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.true_positives(labels, predictions)
    def _expected_fn(num_batches):
      return [1., 2., 3., 3.][num_batches - 1]
    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testTruePositivesAtThresholds(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.true_positives_at_thresholds(labels, predictions, [.5])
    def _expected_fn(num_batches):
      return [[1.], [2.], [3.], [3.]][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testPrecision(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.precision(labels, predictions)
    def _expected_fn(num_batches):
      return [0.5, 0.5, 0.6, 0.5][num_batches - 1]
    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testPrecisionAtThreshold(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.precision_at_thresholds(labels, predictions, [0.5])
    def _expected_fn(num_batches):
      return [[0.5], [0.5], [0.6], [0.5]][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testRecall(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.recall(labels, predictions)
    def _expected_fn(num_batches):
      return [0.5, 2./3, 0.6, 0.5][num_batches - 1]
    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testRecallAtThreshold(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.recall_at_thresholds(labels, predictions, [0.5])
    def _expected_fn(num_batches):
      return [[0.5], [2./3], [0.6], [0.5]][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testMeanSquaredError(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.mean_squared_error(labels, predictions)
    def _expected_fn(num_batches):
      return [0., 1./32, 0.208333, 0.15625][num_batches - 1]
    self._test_metric(
        distribution, _regression_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations() + tpu_combinations())
  def testRootMeanSquaredError(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.root_mean_squared_error(labels, predictions)
    def _expected_fn(num_batches):
      return [0., 0.176777, 0.456435, 0.395285][num_batches - 1]
    self._test_metric(
        distribution, _regression_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations())
  def testSensitivityAtSpecificity(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.sensitivity_at_specificity(labels, predictions, 0.8)
    def _expected_fn(num_batches):
      return [0.5, 2./3, 0.6, 0.5][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
  @combinations.generate(all_combinations())
  def testSpecificityAtSensitivity(self, distribution):
    def _metric_fn(x):
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.specificity_at_sensitivity(labels, predictions, 0.95)
    def _expected_fn(num_batches):
      return [0., 1./3, 0.5, 0.5][num_batches - 1]
    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)
if __name__ == "__main__":
  test.main()
