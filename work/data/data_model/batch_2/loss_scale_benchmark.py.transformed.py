
import time
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
def _get_strategy(num_gpus):
  if num_gpus > 1:
    return mirrored_strategy.MirroredStrategy(
        ['/GPU:%d' % i for i in range(num_gpus)])
  else:
class LossScaleBenchmark(test.Benchmark):
  def _benchmark(self, gradient_type, num_gpus, mode, loss_scaling):
    ls_str = loss_scaling or 'no_loss_scaling'
    name = '%s_%d_GPU_%s_%s' % (gradient_type, num_gpus, mode, ls_str)
    with context.eager_mode(), _get_strategy(num_gpus).scope() as strategy:
      opt = adam.Adam()
      if loss_scaling == 'fixed':
        loss_scale = loss_scale_module.FixedLossScale(2.)
      elif loss_scaling == 'dynamic':
        increment_period = 1000000
        loss_scale = loss_scale_module.DynamicLossScale(
            initial_loss_scale=2., increment_period=increment_period)
      else:
        assert loss_scaling is None
        loss_scale = None
      if loss_scale:
        opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      num_vars = 200
      num_warmup_iters = 1
      num_iters = 20
      var_list = [
          variables.Variable(i, dtype='float32') for i in range(num_vars)]
      def get_loss():
        return math_ops.add_n(var_list)
      if gradient_type == 'gradient_tape':
        if loss_scale is None:
          def minimize_fn():
            with backprop.GradientTape() as tape:
              loss = get_loss()
            grads = tape.gradient(loss, var_list)
            return opt.apply_gradients(zip(grads, var_list))
        else:
          def minimize_fn():
            with backprop.GradientTape() as tape:
              loss = get_loss()
              scaled_loss = opt.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_loss, var_list)
            grads = opt.get_unscaled_gradients(scaled_grads)
            return opt.apply_gradients(zip(grads, var_list))
      else:
        assert gradient_type == 'optimizer'
        def minimize_fn():
          return opt.minimize(get_loss, var_list)
      def run_fn():
        strategy.run(minimize_fn)
      if mode == 'tf_function':
        run_fn = def_function.function(run_fn)
      for _ in range(num_warmup_iters):
        run_fn()
      start = time.time()
      for _ in range(num_iters):
        run_fn()
      end = time.time()
      self.report_benchmark(iters=num_iters,
                            wall_time=(end - start) / num_iters, name=name)
  def _gpus_to_test_with(self):
    num_gpus = len(config.list_logical_devices('GPU'))
    gpus_to_test_with = []
    if num_gpus >= 1:
      gpus_to_test_with.append(1)
    if num_gpus >= 2:
      gpus_to_test_with.append(2)
    if num_gpus >= 8:
      gpus_to_test_with.append(8)
    return gpus_to_test_with
  def benchmark_optimizer(self):
    for num_gpus in self._gpus_to_test_with():
      for mode in 'eager', 'tf_function':
        for loss_scaling in None, 'fixed', 'dynamic':
          self._benchmark('optimizer', num_gpus, mode, loss_scaling)
  def benchmark_gradient_tape(self):
    for num_gpus in self._gpus_to_test_with():
      for mode in 'eager', 'tf_function':
        for loss_scaling in None, 'fixed', 'dynamic':
          self._benchmark('gradient_tape', num_gpus, mode, loss_scaling)
if __name__ == '__main__':
  test.main()
