
import time
from tensorflow.python.eager import monitoring
from tensorflow.python.util import tf_contextlib
enable_metrics = False
_METRICS_MAPPING = {}
def _init():
  global _METRICS_MAPPING
  time_buckets = monitoring.ExponentialBuckets(0.001, 10, 6)
  function_tracing_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/function_tracing', time_buckets,
      'Sampler to track the time (in seconds) for tracing functions.')
  closure_execution_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/closure_execution',
      time_buckets,
      'Sampler to track the time (in seconds) for executing closures.')
  remote_value_fetch_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/remote_value_fetch',
      time_buckets,
      'Sampler to track the time (in seconds) for fetching remote_value.')
  _METRICS_MAPPING = {
      'function_tracing': function_tracing_sampler,
      'closure_execution': closure_execution_sampler,
      'remote_value_fetch': remote_value_fetch_sampler
  }
@tf_contextlib.contextmanager
def monitored_timer(metric_name, state_tracker=None):
  if not enable_metrics:
    yield
  else:
    if not _METRICS_MAPPING:
      _init()
    start_time = time.time()
    start_state = state_tracker() if state_tracker else None
    yield
    duration_sec = time.time() - start_time
    if state_tracker is None or state_tracker() != start_state:
      metric = _METRICS_MAPPING[metric_name]
      metric.get_cell().add(duration_sec)
def get_metric_summary(metric_name):
  metric = _METRICS_MAPPING[metric_name]
  histogram_proto = metric.get_cell().value()
  ret = dict()
  ret['min'] = histogram_proto.min
  ret['max'] = histogram_proto.max
  ret['num'] = histogram_proto.num
  ret['sum'] = histogram_proto.sum
  return ret
