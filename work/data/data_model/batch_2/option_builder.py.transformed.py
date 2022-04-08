
import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['profiler.ProfileOptionBuilder'])
class ProfileOptionBuilder(object):
  """Option Builder for Profiling API.
  For tutorial on the options, see
  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/options.md
  ```python
  opts = (
      tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
  opts = (tf.compat.v1.profiler.ProfileOptionBuilder()
      .with_max_depth(10)
      .with_min_micros(1000)
      .select(['accelerator_micros'])
      .with_stdout_output()
      .build()
  opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
      tf.profiler.ProfileOptionBuilder.time_and_memory())
      .with_displaying_options(show_name_regexes=['.*rnn.*'])
      .build())
  _ = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(),
                          run_meta=run_meta,
                          cmd='scope',
                          options=opts)
  ```
  """
  def __init__(self, options=None):
    if options is not None:
      self._options = copy.deepcopy(options)
    else:
      self._options = {'max_depth': 100,
                       'min_bytes': 0,
                       'min_micros': 0,
                       'min_params': 0,
                       'min_float_ops': 0,
                       'min_occurrence': 0,
                       'order_by': 'name',
                       'account_type_regexes': ['.*'],
                       'start_name_regexes': ['.*'],
                       'trim_name_regexes': [],
                       'show_name_regexes': ['.*'],
                       'hide_name_regexes': [],
                       'account_displayed_op_only': False,
                       'select': ['micros'],
                       'step': -1,
                       'output': 'stdout'}
  @staticmethod
  def trainable_variables_parameter():
    return {'max_depth': 10000,
            'min_bytes': 0,
            'min_micros': 0,
            'min_params': 0,
            'min_float_ops': 0,
            'min_occurrence': 0,
            'order_by': 'name',
            'account_type_regexes': [tfprof_logger.TRAINABLE_VARIABLES],
            'start_name_regexes': ['.*'],
            'trim_name_regexes': [],
            'show_name_regexes': ['.*'],
            'hide_name_regexes': [],
            'account_displayed_op_only': True,
            'select': ['params'],
            'step': -1,
            'output': 'stdout'}
  @staticmethod
  def float_operation():
    return {'max_depth': 10000,
            'min_bytes': 0,
            'min_micros': 0,
            'min_params': 0,
            'min_float_ops': 1,
            'min_occurrence': 0,
            'order_by': 'float_ops',
            'account_type_regexes': ['.*'],
            'start_name_regexes': ['.*'],
            'trim_name_regexes': [],
            'show_name_regexes': ['.*'],
            'hide_name_regexes': [],
            'account_displayed_op_only': True,
            'select': ['float_ops'],
            'step': -1,
            'output': 'stdout'}
  @staticmethod
  def time_and_memory(min_micros=1, min_bytes=1, min_accelerator_micros=0,
                      min_cpu_micros=0, min_peak_bytes=0, min_residual_bytes=0,
                      min_output_bytes=0):
    """Show operation time and memory consumptions.
    Args:
      min_micros: Only show profiler nodes with execution time
          no less than this. It sums accelerator and cpu times.
      min_bytes: Only show profiler nodes requested to allocate no less bytes
          than this.
      min_accelerator_micros: Only show profiler nodes spend no less than
          this time on accelerator (e.g. GPU).
      min_cpu_micros: Only show profiler nodes spend no less than
          this time on cpu.
      min_peak_bytes: Only show profiler nodes using no less than this bytes
          at peak (high watermark). For profiler nodes consist of multiple
          graph nodes, it sums the graph nodes' peak_bytes.
      min_residual_bytes: Only show profiler nodes have no less than
          this bytes not being de-allocated after Compute() ends. For
          profiler nodes consist of multiple graph nodes, it sums the
          graph nodes' residual_bytes.
      min_output_bytes: Only show profiler nodes have no less than this bytes
          output. The output are not necessarily allocated by this profiler
          nodes.
    Returns:
      A dict of profiling options.
    """
    return {'max_depth': 10000,
            'min_bytes': min_bytes,
            'min_peak_bytes': min_peak_bytes,
            'min_residual_bytes': min_residual_bytes,
            'min_output_bytes': min_output_bytes,
            'min_micros': min_micros,
            'min_accelerator_micros': min_accelerator_micros,
            'min_cpu_micros': min_cpu_micros,
            'min_params': 0,
            'min_float_ops': 0,
            'min_occurrence': 0,
            'order_by': 'micros',
            'account_type_regexes': ['.*'],
            'start_name_regexes': ['.*'],
            'trim_name_regexes': [],
            'show_name_regexes': ['.*'],
            'hide_name_regexes': [],
            'account_displayed_op_only': True,
            'select': ['micros', 'bytes'],
            'step': -1,
            'output': 'stdout'}
  def build(self):
    return copy.deepcopy(self._options)
  def with_max_depth(self, max_depth):
    """Set the maximum depth of display.
    The depth depends on profiling view. For 'scope' view, it's the
    depth of name scope hierarchy (tree), for 'op' view, it's the number
    of operation types (list), etc.
    Args:
      max_depth: Maximum depth of the data structure to display.
    Returns:
      self
    """
    self._options['max_depth'] = max_depth
    return self
  def with_min_memory(self,
                      min_bytes=0,
                      min_peak_bytes=0,
                      min_residual_bytes=0,
                      min_output_bytes=0):
    """Only show profiler nodes consuming no less than 'min_bytes'.
    Args:
      min_bytes: Only show profiler nodes requested to allocate no less bytes
          than this.
      min_peak_bytes: Only show profiler nodes using no less than this bytes
          at peak (high watermark). For profiler nodes consist of multiple
          graph nodes, it sums the graph nodes' peak_bytes.
      min_residual_bytes: Only show profiler nodes have no less than
          this bytes not being de-allocated after Compute() ends. For
          profiler nodes consist of multiple graph nodes, it sums the
          graph nodes' residual_bytes.
      min_output_bytes: Only show profiler nodes have no less than this bytes
          output. The output are not necessarily allocated by this profiler
          nodes.
    Returns:
      self
    """
    self._options['min_bytes'] = min_bytes
    self._options['min_peak_bytes'] = min_peak_bytes
    self._options['min_residual_bytes'] = min_residual_bytes
    self._options['min_output_bytes'] = min_output_bytes
    return self
  def with_min_execution_time(self,
                              min_micros=0,
                              min_accelerator_micros=0,
                              min_cpu_micros=0):
    """Only show profiler nodes consuming no less than 'min_micros'.
    Args:
      min_micros: Only show profiler nodes with execution time
          no less than this. It sums accelerator and cpu times.
      min_accelerator_micros: Only show profiler nodes spend no less than
          this time on accelerator (e.g. GPU).
      min_cpu_micros: Only show profiler nodes spend no less than
          this time on cpu.
    Returns:
      self
    """
    self._options['min_micros'] = min_micros
    self._options['min_accelerator_micros'] = min_accelerator_micros
    self._options['min_cpu_micros'] = min_cpu_micros
    return self
  def with_min_parameters(self, min_params):
    self._options['min_params'] = min_params
    return self
  def with_min_occurrence(self, min_occurrence):
    """Only show profiler nodes including no less than 'min_occurrence' graph nodes.
    A "node" means a profiler output node, which can be a python line
    (code view), an operation type (op view), or a graph node
    (graph/scope view). A python line includes all graph nodes created by that
    line, while an operation type includes all graph nodes of that type.
    Args:
      min_occurrence: Only show nodes including no less than this.
    Returns:
      self
    """
    self._options['min_occurrence'] = min_occurrence
    return self
  def with_min_float_operations(self, min_float_ops):
    self._options['min_float_ops'] = min_float_ops
    return self
  def with_accounted_types(self, account_type_regexes):
    """Selectively counting statistics based on node types.
    Here, 'types' means the profiler nodes' properties. Profiler by default
    consider device name (e.g. /job:xx/.../device:GPU:0) and operation type
    (e.g. MatMul) as profiler nodes' properties. User can also associate
    customized 'types' to profiler nodes through OpLogProto proto.
    For example, user can select profiler nodes placed on gpu:0 with:
    `account_type_regexes=['.*gpu:0.*']`
    If none of a node's properties match the specified regexes, the node is
    not displayed nor accounted.
    Args:
      account_type_regexes: A list of regexes specifying the types.
    Returns:
      self.
    """
    self._options['account_type_regexes'] = copy.copy(account_type_regexes)
    return self
  def with_node_names(self,
                      start_name_regexes=None,
                      show_name_regexes=None,
                      hide_name_regexes=None,
                      trim_name_regexes=None):
    if start_name_regexes is not None:
      self._options['start_name_regexes'] = copy.copy(start_name_regexes)
    if show_name_regexes is not None:
      self._options['show_name_regexes'] = copy.copy(show_name_regexes)
    if hide_name_regexes is not None:
      self._options['hide_name_regexes'] = copy.copy(hide_name_regexes)
    if trim_name_regexes is not None:
      self._options['trim_name_regexes'] = copy.copy(trim_name_regexes)
    return self
  def account_displayed_op_only(self, is_true):
    self._options['account_displayed_op_only'] = is_true
    return self
  def with_empty_output(self):
    self._options['output'] = 'none'
    return self
  def with_stdout_output(self):
    self._options['output'] = 'stdout'
    return self
  def with_file_output(self, outfile):
    self._options['output'] = 'file:outfile=%s' % outfile
    return self
  def with_timeline_output(self, timeline_file):
    self._options['output'] = 'timeline:outfile=%s' % timeline_file
    return self
  def with_pprof_output(self, pprof_file):
    self._options['output'] = 'pprof:outfile=%s' % pprof_file
    return self
  def order_by(self, attribute):
    self._options['order_by'] = attribute
    return self
  def select(self, attributes):
    self._options['select'] = copy.copy(attributes)
    return self
  def with_step(self, step):
    """Which profile step to use for profiling.
    The 'step' here refers to the step defined by `Profiler.add_step()` API.
    Args:
      step: When multiple steps of profiles are available, select which step's
         profile to use. If -1, use average of all available steps.
    Returns:
      self
    """
    self._options['step'] = step
    return self
