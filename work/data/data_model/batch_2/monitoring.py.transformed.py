
import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
_MetricMethod = collections.namedtuple('MetricMethod', 'create delete get_cell')
_counter_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter0,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter1,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter2,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter2),
]
_int_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge2),
]
_string_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge2),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge3,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge3,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge3),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge4,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge4,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge4),
]
_bool_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge2),
]
_sampler_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler0,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler1,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler2,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler2),
]
class Metric(object):
  __slots__ = ["_metric", "_metric_name", "_metric_methods", "_label_length"]
  def __init__(self, metric_name, metric_methods, label_length, *args):
    self._metric_name = metric_name
    self._metric_methods = metric_methods
    self._label_length = label_length
    if label_length >= len(self._metric_methods):
      raise ValueError('Cannot create {} metric with label >= {}'.format(
          self._metric_name, len(self._metric_methods)))
    self._metric = self._metric_methods[self._label_length].create(*args)
  def __del__(self):
    try:
      deleter = self._metric_methods[self._label_length].delete
      metric = self._metric
    except AttributeError:
      return
    if deleter is not None:
      deleter(metric)
  def get_cell(self, *labels):
    if len(labels) != self._label_length:
      raise ValueError('The {} expects taking {} labels'.format(
          self._metric_name, self._label_length))
    return self._metric_methods[self._label_length].get_cell(
        self._metric, *labels)
class CounterCell(object):
  __slots__ = ["_cell"]
  def __init__(self, cell):
    self._cell = cell
  def increase_by(self, value):
    pywrap_tfe.TFE_MonitoringCounterCellIncrementBy(self._cell, value)
  def value(self):
    return pywrap_tfe.TFE_MonitoringCounterCellValue(self._cell)
class Counter(Metric):
  """A stateful class for updating a cumulative integer metric.
  This class encapsulates a set of values (or a single value for a label-less
  metric). Each value is identified by a tuple of labels. The class allows the
  user to increment each value.
  """
  __slots__ = []
  def __init__(self, name, description, *labels):
    super(Counter, self).__init__('Counter', _counter_methods, len(labels),
                                  name, description, *labels)
  def get_cell(self, *labels):
    return CounterCell(super(Counter, self).get_cell(*labels))
class IntGaugeCell(object):
  __slots__ = ["_cell"]
  def __init__(self, cell):
    self._cell = cell
  def set(self, value):
    pywrap_tfe.TFE_MonitoringIntGaugeCellSet(self._cell, value)
  def value(self):
    return pywrap_tfe.TFE_MonitoringIntGaugeCellValue(self._cell)
class IntGauge(Metric):
  """A stateful class for updating a gauge-like integer metric.
  This class encapsulates a set of integer values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """
  __slots__ = []
  def __init__(self, name, description, *labels):
    super(IntGauge, self).__init__('IntGauge', _int_gauge_methods, len(labels),
                                   name, description, *labels)
  def get_cell(self, *labels):
    return IntGaugeCell(super(IntGauge, self).get_cell(*labels))
class StringGaugeCell(object):
  __slots__ = ["_cell"]
  def __init__(self, cell):
    self._cell = cell
  def set(self, value):
    pywrap_tfe.TFE_MonitoringStringGaugeCellSet(self._cell, value)
  def value(self):
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_MonitoringStringGaugeCellValue(self._cell, buffer_)
      value = pywrap_tf_session.TF_GetBuffer(buffer_).decode('utf-8')
    return value
class StringGauge(Metric):
  """A stateful class for updating a gauge-like string metric.
  This class encapsulates a set of string values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """
  __slots__ = []
  def __init__(self, name, description, *labels):
    super(StringGauge, self).__init__('StringGauge', _string_gauge_methods,
                                      len(labels), name, description, *labels)
  def get_cell(self, *labels):
    return StringGaugeCell(super(StringGauge, self).get_cell(*labels))
class BoolGaugeCell(object):
  __slots__ = ["_cell"]
  def __init__(self, cell):
    self._cell = cell
  def set(self, value):
    pywrap_tfe.TFE_MonitoringBoolGaugeCellSet(self._cell, value)
  def value(self):
    return pywrap_tfe.TFE_MonitoringBoolGaugeCellValue(self._cell)
@tf_export("__internal__.monitoring.BoolGauge", v1=[])
class BoolGauge(Metric):
  """A stateful class for updating a gauge-like bool metric.
  This class encapsulates a set of boolean values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """
  __slots__ = []
  def __init__(self, name, description, *labels):
    super(BoolGauge, self).__init__('BoolGauge', _bool_gauge_methods,
                                    len(labels), name, description, *labels)
  def get_cell(self, *labels):
    return BoolGaugeCell(super(BoolGauge, self).get_cell(*labels))
class SamplerCell(object):
  __slots__ = ["_cell"]
  def __init__(self, cell):
    self._cell = cell
  def add(self, value):
    pywrap_tfe.TFE_MonitoringSamplerCellAdd(self._cell, value)
  def value(self):
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_MonitoringSamplerCellValue(self._cell, buffer_)
      proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(compat.as_bytes(proto_data))
    return histogram_proto
class Buckets(object):
  __slots__ = ["buckets"]
  def __init__(self, buckets):
    self.buckets = buckets
  def __del__(self):
    pywrap_tfe.TFE_MonitoringDeleteBuckets(self.buckets)
class ExponentialBuckets(Buckets):
  """Exponential bucketing strategy.
  Sets up buckets of the form:
      [-DBL_MAX, ..., scale * growth^i,
       scale * growth_factor^(i + 1), ..., DBL_MAX].
  """
  __slots__ = []
  def __init__(self, scale, growth_factor, bucket_count):
    super(ExponentialBuckets, self).__init__(
        pywrap_tfe.TFE_MonitoringNewExponentialBuckets(scale, growth_factor,
                                                       bucket_count))
class Sampler(Metric):
  """A stateful class for updating a cumulative histogram metric.
  This class encapsulates a set of histograms (or a single histogram for a
  label-less metric) configured with a list of increasing bucket boundaries.
  Each histogram is identified by a tuple of labels. The class allows the
  user to add a sample to each histogram value.
  """
  __slots__ = []
  def __init__(self, name, buckets, description, *labels):
    super(Sampler, self).__init__('Sampler', _sampler_methods, len(labels),
                                  name, buckets.buckets, description, *labels)
  def get_cell(self, *labels):
    return SamplerCell(super(Sampler, self).get_cell(*labels))
class MonitoredTimer(object):
  __slots__ = ["cell", "t"]
  def __init__(self, cell):
    self.cell = cell
  def __enter__(self):
    self.t = time.time()
    return self
  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback
    micro_seconds = (time.time() - self.t) * 1000000
    self.cell.increase_by(int(micro_seconds))
def monitored_timer(cell):
  def actual_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      with MonitoredTimer(cell):
        return func(*args, **kwargs)
    return wrapper
  return actual_decorator
