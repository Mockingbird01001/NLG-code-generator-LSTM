
import os
from typing import Optional, Text
if not os.path.splitext(__file__)[0].endswith(
    os.path.join('tflite_runtime', 'metrics_portable')):
else:
class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
  def __init__(self,
               model_hash: Optional[Text] = None,
               model_path: Optional[Text] = None) -> None:
    pass
  def increase_counter_debugger_creation(self):
    pass
  def increase_counter_interpreter_creation(self):
    pass
  def increase_counter_converter_attempt(self):
    pass
  def increase_counter_converter_success(self):
    pass
  def set_converter_param(self, name, value):
    pass
  def set_converter_error(self, error_data):
    pass
  def set_converter_latency(self, value):
    pass
class TFLiteConverterMetrics(TFLiteMetrics):
  def __del__(self):
    pass
  def set_export_required(self):
    pass
  def export_metrics(self):
    pass
