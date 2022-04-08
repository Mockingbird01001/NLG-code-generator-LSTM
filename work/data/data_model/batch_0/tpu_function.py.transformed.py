
import contextlib
import threading
class TpuContext(threading.local):
  def __init__(self):
    self._number_of_shards = None
  @property
  def number_of_shards(self):
    return self._number_of_shards
  def set_number_of_shards(self, number_of_shards):
    self._number_of_shards = number_of_shards
_current_tpu_context = TpuContext()
@contextlib.contextmanager
def tpu_shard_context(number_of_shards):
  if _current_tpu_context.number_of_shards is not None:
    raise NotImplementedError(
        "tpu_shard_context cannot be nested."
        "If you're using TPUEstimator with inference_on_tpu, "
        "make sure you have set "
        "export_saved_model_api_version=ExportSavedModelApiVersion.V2 in "
        "the creation of TPUEstimator.")
  try:
    _current_tpu_context.set_number_of_shards(number_of_shards)
    yield
  finally:
    _current_tpu_context.set_number_of_shards(None)
def get_tpu_context():
  return _current_tpu_context
def on_device_training_loop(func):
  setattr(func, "step_marker_location", "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP")
  return func
