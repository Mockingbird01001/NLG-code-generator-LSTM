
from tensorflow.python import pywrap_tfe
class CancellationManager(object):
  __slots__ = ["_impl"]
  def __init__(self):
    self._impl = pywrap_tfe.TFE_NewCancellationManager()
  @property
  def is_cancelled(self):
    return pywrap_tfe.TFE_CancellationManagerIsCancelled(self._impl)
  def start_cancel(self):
    pywrap_tfe.TFE_CancellationManagerStartCancel(self._impl)
  def get_cancelable_function(self, concrete_function):
    return concrete_function._experimental_with_cancellation_manager(self)
