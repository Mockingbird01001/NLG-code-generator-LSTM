
from tensorflow.python.platform import _pywrap_tf2
from tensorflow.python.util.tf_export import tf_export
def enable():
  _pywrap_tf2.enable(True)
def disable():
  _pywrap_tf2.enable(False)
@tf_export("__internal__.tf2.enabled", v1=[])
def enabled():
  return _pywrap_tf2.is_enabled()
