
from tensorflow.python.util.tf_export import tf_export
_mixed_precision_graph_rewrite_is_enabled = False
_non_mixed_precision_session_created = False
_using_mixed_precision_policy = False
@tf_export('__internal__.train.is_mixed_precision_graph_rewrite_enabled', v1=[])
def is_mixed_precision_graph_rewrite_enabled():
  return _mixed_precision_graph_rewrite_is_enabled
def set_mixed_precision_graph_rewrite_enabled(enabled):
  global _mixed_precision_graph_rewrite_is_enabled
  _mixed_precision_graph_rewrite_is_enabled = enabled
def non_mixed_precision_session_created():
  return _non_mixed_precision_session_created
def set_non_mixed_precision_session_created(created):
  global _non_mixed_precision_session_created
  _non_mixed_precision_session_created = created
def is_using_mixed_precision_policy():
  return _using_mixed_precision_policy
@tf_export('__internal__.train.set_using_mixed_precision_policy', v1=[])
def set_using_mixed_precision_policy(is_using):
  global _using_mixed_precision_policy
  _using_mixed_precision_policy = is_using
