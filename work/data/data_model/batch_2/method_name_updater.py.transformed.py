
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl as loader
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=["saved_model.signature_def_utils.MethodNameUpdater"])
class MethodNameUpdater(object):
  """Updates the method name(s) of the SavedModel stored in the given path.
  The `MethodNameUpdater` class provides the functionality to update the method
  name field in the signature_defs of the given SavedModel. For example, it
  can be used to replace the `predict` `method_name` to `regress`.
  Typical usages of the `MethodNameUpdater`
  ```python
  ...
  updater = tf.compat.v1.saved_model.signature_def_utils.MethodNameUpdater(
      export_dir)
  updater.replace_method_name(signature_key="foo", method_name="regress")
  updater.replace_method_name(signature_key="bar", method_name="classify",
                              tags="serve")
  updater.save(new_export_dir)
  ```
  Note: This function will only be available through the v1 compatibility
  library as tf.compat.v1.saved_model.builder.MethodNameUpdater.
  """
  def __init__(self, export_dir):
    self._export_dir = export_dir
    self._saved_model = loader.parse_saved_model(export_dir)
  def replace_method_name(self, signature_key, method_name, tags=None):
    """Replaces the method_name in the specified signature_def.
    This will match and replace multiple sig defs iff tags is None (i.e when
    multiple `MetaGraph`s have a signature_def with the same key).
    If tags is not None, this will only replace a single signature_def in the
    `MetaGraph` with matching tags.
    Args:
      signature_key: Key of the signature_def to be updated.
      method_name: new method_name to replace the existing one.
      tags: A tag or sequence of tags identifying the `MetaGraph` to update. If
          None, all meta graphs will be updated.
    Raises:
      ValueError: if signature_key or method_name are not defined or
          if no metagraphs were found with the associated tags or
          if no meta graph has a signature_def that matches signature_key.
    """
    if not signature_key:
      raise ValueError("`signature_key` must be defined.")
    if not method_name:
      raise ValueError("`method_name` must be defined.")
    if (tags is not None and not isinstance(tags, list)):
      tags = [tags]
    found_match = False
    for meta_graph_def in self._saved_model.meta_graphs:
      if tags is None or set(tags) == set(meta_graph_def.meta_info_def.tags):
        if signature_key not in meta_graph_def.signature_def:
          raise ValueError(
              f"MetaGraphDef associated with tags {tags} "
              f"does not have a signature_def with key: '{signature_key}'. "
              "This means either you specified the wrong signature key or "
              "forgot to put the signature_def with the corresponding key in "
              "your SavedModel.")
        meta_graph_def.signature_def[signature_key].method_name = method_name
        found_match = True
    if not found_match:
      raise ValueError(
          f"MetaGraphDef associated with tags {tags} could not be found in "
          "SavedModel. This means either you specified invalid tags or your "
          "SavedModel does not have a MetaGraphDef with the specified tags.")
  def save(self, new_export_dir=None):
    is_input_text_proto = file_io.file_exists(
        file_io.join(
            compat.as_bytes(self._export_dir),
            compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT)))
    if not new_export_dir:
      new_export_dir = self._export_dir
    if is_input_text_proto:
      path = file_io.join(
          compat.as_bytes(new_export_dir),
          compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
      file_io.write_string_to_file(path, str(self._saved_model))
    else:
      path = file_io.join(
          compat.as_bytes(new_export_dir),
          compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
      file_io.write_string_to_file(
          path, self._saved_model.SerializeToString(deterministic=True))
    tf_logging.info("SavedModel written to: %s", compat.as_text(path))
