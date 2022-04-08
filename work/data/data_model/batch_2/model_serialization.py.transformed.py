
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.saving.saved_model import save_impl
class ModelSavedModelSaver(layer_serialization.LayerSavedModelSaver):
  @property
  def object_identifier(self):
    return constants.MODEL_IDENTIFIER
  def _python_properties_internal(self):
    metadata = super(ModelSavedModelSaver, self)._python_properties_internal()
    metadata.pop('stateful')
    metadata.update(
        saving_utils.model_metadata(
            self.obj, include_optimizer=True, require_config=False))
    return metadata
  def _get_serialized_attributes_internal(self, serialization_cache):
    default_signature = None
    if len(serialization_cache[constants.KERAS_CACHE_KEY]) == 1:
      default_signature = save_impl.default_save_signature(self.obj)
    objects, functions = (
        super(ModelSavedModelSaver, self)._get_serialized_attributes_internal(
            serialization_cache))
    functions['_default_save_signature'] = default_signature
    return objects, functions
class SequentialSavedModelSaver(ModelSavedModelSaver):
  @property
  def object_identifier(self):
    return constants.SEQUENTIAL_IDENTIFIER
