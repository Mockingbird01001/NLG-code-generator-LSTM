
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
class LayerSavedModelSaver(base_serialization.SavedModelSaver):
  @property
  def object_identifier(self):
    return constants.LAYER_IDENTIFIER
  @property
  def python_properties(self):
    return self._python_properties_internal()
  def _python_properties_internal(self):
    metadata = dict(
        name=self.obj.name,
        trainable=self.obj.trainable,
        batch_input_shape=getattr(self.obj, '_batch_input_shape', None),
        stateful=self.obj.stateful,
    )
    metadata.update(get_serialized(self.obj))
    if self.obj.input_spec is not None:
      metadata['input_spec'] = nest.map_structure(
          lambda x: generic_utils.serialize_keras_object(x) if x else None,
          self.obj.input_spec)
    if (self.obj.activity_regularizer is not None and
        hasattr(self.obj.activity_regularizer, 'get_config')):
      metadata['activity_regularizer'] = generic_utils.serialize_keras_object(
          self.obj.activity_regularizer)
    return metadata
  def objects_to_serialize(self, serialization_cache):
    return (self._get_serialized_attributes(
        serialization_cache).objects_to_serialize)
  def functions_to_serialize(self, serialization_cache):
    return (self._get_serialized_attributes(
        serialization_cache).functions_to_serialize)
  def _get_serialized_attributes(self, serialization_cache):
    keras_cache = serialization_cache.setdefault(constants.KERAS_CACHE_KEY, {})
    if self.obj in keras_cache:
      return keras_cache[self.obj]
    serialized_attr = keras_cache[self.obj] = (
        serialized_attributes.SerializedAttributes.new(self.obj))
    if (save_impl.should_skip_serialization(self.obj) or
      return serialized_attr
    object_dict, function_dict = self._get_serialized_attributes_internal(
        serialization_cache)
    serialized_attr.set_and_validate_objects(object_dict)
    serialized_attr.set_and_validate_functions(function_dict)
    return serialized_attr
  def _get_serialized_attributes_internal(self, serialization_cache):
    objects = save_impl.wrap_layer_objects(self.obj, serialization_cache)
    functions = save_impl.wrap_layer_functions(self.obj, serialization_cache)
    functions['_default_save_signature'] = None
    return objects, functions
def get_serialized(obj):
  with generic_utils.skip_failed_serialization():
    return generic_utils.serialize_keras_object(obj)
class InputLayerSavedModelSaver(base_serialization.SavedModelSaver):
  @property
  def object_identifier(self):
    return constants.INPUT_LAYER_IDENTIFIER
  @property
  def python_properties(self):
    return dict(
        class_name=type(self.obj).__name__,
        name=self.obj.name,
        dtype=self.obj.dtype,
        sparse=self.obj.sparse,
        ragged=self.obj.ragged,
        config=self.obj.get_config())
  def objects_to_serialize(self, serialization_cache):
    return {}
  def functions_to_serialize(self, serialization_cache):
    return {}
class RNNSavedModelSaver(LayerSavedModelSaver):
  @property
  def object_identifier(self):
    return constants.RNN_LAYER_IDENTIFIER
  def _get_serialized_attributes_internal(self, serialization_cache):
    objects, functions = (
        super(RNNSavedModelSaver, self)._get_serialized_attributes_internal(
            serialization_cache))
    states = data_structures.wrap_or_unwrap(self.obj.states)
    if isinstance(states, tuple):
      states = data_structures.wrap_or_unwrap(list(states))
    objects['states'] = states
    return objects, functions
class IndexLookupLayerSavedModelSaver(LayerSavedModelSaver):
  @property
  def python_properties(self):
    metadata = self._python_properties_internal()
    if metadata['config'].get('has_static_table', False):
      metadata['config']['vocabulary'] = None
    return metadata
