
"""A mixin class that delegates another Trackable to be used when saving.
This is intended to be used with wrapper classes that cannot directly proxy the
wrapped object (e.g. with wrapt.ObjectProxy), because there are inner attributes
that cannot be exposed.
The Wrapper class itself cannot contain any Trackable children, as only the
delegated Trackable will be saved to checkpoint and SavedModel.
This class will "disappear" and be replaced with the wrapped inner Trackable
after a cycle of SavedModel saving and loading, unless the object is registered
and loaded with Keras.
"""
from tensorflow.python.util.tf_export import tf_export
@tf_export("__internal__.tracking.DelegatingTrackableMixin", v1=[])
class DelegatingTrackableMixin(object):
  def __init__(self, trackable_obj):
    self._trackable = trackable_obj
  @property
  def _setattr_tracking(self):
    return self._trackable._setattr_tracking
  @_setattr_tracking.setter
  def _setattr_tracking(self, value):
    self._trackable._setattr_tracking = value
  @property
  def _update_uid(self):
    return self._trackable._update_uid
  @_update_uid.setter
  def _update_uid(self, value):
    self._trackable._update_uid = value
  @property
  def _unconditional_checkpoint_dependencies(self):
    return self._trackable._unconditional_checkpoint_dependencies
  @property
  def _unconditional_dependency_names(self):
    return self._trackable._unconditional_dependency_names
  @property
  def _name_based_restores(self):
    return self._trackable._name_based_restores
  def _maybe_initialize_trackable(self):
    return self._trackable._maybe_initialize_trackable()
  @property
  def _object_identifier(self):
    return self._trackable._object_identifier
  @property
  def _tracking_metadata(self):
    return self._trackable._tracking_metadata
  def _no_dependency(self, *args, **kwargs):
    return self._trackable._no_dependency(*args, **kwargs)
  def _name_based_attribute_restore(self, *args, **kwargs):
    return self._trackable._name_based_attribute_restore(*args, **kwargs)
  @property
  def _checkpoint_dependencies(self):
    return self._trackable._checkpoint_dependencies
  @property
  def _deferred_dependencies(self):
    return self._trackable._deferred_dependencies
  def _lookup_dependency(self, *args, **kwargs):
    return self._trackable._lookup_dependency(*args, **kwargs)
  def _add_variable_with_custom_getter(self, *args, **kwargs):
    return self._trackable._add_variable_with_custom_getter(*args, **kwargs)
  def _preload_simple_restoration(self, *args, **kwargs):
    return self._trackable._preload_simple_restoration(*args, **kwargs)
    return self._trackable._track_trackable(*args, **kwargs)
    return self._trackable._handle_deferred_dependencies(name, trackable)
  def _restore_from_checkpoint_position(self, checkpoint_position):
    return self._trackable._restore_from_checkpoint_position(
        checkpoint_position)
  def _single_restoration_from_checkpoint_position(self, *args, **kwargs):
    return self._trackable._single_restoration_from_checkpoint_position(
        *args, **kwargs)
  def _gather_saveables_for_checkpoint(self, *args, **kwargs):
    return self._trackable._gather_saveables_for_checkpoint(*args, **kwargs)
  def _list_extra_dependencies_for_serialization(self, *args, **kwargs):
    return self._trackable._list_extra_dependencies_for_serialization(
        *args, **kwargs)
  def _list_functions_for_serialization(self, *args, **kwargs):
    return self._trackable._list_functions_for_serialization(*args, **kwargs)
  def _trackable_children(self, *args, **kwargs):
    return self._trackable._trackable_children(*args, **kwargs)
  def _deserialization_dependencies(self, *args, **kwargs):
    return self._trackable._deserialization_dependencies(*args, **kwargs)
  def _export_to_saved_model_graph(self, *args, **kwargs):
    return self._trackable._export_to_saved_model_graph(*args, **kwargs)
