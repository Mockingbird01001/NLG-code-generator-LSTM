
import abc
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
class SavedModelSaver(object, metaclass=abc.ABCMeta):
  def __init__(self, obj):
    self.obj = obj
  @abc.abstractproperty
  def object_identifier(self):
    raise NotImplementedError
  @property
  def tracking_metadata(self):
    return json_utils.Encoder().encode(self.python_properties)
  def trackable_children(self, serialization_cache):
    if not utils.should_save_traces():
      return {}
    children = self.objects_to_serialize(serialization_cache)
    children.update(self.functions_to_serialize(serialization_cache))
    return children
  @abc.abstractproperty
  def python_properties(self):
    raise NotImplementedError
  @abc.abstractmethod
  def objects_to_serialize(self, serialization_cache):
    raise NotImplementedError
  @abc.abstractmethod
  def functions_to_serialize(self, serialization_cache):
    """Returns extra functions to include when serializing a Keras object.
    Normally, when calling exporting an object to SavedModel, only the
    functions and objects defined by the user are saved. For example:
    ```
    obj = tf.Module()
    obj.v = tf.Variable(1.)
    @tf.function
    def foo(...): ...
    obj.foo = foo
    w = tf.Variable(1.)
    tf.saved_model.save(obj, 'path/to/saved/model')
    loaded = tf.saved_model.load('path/to/saved/model')
    ```
    Assigning trackable objects to attributes creates a graph, which is used for
    both checkpointing and SavedModel serialization.
    When the graph generated from attribute tracking is insufficient, extra
    objects and functions may be added at serialization time. For example,
    most models do not have their call function wrapped with a @tf.function
    decorator. This results in `model.call` not being saved. Since Keras objects
    should be revivable from the SavedModel format, the call function is added
    as an extra function to serialize.
    This function and `objects_to_serialize` is called multiple times when
    exporting to SavedModel. Please use the cache to avoid generating new
    functions and objects. A fresh cache is created for each SavedModel export.
    Args:
      serialization_cache: Dictionary passed to all objects in the same object
        graph during serialization.
    Returns:
        A dictionary mapping attribute names to `Function` or
        `ConcreteFunction`.
    """
    raise NotImplementedError
