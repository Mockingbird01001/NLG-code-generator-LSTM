
import functools
from typing import Any, Dict, Iterable, Optional, Union, Text, Callable
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.training.saving import saveable_hook
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import nest
_HOOK_KEY = "TPUEmbedding_saveable"
class TPUEmbeddingBase(tracking.AutoTrackable):
  def __init__(
      self,
    self._feature_config = feature_config
    self._output_shapes = []
    for feature in nest.flatten(feature_config):
      self._output_shapes.append(feature.output_shape)
    self._table_config = []
    for feature in nest.flatten(feature_config):
      if feature.table not in self._table_config:
        self._table_config.append(feature.table)
    table_names = []
    for i, table in enumerate(self._table_config):
      if table.optimizer is None:
        table.optimizer = optimizer
      if (table.optimizer is not None and
        raise ValueError("{} is an unsupported optimizer class. Please pass an "
                         "instance of one of the optimizer classes under "
                         "tf.tpu.experimental.embedding.".format(
                             type(table.optimizer)))
      if table.name is None:
        table.name = "table_{}".format(i)
      if table.name in table_names:
        raise ValueError("Tables must have a unique name. "
                         f"Multiple tables with name {table.name} found.")
      table_names.append(table.name)
    self._built = False
  @property
  def embedding_tables(self):
    raise NotImplementedError
  def _create_variables(self, table: tpu_embedding_v2_utils.TableConfig,
                        trainable: bool) -> Dict[Text, tf_variables.Variable]:
    variable_shape = (table.vocabulary_size, table.dim)
    def getter(name, shape, dtype, initializer, trainable):
      del shape
      initial_value = functools.partial(
          initializer, variable_shape, dtype=dtype)
      return tf_variables.Variable(
          name=name,
          initial_value=initial_value,
          shape=variable_shape,
          dtype=dtype,
          trainable=trainable)
    def variable_creator(name, initializer, trainable=True):
      return self._add_variable_with_custom_getter(
          name=name,
          initializer=initializer,
          shape=variable_shape,
          dtype=dtypes.float32,
          getter=getter,
          trainable=trainable)
    parameters = variable_creator(
        table.name, table.initializer, trainable=trainable)
    def slot_creator(name, initializer):
      return variable_creator(table.name + "/" + name, initializer, False)
    if table.optimizer is not None:
    else:
      slot_vars = {}
    slot_vars["parameters"] = parameters
    return slot_vars
  def _create_variables_and_slots(self):
    raise NotImplementedError
  def _gather_saveables_for_checkpoint(
      self) -> Dict[Text, Callable[[Text], saveable_hook.SaveableHook]]:
    def factory(name=_HOOK_KEY):
      return saveable_hook.SaveableHook(name)
    return {_HOOK_KEY: factory}
  def build(self):
    if self._built:
      return
    self._variables = self._create_variables_and_slots()
    self._built = True
  def __call__(self, features: Any, weights: Optional[Any] = None) -> Any:
    if not self._built:
      self.build()
    return self.embedding_lookup(features, weights)
  def embedding_lookup(self,
                       features: Any,
                       weights: Optional[Any] = None) -> Any:
    raise NotImplementedError
