
import functools
from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils
KERAS_MODEL_TYPES = ['functional', 'subclass', 'sequential']
def keras_mode_combinations(mode=None, run_eagerly=None):
  if mode is None:
    mode = ['eager'] if tf2.enabled() else ['graph', 'eager']
  if run_eagerly is None:
    run_eagerly = [True, False]
  result = []
  if 'eager' in mode:
    result += combinations.combine(mode=['eager'], run_eagerly=run_eagerly)
  if 'graph' in mode:
    result += combinations.combine(mode=['graph'], run_eagerly=[False])
  return result
def keras_model_type_combinations():
  return combinations.combine(model_type=KERAS_MODEL_TYPES)
class KerasModeCombination(test_combinations.TestCombination):
  def context_managers(self, kwargs):
    run_eagerly = kwargs.pop('run_eagerly', None)
    if run_eagerly is not None:
      return [testing_utils.run_eagerly_scope(run_eagerly)]
    else:
      return []
  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter('run_eagerly')]
class KerasModelTypeCombination(test_combinations.TestCombination):
  def context_managers(self, kwargs):
    model_type = kwargs.pop('model_type', None)
    if model_type in KERAS_MODEL_TYPES:
      return [testing_utils.model_type_scope(model_type)]
    else:
      return []
  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter('model_type')]
_defaults = combinations.generate.keywords['test_combinations']
generate = functools.partial(
    combinations.generate,
    test_combinations=_defaults +
    (KerasModeCombination(), KerasModelTypeCombination()))
combine = test_combinations.combine
times = test_combinations.times
NamedObject = test_combinations.NamedObject
