
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.distribute import simple_models
simple_functional_model = combinations.NamedObject(
    "SimpleFunctionalModel", simple_models.SimpleFunctionalModel())
simple_sequential_model = combinations.NamedObject(
    "SimpleSequentialModel", simple_models.SimpleSequentialModel())
simple_subclass_model = combinations.NamedObject(
    "SimpleSubclassModel", simple_models.SimpleSubclassModel())
simple_tfmodule_model = combinations.NamedObject(
    "SimpleTFModuleModel", simple_models.SimpleTFModuleModel())
