
"""Contains the get_layer_policy function.
This is a separate file from policy.py to avoid a circular dependency.
get_layer_policy() relies on base_layer.py, itself which relies on policy.py.
"""
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.mixed_precision.experimental.get_layer_policy', v1=[])
def get_layer_policy(layer):
  if not isinstance(layer, base_layer.Layer):
    raise ValueError('get_policy can only be called on a layer, but got: %s'
                     % (layer,))
  return layer.dtype_policy
