
from tensorflow.python.util import lazy_loader
normalization = lazy_loader.LazyLoader(
    'normalization', globals(),
    'keras.legacy_tf_layers.normalization')
def __getattr__(name):
  if name in ['BatchNormalization', 'BatchNorm']:
    return normalization.BatchNormalization
  elif name in ['batch_normalization', 'batch_norm']:
    return normalization.batch_normalization
  else:
    raise AttributeError(f'module {__name__} doesn\'t have attribute {name}')
