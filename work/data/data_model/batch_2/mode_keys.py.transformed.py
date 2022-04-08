
  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'infer'
def is_predict(mode):
  return mode in [KerasModeKeys.PREDICT, EstimatorModeKeys.PREDICT]
def is_eval(mode):
  return mode in [KerasModeKeys.TEST, EstimatorModeKeys.EVAL]
def is_train(mode):
  return mode in [KerasModeKeys.TRAIN, EstimatorModeKeys.TRAIN]
class ModeKeyMap(collections.abc.Mapping):
  def __init__(self, **kwargs):
    self._internal_dict = {}
    self._keys = []
    for key in kwargs:
      self._keys.append(key)
      dict_key = self._get_internal_key(key)
      if dict_key in self._internal_dict:
        raise ValueError(
            'Error creating ModeKeyMap. Multiple keys/values found for {} mode.'
            .format(dict_key))
      self._internal_dict[dict_key] = kwargs[key]
  def _get_internal_key(self, key):
    if is_train(key):
      return KerasModeKeys.TRAIN
    if is_eval(key):
      return KerasModeKeys.TEST
    if is_predict(key):
      return KerasModeKeys.PREDICT
    raise ValueError('Invalid mode key: {}.'.format(key))
  def __getitem__(self, key):
    return self._internal_dict[self._get_internal_key(key)]
  def __iter__(self):
    return iter(self._keys)
  def __len__(self):
    return len(self._keys)
