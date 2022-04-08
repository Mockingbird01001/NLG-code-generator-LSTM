
import collections
import os
import time
from tensorflow.python.keras.saving.utils_v1 import export_output as export_output_lib
from tensorflow.python.keras.saving.utils_v1 import mode_keys
from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.keras.saving.utils_v1.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
EXPORT_TAG_MAP = mode_keys.ModeKeyMap(**{
    ModeKeys.PREDICT: [tag_constants.SERVING],
    ModeKeys.TRAIN: [tag_constants.TRAINING],
    ModeKeys.TEST: [unexported_constants.EVAL]})
SIGNATURE_KEY_MAP = mode_keys.ModeKeyMap(**{
    ModeKeys.PREDICT: signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    ModeKeys.TRAIN: unexported_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY,
    ModeKeys.TEST: unexported_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY})
SINGLE_FEATURE_DEFAULT_NAME = 'feature'
SINGLE_RECEIVER_DEFAULT_NAME = 'input'
SINGLE_LABEL_DEFAULT_NAME = 'label'
def build_all_signature_defs(receiver_tensors,
                             export_outputs,
                             receiver_tensors_alternatives=None,
                             serving_only=True):
  """Build `SignatureDef`s for all export outputs.
  Args:
    receiver_tensors: a `Tensor`, or a dict of string to `Tensor`, specifying
      input nodes where this receiver expects to be fed by default.  Typically,
      this is a single placeholder expecting serialized `tf.Example` protos.
    export_outputs: a dict of ExportOutput instances, each of which has
      an as_signature_def instance method that will be called to retrieve
      the signature_def for all export output tensors.
    receiver_tensors_alternatives: a dict of string to additional
      groups of receiver tensors, each of which may be a `Tensor` or a dict of
      string to `Tensor`.  These named receiver tensor alternatives generate
      additional serving signatures, which may be used to feed inputs at
      different points within the input receiver subgraph.  A typical usage is
      to allow feeding raw feature `Tensor`s *downstream* of the
      tf.io.parse_example() op.  Defaults to None.
    serving_only: boolean; if true, resulting signature defs will only include
      valid serving signatures. If false, all requested signatures will be
      returned.
  Returns:
    signature_def representing all passed args.
  Raises:
    ValueError: if export_outputs is not a dict
  """
  if not isinstance(receiver_tensors, dict):
    receiver_tensors = {SINGLE_RECEIVER_DEFAULT_NAME: receiver_tensors}
  if export_outputs is None or not isinstance(export_outputs, dict):
    raise ValueError('export_outputs must be a dict and not'
                     '{}'.format(type(export_outputs)))
  signature_def_map = {}
  excluded_signatures = {}
  for output_key, export_output in export_outputs.items():
    signature_name = '{}'.format(output_key or 'None')
    try:
      signature = export_output.as_signature_def(receiver_tensors)
      signature_def_map[signature_name] = signature
    except ValueError as e:
      excluded_signatures[signature_name] = str(e)
  if receiver_tensors_alternatives:
    for receiver_name, receiver_tensors_alt in (
        receiver_tensors_alternatives.items()):
      if not isinstance(receiver_tensors_alt, dict):
        receiver_tensors_alt = {
            SINGLE_RECEIVER_DEFAULT_NAME: receiver_tensors_alt
        }
      for output_key, export_output in export_outputs.items():
        signature_name = '{}:{}'.format(receiver_name or 'None', output_key or
                                        'None')
        try:
          signature = export_output.as_signature_def(receiver_tensors_alt)
          signature_def_map[signature_name] = signature
        except ValueError as e:
          excluded_signatures[signature_name] = str(e)
  _log_signature_report(signature_def_map, excluded_signatures)
  if serving_only:
    signature_def_map = {
        k: v
        for k, v in signature_def_map.items()
        if signature_def_utils.is_valid_signature(v)
    }
  return signature_def_map
_FRIENDLY_METHOD_NAMES = {
    signature_constants.CLASSIFY_METHOD_NAME: 'Classify',
    signature_constants.REGRESS_METHOD_NAME: 'Regress',
    signature_constants.PREDICT_METHOD_NAME: 'Predict',
    unexported_constants.SUPERVISED_TRAIN_METHOD_NAME: 'Train',
    unexported_constants.SUPERVISED_EVAL_METHOD_NAME: 'Eval',
}
def _log_signature_report(signature_def_map, excluded_signatures):
  sig_names_by_method_name = collections.defaultdict(list)
  for method_name in _FRIENDLY_METHOD_NAMES:
    sig_names_by_method_name[method_name] = []
  for signature_name, sig in signature_def_map.items():
    sig_names_by_method_name[sig.method_name].append(signature_name)
  for method_name, sig_names in sig_names_by_method_name.items():
    if method_name in _FRIENDLY_METHOD_NAMES:
      method_name = _FRIENDLY_METHOD_NAMES[method_name]
    logging.info('Signatures INCLUDED in export for {}: {}'.format(
        method_name, sig_names if sig_names else 'None'))
  if excluded_signatures:
    logging.info('Signatures EXCLUDED from export because they cannot be '
                 'be served via TensorFlow Serving APIs:')
    for signature_name, message in excluded_signatures.items():
      logging.info('\'{}\' : {}'.format(signature_name, message))
  if not signature_def_map:
    logging.warning('Export includes no signatures!')
  elif (signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY not in
        signature_def_map):
    logging.warning('Export includes no default signature!')
MAX_DIRECTORY_CREATION_ATTEMPTS = 10
def get_timestamped_export_dir(export_dir_base):
  """Builds a path to a new subdirectory within the base directory.
  Each export is written into a new subdirectory named using the
  current time.  This guarantees monotonically increasing version
  numbers even across multiple runs of the pipeline.
  The timestamp used is the number of seconds since epoch UTC.
  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
  Returns:
    The full path of the new subdirectory (which is not actually created yet).
  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
  attempts = 0
  while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
    timestamp = int(time.time())
    result_dir = os.path.join(
        compat.as_bytes(export_dir_base), compat.as_bytes(str(timestamp)))
    if not gfile.Exists(result_dir):
      return result_dir
    time.sleep(1)
    attempts += 1
    logging.warning(
        'Directory {} already exists; retrying (attempt {}/{})'.format(
            compat.as_str(result_dir), attempts,
            MAX_DIRECTORY_CREATION_ATTEMPTS))
  raise RuntimeError('Failed to obtain a unique export directory name after '
                     '{} attempts.'.format(MAX_DIRECTORY_CREATION_ATTEMPTS))
def get_temp_export_dir(timestamped_export_dir):
  (dirname, basename) = os.path.split(timestamped_export_dir)
  if isinstance(basename, bytes):
    str_name = basename.decode('utf-8')
  else:
    str_name = str(basename)
  temp_export_dir = os.path.join(
      compat.as_bytes(dirname),
      compat.as_bytes('temp-{}'.format(str_name)))
  return temp_export_dir
def export_outputs_for_mode(
    mode, serving_export_outputs=None, predictions=None, loss=None,
    metrics=None):
  """Util function for constructing a `ExportOutput` dict given a mode.
  The returned dict can be directly passed to `build_all_signature_defs` helper
  function as the `export_outputs` argument, used for generating a SignatureDef
  map.
  Args:
    mode: A `ModeKeys` specifying the mode.
    serving_export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict or None.
    predictions: A dict of Tensors or single Tensor representing model
        predictions. This argument is only used if serving_export_outputs is not
        set.
    loss: A dict of Tensors or single Tensor representing calculated loss.
    metrics: A dict of (metric_value, update_op) tuples, or a single tuple.
      metric_value must be a Tensor, and update_op must be a Tensor or Op
  Returns:
    Dictionary mapping the a key to an `tf.estimator.export.ExportOutput` object
    The key is the expected SignatureDef key for the mode.
  Raises:
    ValueError: if an appropriate ExportOutput cannot be found for the mode.
  """
  if mode not in SIGNATURE_KEY_MAP:
    raise ValueError(
        'Export output type not found for mode: {}. Expected one of: {}.\n'
        'One likely error is that V1 Estimator Modekeys were somehow passed to '
        'this function. Please ensure that you are using the new ModeKeys.'
        .format(mode, SIGNATURE_KEY_MAP.keys()))
  signature_key = SIGNATURE_KEY_MAP[mode]
  if mode_keys.is_predict(mode):
    return get_export_outputs(serving_export_outputs, predictions)
  elif mode_keys.is_train(mode):
    return {signature_key: export_output_lib.TrainOutput(
        loss=loss, predictions=predictions, metrics=metrics)}
  else:
    return {signature_key: export_output_lib.EvalOutput(
        loss=loss, predictions=predictions, metrics=metrics)}
def get_export_outputs(export_outputs, predictions):
  if export_outputs is None:
    default_output = export_output_lib.PredictOutput(predictions)
    export_outputs = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: default_output}
  if not isinstance(export_outputs, dict):
    raise TypeError('export_outputs must be dict, given: {}'.format(
        export_outputs))
  for v in export_outputs.values():
    if not isinstance(v, export_output_lib.ExportOutput):
      raise TypeError(
          'Values in export_outputs must be ExportOutput objects. '
          'Given: {}'.format(export_outputs))
  _maybe_add_default_serving_output(export_outputs)
  return export_outputs
def _maybe_add_default_serving_output(export_outputs):
  if len(export_outputs) == 1:
    (key, value), = export_outputs.items()
    if key != signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
      export_outputs[
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = value
  if len(export_outputs) > 1:
    if (signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        not in export_outputs):
      raise ValueError(
          'Multiple export_outputs were provided, but none of them is '
          'specified as the default.  Do this by naming one of them with '
          'signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY.')
  return export_outputs
