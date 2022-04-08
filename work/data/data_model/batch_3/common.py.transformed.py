
import tempfile
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf
flags.DEFINE_string('save_model_path', '',
                    'Path to save the model to.')
FLAGS = flags.FLAGS
def do_test(create_module_fn, exported_names=None, show_debug_info=False):
  """Runs test.
  1. Performs absl and tf "main"-like initialization that must run before almost
     anything else.
  2. Converts `tf.Module` to SavedModel
  3. Converts SavedModel to MLIR
  4. Prints the textual MLIR to stdout (it is expected that the caller will have
     FileCheck checks in its file to check this output).
  This is only for use by the MLIR SavedModel importer tests.
  Args:
    create_module_fn: A callable taking no arguments, which returns the
      `tf.Module` to be converted and printed.
    exported_names: A set of exported names for the MLIR converter (default is
      "export all").
    show_debug_info: If true, shows debug locations in the resulting MLIR.
  """
  if exported_names is None:
    exported_names = []
  logging.set_stderrthreshold('error')
  tf.enable_v2_behavior()
  def app_main(argv):
    if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')
    if FLAGS.save_model_path:
      save_model_path = FLAGS.save_model_path
    else:
      save_model_path = tempfile.mkdtemp(suffix='.saved_model')
    save_options = tf.saved_model.SaveOptions(save_debug_info=show_debug_info)
    tf.saved_model.save(
        create_module_fn(), save_model_path, options=save_options)
    logging.info('Saved model to: %s', save_model_path)
    mlir = pywrap_mlir.experimental_convert_saved_model_to_mlir(
        save_model_path, ','.join(exported_names), show_debug_info)
    mlir = pywrap_mlir.experimental_run_pass_pipeline(mlir, 'canonicalize',
                                                      show_debug_info)
    print(mlir)
  app.run(app_main)
