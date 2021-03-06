
from absl import app
from absl import flags
from absl import logging
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS
class ToyModule(module.Module):
  def __init__(self):
    super(ToyModule, self).__init__()
    self.w = variables.Variable(constant_op.constant([[1], [2], [3]]), name='w')
  @def_function.function(input_signature=[
      tensor_spec.TensorSpec([1, 3], dtypes.int32, name='input')
  ])
  def toy(self, x):
    r = math_ops.matmul(x, self.w, name='result')
    return r
def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  v2_compat.enable_v2_behavior()
  save.save(
      ToyModule(),
      FLAGS.saved_model_path,
      options=save_options.SaveOptions(save_debug_info=False))
  logging.info('Saved model to: %s', FLAGS.saved_model_path)
if __name__ == '__main__':
  app.run(main)
