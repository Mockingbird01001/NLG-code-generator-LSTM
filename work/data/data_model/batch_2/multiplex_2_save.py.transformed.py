
import os
import shutil
from absl import app
from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op
from tensorflow.examples.custom_ops_doc.multiplex_4 import model_using_multiplex
def main(argv):
  path = 'model_using_multiplex'
  if os.path.exists(path):
    shutil.rmtree(path, ignore_errors=True)
  model_using_multiplex.save(multiplex_2_op.multiplex, path)
  print('Saved model to', path)
if __name__ == '__main__':
  app.run(main)
