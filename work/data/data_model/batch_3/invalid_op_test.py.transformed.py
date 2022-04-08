
import os
from tensorflow.python.framework import errors
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
class InvalidOpTest(test.TestCase):
  def testBasic(self):
    library_filename = os.path.join(resource_loader.get_data_files_path(),
                                    'invalid_op.so')
    with self.assertRaises(errors.InvalidArgumentError):
      load_library.load_op_library(library_filename)
if __name__ == '__main__':
  test.main()
