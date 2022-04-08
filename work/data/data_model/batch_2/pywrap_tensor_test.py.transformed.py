
import numpy as np
from tensorflow.python.eager import pywrap_tensor_test_util as util
from tensorflow.python.eager import test
class PywrapTensorTest(test.TestCase):
  def testGetScalarOne(self):
    result = util.get_scalar_one()
    self.assertIsInstance(result, np.ndarray)
    self.assertAllEqual(result, 1.0)
if __name__ == "__main__":
  test.main()
