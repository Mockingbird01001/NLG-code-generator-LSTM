
from tensorflow.dtensor import python as dtensor
from tensorflow.python.platform import test
class VerifyDTensorAPITest(test.TestCase):
  def testDTensorAPI(self):
    self.assertTrue(hasattr(dtensor, "call_with_layout"))
if __name__ == "__main__":
  test.main()
