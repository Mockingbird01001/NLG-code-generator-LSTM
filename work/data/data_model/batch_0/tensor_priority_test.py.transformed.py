
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.platform import test as test_lib
class TensorPriorityTest(test_lib.TestCase):
  def testSupportedRhsWithoutDelegation(self):
    class NumpyArraySubclass(np.ndarray):
      pass
    supported_rhs_without_delegation = (3, 3.0, [1.0, 2.0], np.array(
        [1.0, 2.0]), NumpyArraySubclass(
            shape=(1, 2), buffer=np.array([1.0, 2.0])),
                                        ops.convert_to_tensor([[1.0, 2.0]]))
    for rhs in supported_rhs_without_delegation:
      tensor = ops.convert_to_tensor([[10.0, 20.0]])
      res = tensor + rhs
      self.assertIsInstance(res, ops.Tensor)
  def testUnsupportedRhsWithoutDelegation(self):
    class WithoutReverseAdd(object):
      pass
    tensor = ops.convert_to_tensor([[10.0, 20.0]])
    rhs = WithoutReverseAdd()
    with self.assertRaisesWithPredicateMatch(
        TypeError, lambda e: "Expected float" in str(e)):
      tensor + rhs
  def testUnsupportedRhsWithDelegation(self):
    class WithReverseAdd(object):
      def __radd__(self, lhs):
        return "Works!"
    tensor = ops.convert_to_tensor([[10.0, 20.0]])
    rhs = WithReverseAdd()
    res = tensor + rhs
    self.assertEqual(res, "Works!")
  def testFullDelegationControlUsingRegistry(self):
    class NumpyArraySubclass(np.ndarray):
      def __radd__(self, lhs):
        return "Works!"
    def raise_to_delegate(value, dtype=None, name=None, as_ref=False):
      raise TypeError
    ops.register_tensor_conversion_function(
        NumpyArraySubclass, raise_to_delegate, priority=0)
    tensor = ops.convert_to_tensor([[10.0, 20.0]])
    rhs = NumpyArraySubclass(shape=(1, 2), buffer=np.array([1.0, 2.0]))
    res = tensor + rhs
    self.assertEqual(res, "Works!")
if __name__ == "__main__":
  test_lib.main()
