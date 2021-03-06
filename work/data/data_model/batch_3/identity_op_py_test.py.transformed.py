
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
class IdentityOpTest(test.TestCase):
  @test_util.run_v1_only("Don't need to test VariableV1 in TF2.")
  def testRefIdentityShape(self):
    shape = [2, 3]
    tensor = variables.VariableV1(
        constant_op.constant(
            [[1, 2, 3], [6, 5, 4]], dtype=dtypes.int32))
    self.assertEqual(shape, tensor.get_shape())
    self.assertEqual(shape, gen_array_ops.ref_identity(tensor).get_shape())
if __name__ == "__main__":
  test.main()
