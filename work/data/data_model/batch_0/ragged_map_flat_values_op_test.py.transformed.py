
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
@test_util.run_all_in_graph_and_eager_modes
class RaggedMapInnerValuesOpTest(test_util.TensorFlowTestCase):
  def assertRaggedMapInnerValuesReturns(self,
                                        op,
                                        expected,
                                        args=(),
                                        kwargs=None):
    kwargs = kwargs or {}
    result = ragged_functional_ops.map_flat_values(op, *args, **kwargs)
    self.assertAllEqual(result, expected)
  def testDocStringExamples(self):
    rt = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5], [6]])
    v1 = ragged_functional_ops.map_flat_values(array_ops.ones_like, rt)
    v2 = ragged_functional_ops.map_flat_values(math_ops.multiply, rt, rt)
    v3 = ragged_functional_ops.map_flat_values(math_ops.add, rt, 5)
    self.assertAllEqual(v1, [[1, 1, 1], [], [1, 1], [1]])
    self.assertAllEqual(v2, [[1, 4, 9], [], [16, 25], [36]])
    self.assertAllEqual(v3, [[6, 7, 8], [], [9, 10], [11]])
  def testOpWithSingleRaggedTensorArg(self):
    tensor = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=array_ops.zeros_like,
        args=(tensor,),
        expected=[[0, 0, 0], [], [0, 0]])
  def testOpWithTwoRaggedTensorArgs(self):
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply, args=(x, y), expected=[[3, 2, 12], [], [4, 25]])
  def testOpWithRaggedTensorAndScalarArgs(self):
    y = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply, args=(5, y), expected=[[5, 10, 15], [], [20, 25]])
  def testOpWithThreeRaggedTensorArgs(self):
    condition = ragged_factory_ops.constant(
    x = ragged_factory_ops.constant([['a', 'b', 'c'], [], ['d', 'e']])
    y = ragged_factory_ops.constant([['A', 'B', 'C'], [], ['D', 'E']])
    self.assertRaggedMapInnerValuesReturns(
        op=array_ops.where_v2,
        args=(condition, x, y),
        expected=[[b'a', b'b', b'C'], [], [b'd', b'E']])
  def testOpWithRaggedTensorListArg(self):
    x = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    y = ragged_factory_ops.constant([[10, 20, 30], [], [40, 50]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.add_n,
        args=([x, y, x],),
        expected=[[12, 24, 36], [], [48, 60]])
  def testOpWithKeywordArgs(self):
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        kwargs=dict(x=x, y=y),
        expected=[[3, 2, 12], [], [4, 25]])
  def testOpWithMixedPositionalAndKeywordArgs(self):
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x,),
        kwargs=dict(y=y),
        expected=[[3, 2, 12], [], [4, 25]])
  def testNonElementWiseOp(self):
    x = ragged_factory_ops.constant(
        [[[3, 1, 4], [1, 5, 9], [2, 6, 5]], [], [[3, 5, 8], [9, 7, 9]]],
        ragged_rank=1)
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.reduce_sum,
        kwargs={
            'input_tensor': x,
            'axis': 1,
        },
        expected=[[8, 15, 13], [], [16, 25]])
  def testOpWithRaggedRankGreaterThanOne(self):
    x0 = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    y0 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    self.assertAllEqual(
        math_ops.multiply(x0, y0), [3, 2, 12, 4, 25, 54, 14, 48, 45])
    x1 = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5], [9, 2], [6, 5]])
    y1 = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5], [6, 7], [8, 9]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x1, y1),
        expected=[[3, 2, 12], [], [4, 25], [54, 14], [48, 45]])
    x2 = ragged_factory_ops.constant([[[3, 1, 4]], [], [[], [1, 5]],
                                      [[9, 2], [6, 5]]])
    y2 = ragged_factory_ops.constant([[[1, 2, 3]], [], [[], [4, 5]],
                                      [[6, 7], [8, 9]]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x2, y2),
    x3 = ragged_factory_ops.constant([[[[3, 1, 4]], []], [], [[[], [1, 5]]],
                                      [[[9, 2], [6, 5]]]])
    y3 = ragged_factory_ops.constant([[[[1, 2, 3]], []], [], [[[], [4, 5]]],
                                      [[[6, 7], [8, 9]]]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x3, y3),
        expected=[
  def testOpWithRaggedRankThree(self):
    x = ragged_factory_ops.constant([[[3, 1, 4]], [], [[], [1, 5]]])
    y = ragged_factory_ops.constant([[[1, 2, 3]], [], [[], [4, 5]]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x, y),
        expected=[[[3, 2, 12]], [], [[], [4, 25]]])
  def testOpWithInnerValuesOnly(self):
    x = constant_op.constant([[1, 2], [3, 4], [5, 6]])
    y = constant_op.constant(2)
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply, args=(x, y), expected=[[2, 4], [6, 8], [10, 12]])
  def testRaggedTensorSplitsRaggedRankMismatchError(self):
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[[3, 1, 4], []], [], [[1, 5]]])
    with self.assertRaisesRegex(
        ValueError, r'All ragged inputs must have the same ragged_rank.'):
      ragged_functional_ops.map_flat_values(math_ops.add, x, y)
  def testRaggedTensorSplitsValueMismatchError(self):
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[1], [2, 3], [4, 5]])
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r'partitions have incompatible'):
      ragged_functional_ops.map_flat_values(math_ops.add, x, y)
    z_splits = array_ops.placeholder_with_default(
        constant_op.constant([0, 3], dtypes.int64), None)
    z = ragged_tensor.RaggedTensor.from_row_splits([0, 1, 2], z_splits)
    with self.assertRaisesRegex(
        ValueError,
        r"Input RaggedTensors' flat_values must all have the same "
        r'outer-dimension size.  Got sizes: \{3, 5\}'):
      ragged_functional_ops.map_flat_values(math_ops.add, x, z)
  def testRaggedTensorShapeMismatchError(self):
    x = ragged_factory_ops.constant([[1, 2, 3], [4, 5]])
    with self.assertRaisesRegex(
        ValueError, r'tf.ragged.map_flat_values requires that the output of '
        '`op` have the same outer-dimension size as flat_values of any ragged '
        r'inputs. \(output shape: \(\); expected outer dimension size: 5\)'):
      ragged_functional_ops.map_flat_values(math_ops.argmax, x)
  def testRaggedTensorSplitsMismatchErrorAtRuntime(self):
    splits1 = array_ops.placeholder_with_default(
        constant_op.constant([0, 3, 3, 5], dtypes.int64), None)
    splits2 = array_ops.placeholder_with_default(
        constant_op.constant([0, 1, 3, 5], dtypes.int64), None)
    x = ragged_tensor.RaggedTensor.from_row_splits([3, 1, 4, 1, 5], splits1)
    y = ragged_tensor.RaggedTensor.from_row_splits([1, 2, 3, 4, 5], splits2)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r'partitions have incompatible'):
      self.evaluate(ragged_functional_ops.map_flat_values(math_ops.add, x, y))
  def testRaggedMapFnPreservesUniformRowLength(self):
    x = ragged_tensor.RaggedTensor.from_uniform_row_length(
        ragged_factory_ops.constant([[1, 2], [3]]), uniform_row_length=2)
    y = ragged_factory_ops.constant([[[1, 2], [3]]])
    a = ragged_functional_ops.map_flat_values(math_ops.add, x, y)
    self.assertAllEqual(x.uniform_row_length, a.uniform_row_length)
    b = ragged_functional_ops.map_flat_values(math_ops.add, y, x)
    self.assertAllEqual(x.uniform_row_length, b.uniform_row_length)
    c = ragged_functional_ops.map_flat_values(math_ops.add_n, [x, x])
    self.assertAllEqual(x.uniform_row_length, c.uniform_row_length)
    d = ragged_functional_ops.map_flat_values(math_ops.add_n, [y, x, y])
    self.assertAllEqual(x.uniform_row_length, d.uniform_row_length)
if __name__ == '__main__':
  googletest.main()
