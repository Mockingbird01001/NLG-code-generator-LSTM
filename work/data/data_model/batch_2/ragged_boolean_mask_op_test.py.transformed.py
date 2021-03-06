
from absl.testing import parameterized
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest
@test_util.run_all_in_graph_and_eager_modes
class RaggedBooleanMaskOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  T = True
  F = False
  @parameterized.parameters([
      dict(
          descr='Docstring example 1',
          data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          mask=[[T, F, T], [F, F, F], [T, F, F]],
          expected=ragged_factory_ops.constant_value([[1, 3], [], [7]])),
      dict(
          descr='Docstring example 2',
          data=ragged_factory_ops.constant_value([[1, 2, 3], [4], [5, 6]]),
          mask=ragged_factory_ops.constant_value([[F, F, T], [F], [T, T]]),
          expected=ragged_factory_ops.constant_value([[3], [], [5, 6]])),
      dict(
          descr='Docstring example 3',
          data=ragged_factory_ops.constant_value([[1, 2, 3], [4], [5, 6]]),
          mask=[True, False, True],
          expected=ragged_factory_ops.constant_value([[1, 2, 3], [5, 6]])),
      dict(
          descr='data.shape=[7]; mask.shape=[7]',
          data=[1, 2, 3, 4, 5, 6, 7],
          mask=[T, F, T, T, F, F, F],
          expected=[1, 3, 4]),
      dict(
          descr='data.shape=[5, 3]; mask.shape=[5]',
          data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
          mask=[True, False, True, True, False],
          expected=[[1, 2, 3], [7, 8, 9], [10, 11, 12]]),
      dict(
          descr='data.shape=[5, 3]; mask.shape=[5, 3]',
          data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2], [3, 4, 5]],
          mask=[[F, F, F], [T, F, T], [T, T, T], [F, F, F], [T, T, F]],
          expected=ragged_factory_ops.constant_value(
              [[], [4, 6], [7, 8, 9], [], [3, 4]])),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3]',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[F, F, T],
          expected=[[[2, 4], [6, 8]]]),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3]',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[F, F, T],
          expected=[[[2, 4], [6, 8]]]),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3, 2]',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[[T, F], [T, T], [F, F]],
          expected=ragged_factory_ops.constant_value(
              [[[1, 2]], [[5, 6], [7, 8]], []],
              ragged_rank=1)),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3, 2, 2]',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[[[T, T], [F, T]], [[F, F], [F, F]], [[T, F], [T, T]]],
          expected=ragged_factory_ops.constant_value(
              [[[1, 2], [4]], [[], []], [[2], [6, 8]]])),
      dict(
          descr='data.shape=mask.shape=[2, 2, 2, 2]',
          data=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[2, 4], [6, 8]], [[1, 3], [5, 7]]]],
          mask=[[[[T, T], [F, F]], [[T, F], [F, F]]],
                [[[F, F], [F, F]], [[T, T], [T, F]]]],
          expected=ragged_factory_ops.constant_value(
              [[[[1, 2], []], [[5], []]], [[[], []], [[1, 3], [5]]]])),
      dict(
          descr='data.shape=[5, (D2)]; mask.shape=[5, (D2)]',
          data=ragged_factory_ops.constant_value(
              [[1, 2], [3, 4, 5, 6], [7, 8, 9], [], [1, 2, 3]]),
          mask=ragged_factory_ops.constant_value(
              [[F, F], [F, T, F, T], [F, F, F], [], [T, F, T]]),
          expected=ragged_factory_ops.constant_value(
              [[], [4, 6], [], [], [1, 3]])),
      dict(
          descr='data.shape=[3, (D2), (D3)]; mask.shape=[3, (D2)]',
          data=ragged_factory_ops.constant_value(
              [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]]),
          mask=ragged_factory_ops.constant_value([[T, F], [T, T], [F, F]]),
          expected=ragged_factory_ops.constant_value(
              [[[1, 2]], [[5, 6], [7, 8]], []])),
      dict(
          descr='data.shape=[3, (D2), D3]; mask.shape=[3, (D2)]',
          data=ragged_factory_ops.constant_value(
              [[[1, 2], [3, 4]], [[5, 6], [7, 8], [2, 4]], [[6, 8]]],
              ragged_rank=1),
          mask=ragged_factory_ops.constant_value([[T, F], [T, T, F], [F]]),
          expected=ragged_factory_ops.constant_value(
              [[[1, 2]], [[5, 6], [7, 8]], []],
              ragged_rank=1)),
      dict(
          descr='data.shape=[3, (D2), (D3)]; mask.shape=[3, (D2), (D3)]',
          data=ragged_factory_ops.constant_value(
              [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4]]]),
          mask=ragged_factory_ops.constant_value(
              [[[T, T], [F, T]], [[F, F], [F, F]], [[T, F]]]),
          expected=ragged_factory_ops.constant_value(
              [[[1, 2], [4]], [[], []], [[2]]])),
      dict(
          descr=('data.shape=[3, (D2), (D3), (D4)]; '
                 'mask.shape=[3, (D2), (D3), (D4)]'),
          data=ragged_factory_ops.constant_value(
              [[[[1, 2], [3, 4]], [[5, 6]]], [[[2, 4], [6, 8]]]]),
          mask=ragged_factory_ops.constant_value(
              [[[[T, T], [F, F]], [[T, F]]], [[[F, F], [T, T]]]]),
          expected=ragged_factory_ops.constant_value(
              [[[[1, 2], []], [[5]]], [[[], [6, 8]]]])),
      dict(
          descr='data.shape=[2, 3]; mask.shape=[2, (3)]',
          data=[[1, 2, 3], [4, 5, 6]],
          mask=ragged_factory_ops.constant_value([[T, F, F], [F, T, T]]),
          expected=ragged_factory_ops.constant_value([[1], [5, 6]])),
      dict(
          descr='data.shape=[2, 3, 2]; mask.shape=[2, (3)]',
          data=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 0], [2, 4]]],
          mask=ragged_factory_ops.constant_value([[T, F, F], [F, T, T]]),
          expected=ragged_factory_ops.constant_value(
              [[[1, 2]], [[9, 0], [2, 4]]],
              ragged_rank=1)),
      dict(
          descr='data.shape=[2, 3, 2]; mask.shape=[2, (3), 2]',
          data=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 0], [2, 4]]],
          mask=ragged_factory_ops.constant_value(
              [[[T, F], [F, F], [T, T]], [[T, F], [F, T], [F, F]]],
              ragged_rank=1),
          expected=ragged_factory_ops.constant_value(
              [[[1], [], [5, 6]], [[7], [0], []]])),
      dict(
          descr='data.shape=[4, (D2)]; mask.shape=[4]',
          data=ragged_factory_ops.constant_value([[1, 2, 3], [4], [], [5, 6]]),
          mask=[T, F, T, F],
          expected=ragged_factory_ops.constant_value([[1, 2, 3], []])),
      dict(
          descr='data.shape=[4, (D2), (D3)]; mask.shape=[4]',
          data=ragged_factory_ops.constant_value(
              [[[1, 2, 3]], [[4], []], [[5, 6]], []]),
          mask=[T, F, T, T],
          expected=ragged_factory_ops.constant_value(
              [[[1, 2, 3]], [[5, 6]], []])),
      dict(
          descr='data.shape=[4, (D2), 2]; mask.shape=[4]',
          data=ragged_factory_ops.constant_value(
              [[[1, 2], [3, 4]], [], [[5, 6]], [[7, 8], [9, 0], [1, 2]]],
              ragged_rank=1),
          mask=[T, F, F, T],
          expected=ragged_factory_ops.constant_value(
              [[[1, 2], [3, 4]], [[7, 8], [9, 0], [1, 2]]],
              ragged_rank=1)),
      dict(
          descr='data.shape=[4, (D2), 2]; mask.shape=[4]',
          data=ragged_factory_ops.constant_value(
              [[[1, 2], [3, 4]], [], [[5, 6]], [[7, 8], [9, 0], [1, 2]]],
              ragged_rank=1),
          mask=[T, F, F, T],
          expected=ragged_factory_ops.constant_value(
              [[[1, 2], [3, 4]], [[7, 8], [9, 0], [1, 2]]],
              ragged_rank=1)),
      dict(
          descr='data.shape=[1, (2)]; mask.shape=[1, 2]',
          data=ragged_factory_ops.constant_value([[1, 2]]),
          mask=[[T, F]],
          expected=ragged_factory_ops.constant_value([[1]])),
      dict(
          descr='data.shape=[2, (2), (D3)]; mask.shape=[2, 2]',
          data=ragged_factory_ops.constant_value(
              [[[1], [2, 3]], [[], [4, 5, 6]]]),
          mask=[[T, F], [T, T]],
          expected=ragged_factory_ops.constant_value([[[1]], [[], [4, 5, 6]]])),
      dict(
          descr='data.shape=[2, (2), 3]; mask.shape=[2, 2]',
          data=ragged_factory_ops.constant_value(
              [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [2, 4, 6]]],
              ragged_rank=1),
          mask=[[T, F], [T, T]],
          expected=ragged_factory_ops.constant_value(
              [[[1, 2, 3]], [[7, 8, 9], [2, 4, 6]]],
              ragged_rank=1)),
      dict(
          descr='data.shape=[2, (2), 3]; mask.shape=[2, 2, 3]',
          data=ragged_factory_ops.constant_value(
              [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [2, 4, 6]]],
              ragged_rank=1),
          mask=[[[T, F, F], [T, F, T]], [[T, F, T], [F, F, F]]],
          expected=ragged_factory_ops.constant_value(
              [[[1], [4, 6]], [[7, 9], []]])),
  def testBooleanMask(self, descr, data, mask, expected):
    actual = ragged_array_ops.boolean_mask(data, mask)
    self.assertAllEqual(actual, expected)
  def testErrors(self):
    if not context.executing_eagerly():
      self.assertRaisesRegex(ValueError,
                             r'mask\.shape\.ndims must be known statically',
                             ragged_array_ops.boolean_mask, [[1, 2]],
                             array_ops.placeholder(dtypes.bool))
    self.assertRaises(TypeError, ragged_array_ops.boolean_mask, [[1, 2]],
                      [[0, 1]])
    self.assertRaisesRegex(
        ValueError, 'Tensor conversion requested dtype bool for '
        'RaggedTensor with dtype int32', ragged_array_ops.boolean_mask,
        ragged_factory_ops.constant([[1, 2]]),
        ragged_factory_ops.constant([[0, 0]]))
    self.assertRaisesRegex(ValueError,
                           r'Shapes \(1, 2\) and \(1, 3\) are incompatible',
                           ragged_array_ops.boolean_mask, [[1, 2]],
                           [[True, False, True]])
    self.assertRaisesRegex(errors.InvalidArgumentError,
                           r'Inputs must have identical ragged splits',
                           ragged_array_ops.boolean_mask,
                           ragged_factory_ops.constant([[1, 2]]),
                           ragged_factory_ops.constant([[True, False, True]]))
    self.assertRaisesRegex(ValueError, 'mask cannot be scalar',
                           ragged_array_ops.boolean_mask, [[1, 2]], True)
    self.assertRaisesRegex(ValueError, 'mask cannot be scalar',
                           ragged_array_ops.boolean_mask,
                           ragged_factory_ops.constant([[1, 2]]), True)
if __name__ == '__main__':
  googletest.main()
