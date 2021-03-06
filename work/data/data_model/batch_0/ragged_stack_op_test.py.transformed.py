
from absl.testing import parameterized
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest
@test_util.run_all_in_graph_and_eager_modes
class RaggedStackOpTest(test_util.TensorFlowTestCase,
                        parameterized.TestCase):
  @parameterized.parameters(
      dict(
          descr='One rank-2 input (ragged_rank=1), axis=0',
          rt_inputs=(
          axis=0,
          expected=[[[b'a00', b'a01'], [], [b'a20', b'a21']]]),
      dict(
          descr='One rank-2 input (ragged_rank=1), axis=1',
          rt_inputs=(
          axis=1,
          expected=[
              [[b'a00', b'a01']],
              [[]],
              [[b'a20', b'a21', b'a22']]]),
      dict(
          descr='One rank-2 input (ragged_rank=1), axis=2',
          rt_inputs=(
          axis=2,
          expected=[
              [[b'a00'], [b'a01']], [],
              [[b'a20'], [b'a21'], [b'a22']]]),
      dict(
          descr='One rank-2 input (ragged_rank=1), axis=-3',
          rt_inputs=(
          axis=-3,
          expected=[[[b'a00', b'a01'], [], [b'a20', b'a21']]]),
      dict(
          descr='One rank-2 input (ragged_rank=1), axis=-2',
          rt_inputs=(
          axis=-2,
          expected=[
              [[b'a00', b'a01']],
              [[]],
              [[b'a20', b'a21', b'a22']]]),
      dict(
          descr='One rank-2 input (ragged_rank=1), axis=-1',
          rt_inputs=(
          axis=-1,
          expected=[
              [[b'a00'], [b'a01']], [],
              [[b'a20'], [b'a21'], [b'a22']]]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=0',
          rt_inputs=(
          axis=0,
          expected=[[[b'a00', b'a01'], [], [b'a20', b'a21']], [[b'b00'],
                                                               [b'b10']]]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=1',
          rt_inputs=(
          axis=1,
          expected=[
              [[b'a00', b'a01'], [b'b00']],
              [[], [b'b10', b'b11', b'b12']],
              [[b'a20', b'a21', b'a22'], [b'b20']]]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=2',
          rt_inputs=(
          axis=2,
          expected=[
              [[b'a00', b'b00'], [b'a01', b'b01']], [],
              [[b'a20', b'b20'], [b'a21', b'b21'], [b'a22', b'b22']]]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=-3',
          rt_inputs=(
          axis=-3,
          expected=[[[b'a00', b'a01'], [], [b'a20', b'a21']], [[b'b00'],
                                                               [b'b10']]]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=-2',
          rt_inputs=(
          axis=-2,
          expected=[
              [[b'a00', b'a01'], [b'b00']],
              [[], [b'b10', b'b11', b'b12']],
              [[b'a20', b'a21', b'a22'], [b'b20']]]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=-1',
          rt_inputs=(
          axis=-1,
          expected=[
              [[b'a00', b'b00'], [b'a01', b'b01']], [],
              [[b'a20', b'b20'], [b'a21', b'b21'], [b'a22', b'b22']]]),
      dict(
          descr='Three rank-2 inputs (ragged_rank=1), axis=0',
          rt_inputs=(
          axis=0,
          expected=[[[b'a00', b'a01'], [], [b'a20', b'a21', b'a22']],
                    [[b'b00'], [b'b10']],
                    [[b'c00'], [b'c10', b'c11'], [b'c21']]]),
      dict(
          descr='Three rank-2 inputs (ragged_rank=1), axis=1',
          rt_inputs=(
          axis=1,
          expected=[
              [[b'a00', b'a01'], [b'b00'], []],
              [[], [b'b10', b'b11', b'b12'], [b'c10', b'c11']],
              [[b'a20', b'a21', b'a22'], [b'b20'], [b'c20', b'c21']]],
          expected_shape=[3, None, None]),
      dict(
          descr='Three rank-2 inputs (ragged_rank=1), axis=2',
          rt_inputs=(
          axis=2,
          expected=[
              [[b'a00', b'b00', b'c00'], [b'a01', b'b01', b'c01']], [],
              [[b'a20', b'b20', b'c20'], [b'a21', b'b21', b'c21'],
               [b'a22', b'b22', b'c22']]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=0',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']],
               [['a100', 'a101', 'a102'], ['a110', 'a111']]],
              [[['b000']], [['b100', 'b101'], ['b110']]],
              [[], [['c100', 'c101', 'c102', 'c103']], [[], ['c210', 'c211']]]),
          axis=0,
          expected=[
              [[[b'a000', b'a001'], [b'a010']],
               [[b'a100', b'a101', b'a102'], [b'a110', b'a111']]],
              [[[b'b000']],
               [[b'b100', b'b101'], [b'b110']]],
              [[],
               [[b'c100', b'c101', b'c102', b'c103']],
               [[], [b'c210', b'c211']]]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=1',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']],
               [['a100', 'a101', 'a102'], ['a110', 'a111']]],
              [[['b000']], [['b100', 'b101'], ['b110']]],
              [[], [[], ['c110', 'c111']]]),
          axis=1,
          expected=[
              [[[b'a000', b'a001'], [b'a010']], [[b'b000']], []],
              [[[b'a100', b'a101', b'a102'], [b'a110', b'a111']],
               [[b'b100', b'b101'], [b'b110']],
               [[], [b'c110', b'c111']]]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=2',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']],
               [['a100', 'a101', 'a102'], ['a110', 'a111']]],
              [[[], ['b010', 'b011']], [['b100', 'b101'], ['b110']]],
              [[['c000'], ['c010']], [[], ['c110', 'c111']]]),
          axis=2,
          expected=[
              [[[b'a000', b'a001'], [], [b'c000']],
               [[b'a010'], [b'b010', b'b011'], [b'c010']]],
              [[[b'a100', b'a101', b'a102'], [b'b100', b'b101'], []],
               [[b'a110', b'a111'], [b'b110'], [b'c110', b'c111']]]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=3',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']]],
              [[['b000', 'b001'], ['b010']]],
              [[['c000', 'c001'], ['c010']]]),
          axis=3,
          expected=[[
              [[b'a000', b'b000', b'c000'], [b'a001', b'b001', b'c001']],
              [[b'a010', b'b010', b'c010']]]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=-2',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']],
               [['a100', 'a101', 'a102'], ['a110', 'a111']]],
              [[[], ['b010', 'b011']], [['b100', 'b101'], ['b110']]],
              [[['c000'], ['c010']], [[], ['c110', 'c111']]]),
          axis=-2,
          expected=[
              [[[b'a000', b'a001'], [], [b'c000']],
               [[b'a010'], [b'b010', b'b011'], [b'c010']]],
              [[[b'a100', b'a101', b'a102'], [b'b100', b'b101'], []],
               [[b'a110', b'a111'], [b'b110'], [b'c110', b'c111']]]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=-1',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']]],
              [[['b000', 'b001'], ['b010']]],
              [[['c000', 'c001'], ['c010']]]),
          axis=-1,
          expected=[[
              [[b'a000', b'b000', b'c000'], [b'a001', b'b001', b'c001']],
              [[b'a010', b'b010', b'c010']]]]),
      dict(
          descr='ragged_stack([uniform, ragged, uniform], axis=1)',
          ragged_ranks=[0, 1, 0],
          rt_inputs=(
          axis=1,
          expected=[
              [[b'0('], [b'b00'], [b')0']],
              [[b'1('], [b'b10', b'b11', b'b12'], [b')1']],
              [[b'2('], [b'b20'], [b')2']]]),
      dict(
          descr='ragged_stack([uniform, uniform], axis=0)',
          ragged_ranks=[0, 0],
          rt_inputs=(
          axis=0,
          expected=[
              [[b'a00', b'a01'], [b'a10', b'a11'], [b'a20', b'a21']],
              [[b'b00', b'b01', b'b02'], [b'b10', b'b11', b'b12']]]),
      dict(
          descr='ragged_stack([1D, 1D], axis=0)',
          ragged_ranks=[0, 0],
          rt_inputs=(['a', 'b'], ['c', 'd', 'e']),
          axis=0,
          expected=[[b'a', b'b'], [b'c', b'd', b'e']]),
      dict(
          descr='ragged_stack([uniform, ragged], axis=0)',
          ragged_ranks=[0, 1],
          rt_inputs=(
          axis=0,
          expected=[
              [[b'a00', b'a01'], [b'a10', b'a11'], [b'a20', b'a21']],
              [[b'b00', b'b01', b'b02'], [b'b10', b'b11', b'b12']]]),
      dict(
          descr='ragged_stack([uniform, ragged], axis=0) with rank-3 inputs',
          ragged_ranks=[0, 2],
          rt_inputs=(
          axis=0,
          expected=[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8], [8, 8]]]]),
      dict(
          descr='Two rank-3 inputs with ragged_rank=1, axis=-1',
          ragged_ranks=[1, 1],
          rt_inputs=(
              [[[0, 1], [2, 3], [4, 5]], [], [[6, 7], [8, 9]]],
              [[[9, 8], [7, 6], [5, 4]], [], [[3, 2], [1, 0]]]),
          axis=-1,
          expected=[
              [[[0, 9], [1, 8]], [[2, 7], [3, 6]], [[4, 5], [5, 4]]],
              [],
              [[[6, 3], [7, 2]], [[8, 1], [9, 0]]]],
          expected_shape=[3, None, 2, 2]),
      dict(
          descr='Two rank-3 inputs with ragged_rank=1, axis=-2',
          ragged_ranks=[1, 1],
          rt_inputs=(
              [[[0, 1], [2, 3], [4, 5]], [], [[6, 7], [8, 9]]],
              [[[9, 8], [7, 6], [5, 4]], [], [[3, 2], [1, 0]]]),
          axis=-2,
          expected=[
              [[[0, 1], [9, 8]], [[2, 3], [7, 6]], [[4, 5], [5, 4]]], [],
              [[[6, 7], [3, 2]], [[8, 9], [1, 0]]]]),
      dict(
          descr='ragged_stack([vector, vector], axis=0)',
          ragged_ranks=[0, 0],
          rt_inputs=([1, 2, 3], [4, 5, 6]),
          axis=0,
          expected=[[1, 2, 3], [4, 5, 6]]),
      dict(
          descr='One input (so just adds an outer dimension)',
          rt_inputs=([['a00', 'a01'], [], ['a20', 'a21']],),
          axis=0,
          expected=[[[b'a00', b'a01'], [], [b'a20', b'a21']]]),
      dict(
          descr='One input (uniform 0D)',
          rt_inputs=(1,),
          ragged_ranks=[0],
          axis=0,
          expected=[1]),
      dict(
          descr='One input (uniform 1D)',
          rt_inputs=([1, 2],),
          ragged_ranks=[0],
          axis=0,
          expected=[[1, 2]],
          expected_ragged_rank=1),
      dict(
          descr='One input (uniform 2D)',
          rt_inputs=([[1, 2], [3, 4], [5, 6]],),
          ragged_ranks=[0],
          axis=0,
          expected=[[[1, 2], [3, 4], [5, 6]]],
          expected_ragged_rank=2),
  def testRaggedStack(self,
                      descr,
                      rt_inputs,
                      axis,
                      expected,
                      ragged_ranks=None,
                      expected_ragged_rank=None,
                      expected_shape=None):
    if ragged_ranks is None:
      ragged_ranks = [None] * len(rt_inputs)
    rt_inputs = [
        if rrank != 0 else constant_op.constant(rt_input)
        for (rt_input, rrank) in zip(rt_inputs, ragged_ranks)
    ]
    stacked = ragged_concat_ops.stack(rt_inputs, axis)
    if expected_ragged_rank is not None:
      self.assertEqual(stacked.ragged_rank, expected_ragged_rank)
    if expected_shape is not None:
      self.assertEqual(stacked.shape.as_list(), expected_shape)
    self.assertAllEqual(stacked, expected)
  @parameterized.parameters(
      dict(
          rt_inputs=(),
          axis=0,
          error=ValueError,
          message=r'rt_inputs may not be empty\.'),
      dict(
          rt_inputs=([[1, 2]], [[3, 4]]),
          axis=r'foo',
          error=TypeError,
          message='axis must be an int'),
      dict(
          rt_inputs=([[1, 2]], [[3, 4]]),
          axis=-4,
          error=ValueError,
          message='axis=-4 out of bounds: expected -3<=axis<3'),
      dict(
          rt_inputs=([[1, 2]], [[3, 4]]),
          axis=3,
          error=ValueError,
          message='axis=3 out of bounds: expected -3<=axis<3'),
  )
  def testError(self, rt_inputs, axis, error, message):
    self.assertRaisesRegex(error, message, ragged_concat_ops.stack, rt_inputs,
                           axis)
  def testSingleTensorInput(self):
    """Tests ragged_stack with a single tensor input.
    Usually, we pass a list of values in for rt_inputs.  However, you can
    also pass in a single value (as with tf.stack), in which case it is
    equivalent to expand_dims(axis=0).  This test exercises that path.
    """
    rt_inputs = ragged_factory_ops.constant([[1, 2], [3, 4]])
    stacked = ragged_concat_ops.stack(rt_inputs, 0)
    self.assertAllEqual(stacked, [[[1, 2], [3, 4]]])
if __name__ == '__main__':
  googletest.main()
