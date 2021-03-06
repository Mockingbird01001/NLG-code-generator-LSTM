
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_where_op
from tensorflow.python.platform import googletest
@test_util.run_all_in_graph_and_eager_modes
class RaggedWhereV1OpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.parameters([
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          expected=[[0, 0], [0, 2], [1, 1]]),
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          x=ragged_factory_ops.constant_value(
              [['A', 'B', 'C'], ['D', 'E']]),
          y=ragged_factory_ops.constant_value(
              [['a', 'b', 'c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'b', b'C'], [b'd', b'E']])),
          condition=ragged_factory_ops.constant_value([True, False]),
          x=ragged_factory_ops.constant_value([['A', 'B', 'C'], ['D', 'E']]),
          y=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'B', b'C'], [b'd', b'e']])),
          condition=[True, False, True, False, True],
          expected=[[0], [2], [4]]),
          condition=[[True, False], [False, True]],
          expected=[[0, 0], [1, 1]]),
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          expected=[[0, 0], [0, 2], [1, 1]]),
          condition=ragged_factory_ops.constant_value([
              [[True, False, True], [False, True]],
              [[True], [], [False], [False, True, False]]
          ]),
          expected=[[0, 0, 0], [0, 0, 2], [0, 1, 1],
                    [1, 0, 0], [1, 3, 1]]),
          condition=ragged_factory_ops.constant_value([
              [[True, False], [False, True]],
              [[True, False], [False, False], [True, False], [False, True]]
          ], ragged_rank=1),
          expected=[[0, 0, 0], [0, 1, 1],
                    [1, 0, 0], [1, 2, 0], [1, 3, 1]]),
          condition=ragged_factory_ops.constant_value([
              [[[], [True]]],
              [[[True, False, True], [False, True]],
               [[True], [], [False], [False, True, False]]]
          ]),
          expected=[[0, 0, 1, 0],
                    [1, 0, 0, 0], [1, 0, 0, 2], [1, 0, 1, 1],
                    [1, 1, 0, 0], [1, 1, 3, 1]]),
          condition=True, x='A', y='a', expected=b'A'),
          condition=False, x='A', y='a', expected=b'a'),
          condition=[True, False, True],
          x=['A', 'B', 'C'],
          y=['a', 'b', 'c'],
          expected=[b'A', b'b', b'C']),
          condition=[[True, False], [False, True]],
          x=[['A', 'B'], ['D', 'E']],
          y=[['a', 'b'], ['d', 'e']],
          expected=[[b'A', b'b'], [b'd', b'E']]),
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          x=ragged_factory_ops.constant_value([['A', 'B', 'C'], ['D', 'E']]),
          y=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'b', b'C'], [b'd', b'E']])),
          condition=ragged_factory_ops.constant_value([
              [[True, False], [False, True]],
              [[True, False], [False, False], [True, False], [False, True]]
          ], ragged_rank=1),
          x=ragged_factory_ops.constant_value([
              [['A', 'B'], ['C', 'D']],
              [['E', 'F'], ['G', 'H'], ['I', 'J'], ['K', 'L']]
          ], ragged_rank=1),
          y=ragged_factory_ops.constant_value([
              [['a', 'b'], ['c', 'd']],
              [['e', 'f'], ['g', 'h'], ['i', 'j'], ['k', 'l']]
          ], ragged_rank=1),
          expected=ragged_factory_ops.constant_value([
              [[b'A', b'b'], [b'c', b'D']],
              [[b'E', b'f'], [b'g', b'h'], [b'I', b'j'], [b'k', b'L']]
          ], ragged_rank=1)),
          condition=ragged_factory_ops.constant_value([
              [[[], [True]]],
              [[[True, False, True], [False, True]],
               [[True], [], [False], [False, True, False]]]
          ]),
          x=ragged_factory_ops.constant_value([
              [[[], ['A']]],
              [[['B', 'C', 'D'], ['E', 'F']],
               [['G'], [], ['H'], ['I', 'J', 'K']]]
          ]),
          y=ragged_factory_ops.constant_value([
              [[[], ['a']]],
              [[['b', 'c', 'd'], ['e', 'f']],
               [['g'], [], ['h'], ['i', 'j', 'k']]]
          ]),
          expected=ragged_factory_ops.constant_value([
              [[[], [b'A']]],
              [[[b'B', b'c', b'D'], [b'e', b'F']],
               [[b'G'], [], [b'h'], [b'i', b'J', b'k']]]
          ])),
          condition=[True, False, True],
          x=[['A', 'B'], ['C', 'D'], ['E', 'F']],
          y=[['a', 'b'], ['c', 'd'], ['e', 'f']],
          expected=[[b'A', b'B'], [b'c', b'd'], [b'E', b'F']]),
          condition=[True, False, True],
          x=[['A', 'B'], ['C', 'D'], ['E', 'F']],
          y=ragged_factory_ops.constant_value(
              [['a', 'b'], ['c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'B'], [b'c'], [b'E', b'F']])),
          condition=[True, False, True],
          x=ragged_factory_ops.constant_value(
              [['A', 'B', 'C'], ['D', 'E'], ['F', 'G']]),
          y=ragged_factory_ops.constant_value(
              [['a', 'b'], ['c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'B', b'C'], [b'c'], [b'F', b'G']])),
          condition=ragged_factory_ops.constant_value([True, False]),
          x=ragged_factory_ops.constant_value([
              [[[], ['A']]],
              [[['B', 'C', 'D'], ['E', 'F']],
               [['G'], [], ['H'], ['I', 'J', 'K']]]
          ]),
          y=ragged_factory_ops.constant_value([[[['a']]], [[['b']]]]),
          expected=ragged_factory_ops.constant_value(
              [[[[], [b'A']]], [[[b'b']]]])),
  def testRaggedWhere(self, condition, expected, x=None, y=None):
    result = ragged_where_op.where(condition, x, y)
    self.assertAllEqual(result, expected)
  @parameterized.parameters([
      dict(
          condition=[True, False],
          x=[1, 2],
          error=ValueError,
          message='x and y must be either both None or both non-None'),
      dict(
          condition=ragged_factory_ops.constant_value([[True, False, True],
                                                       [False, True]]),
          x=ragged_factory_ops.constant_value([['A', 'B', 'C'], ['D', 'E']]),
          y=[['a', 'b'], ['d', 'e']],
          error=ValueError,
          message='Input shapes do not match.'),
  ])
  def testRaggedWhereErrors(self, condition, error, message, x=None, y=None):
    with self.assertRaisesRegex(error, message):
      ragged_where_op.where(condition, x, y)
@test_util.run_all_in_graph_and_eager_modes
class RaggedWhereV2OpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.parameters([
          condition=[True, False, True, False, True],
          expected=[[0], [2], [4]]),
          condition=[[True, False], [False, True]],
          expected=[[0, 0], [1, 1]]),
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          expected=[[0, 0], [0, 2], [1, 1]]),
          condition=ragged_factory_ops.constant_value([
              [[True, False, True], [False, True]],
              [[True], [], [False], [False, True, False]]
          ]),
          expected=[[0, 0, 0], [0, 0, 2], [0, 1, 1],
                    [1, 0, 0], [1, 3, 1]]),
          condition=ragged_factory_ops.constant_value([
              [[True, False], [False, True]],
              [[True, False], [False, False], [True, False], [False, True]]
          ], ragged_rank=1),
          expected=[[0, 0, 0], [0, 1, 1],
                    [1, 0, 0], [1, 2, 0], [1, 3, 1]]),
          condition=ragged_factory_ops.constant_value([
              [[[], [True]]],
              [[[True, False, True], [False, True]],
               [[True], [], [False], [False, True, False]]]
          ]),
          expected=[[0, 0, 1, 0],
                    [1, 0, 0, 0], [1, 0, 0, 2], [1, 0, 1, 1],
                    [1, 1, 0, 0], [1, 1, 3, 1]]),
          condition=True, x='A', y='a', expected=b'A'),
          condition=False, x='A', y='a', expected=b'a'),
          condition=[True, False, True],
          x=['A', 'B', 'C'],
          y=['a', 'b', 'c'],
          expected=[b'A', b'b', b'C']),
          condition=[[True, False], [False, True]],
          x=[['A', 'B'], ['D', 'E']],
          y=[['a', 'b'], ['d', 'e']],
          expected=[[b'A', b'b'], [b'd', b'E']]),
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          x=ragged_factory_ops.constant_value([['A', 'B', 'C'], ['D', 'E']]),
          y=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'b', b'C'], [b'd', b'E']])),
          condition=ragged_factory_ops.constant_value([
              [[True, False], [False, True]],
              [[True, False], [False, False], [True, False], [False, True]]
          ], ragged_rank=1),
          x=ragged_factory_ops.constant_value([
              [['A', 'B'], ['C', 'D']],
              [['E', 'F'], ['G', 'H'], ['I', 'J'], ['K', 'L']]
          ], ragged_rank=1),
          y=ragged_factory_ops.constant_value([
              [['a', 'b'], ['c', 'd']],
              [['e', 'f'], ['g', 'h'], ['i', 'j'], ['k', 'l']]
          ], ragged_rank=1),
          expected=ragged_factory_ops.constant_value([
              [[b'A', b'b'], [b'c', b'D']],
              [[b'E', b'f'], [b'g', b'h'], [b'I', b'j'], [b'k', b'L']]
          ], ragged_rank=1)),
          condition=ragged_factory_ops.constant_value([
              [[[], [True]]],
              [[[True, False, True], [False, True]],
               [[True], [], [False], [False, True, False]]]
          ]),
          x=ragged_factory_ops.constant_value([
              [[[], ['A']]],
              [[['B', 'C', 'D'], ['E', 'F']],
               [['G'], [], ['H'], ['I', 'J', 'K']]]
          ]),
          y=ragged_factory_ops.constant_value([
              [[[], ['a']]],
              [[['b', 'c', 'd'], ['e', 'f']],
               [['g'], [], ['h'], ['i', 'j', 'k']]]
          ]),
          expected=ragged_factory_ops.constant_value([
              [[[], [b'A']]],
              [[[b'B', b'c', b'D'], [b'e', b'F']],
               [[b'G'], [], [b'h'], [b'i', b'J', b'k']]]
          ])),
          condition=[[True], [False], [True]],
          x=[['A', 'B'], ['C', 'D'], ['E', 'F']],
          y=[['a', 'b'], ['c', 'd'], ['e', 'f']],
          expected=[[b'A', b'B'], [b'c', b'd'], [b'E', b'F']]),
          condition=[[True], [False], [True]],
          x=ragged_factory_ops.constant_value(
              [['A', 'B', 'C'], ['D', 'E'], ['F', 'G']]),
          y=ragged_factory_ops.constant_value(
              [['a', 'b', 'c'], ['d', 'e'], ['f', 'g']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'B', b'C'], [b'd', b'e'], [b'F', b'G']])),
          condition=ragged_factory_ops.constant_value(
              [[True, False, True, True], [True, False]]),
          x=10,
          y=20,
          expected=ragged_factory_ops.constant_value(
              [[10, 20, 10, 10], [10, 20]])),
          condition=[[True, False], [True, False], [False, True]],
          x=[[10], [20], [30]],
          y=[[40, 50]],
          expected=[[10, 50], [20, 50], [40, 30]]),
          condition=ragged_factory_ops.constant_value(
              [[[True, False], [False, True]], [[True, True]]],
              ragged_rank=1),
          x=ragged_factory_ops.constant_value([[[10], [20]], [[30]]],
                                              ragged_rank=1),
          y=np.array([[[40, 50]]]),
          expected=[[[10, 50], [40, 20]], [[30, 30]]]),
  def testRaggedWhere(self, condition, expected, x=None, y=None):
    result = ragged_where_op.where_v2(condition, x, y)
    self.assertAllEqual(result, expected)
  @parameterized.parameters([
      dict(
          condition=[True, False],
          x=[1, 2],
          error=ValueError,
          message='x and y must be either both None or both non-None'),
      dict(
          condition=ragged_factory_ops.constant_value([[True, False, True],
                                                       [False, True]]),
          x=ragged_factory_ops.constant_value([['A', 'B', 'C'], ['D', 'E']]),
          y=[['a', 'b'], ['d', 'e']],
          error=errors.InvalidArgumentError,
          message=r'must be broadcastable|Unable to broadcast'),
  ])
  def testRaggedWhereErrors(self, condition, error, message, x=None, y=None):
    with self.assertRaisesRegex(error, message):
      self.evaluate(ragged_where_op.where_v2(condition, x, y))
if __name__ == '__main__':
  googletest.main()
