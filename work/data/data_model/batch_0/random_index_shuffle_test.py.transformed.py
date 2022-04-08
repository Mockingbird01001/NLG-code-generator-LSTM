
import itertools
from absl.testing import parameterized
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.platform import test
_SEEDS = ((74, 117), (42, 5))
_MAX_VALUES = (129, 2_389)
_DTYPES = (dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64)
class StatelessOpsTest(test.TestCase, parameterized.TestCase):
  @parameterized.parameters(
      itertools.product(_SEEDS, _DTYPES, _MAX_VALUES, _DTYPES))
  def testUnbatched(self, seed, seed_dtype, max_index, index_dtype):
    if max_index > 200:
      self.skipTest('Too slow in graph mode.')
    seen = (max_index + 1) * [False]
    seed = math_ops.cast(seed, seed_dtype)
    for index in range(max_index + 1):
      new_index = stateless.index_shuffle(
          math_ops.cast(index, index_dtype),
          seed,
          max_index=math_ops.cast(max_index, index_dtype))
      self.assertEqual(new_index.dtype, index_dtype)
      new_index = self.evaluate(new_index)
      self.assertGreaterEqual(new_index, 0)
      self.assertLessEqual(new_index, max_index)
      self.assertFalse(seen[new_index])
      seen[new_index] = True
  @parameterized.parameters(
      itertools.product(_SEEDS, _DTYPES, _MAX_VALUES, _DTYPES))
  def testBatchedBroadcastSeedAndMaxval(self, seed, seed_dtype, max_index,
                                        index_dtype):
    seed = math_ops.cast(seed, seed_dtype)
    index = math_ops.cast(range(max_index + 1), index_dtype)
    new_index = stateless.index_shuffle(index, seed, max_index=max_index)
    self.assertEqual(new_index.dtype, index_dtype)
    new_index = self.evaluate(new_index)
    self.assertAllGreaterEqual(new_index, 0)
    self.assertAllLessEqual(new_index, max_index)
    self.assertLen(new_index, max_index + 1)
    self.assertLen(set(new_index), max_index + 1)
if __name__ == '__main__':
  test.main()
