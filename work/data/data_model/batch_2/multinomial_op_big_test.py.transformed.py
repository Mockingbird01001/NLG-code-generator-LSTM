
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
class MultinomialTest(test.TestCase):
  def testLargeDynamicRange(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[-30, 0]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=15)
      for _ in range(100):
        x = self.evaluate(samples)
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertEqual(counts_by_indices[1], 100000000)
  def testLargeDynamicRange2(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[0, -30]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=15)
      for _ in range(100):
        x = self.evaluate(samples)
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertEqual(counts_by_indices[0], 100000000)
  @test_util.run_deprecated_v1
  def testLargeDynamicRange3(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[0, -17]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=22)
      for _ in range(100):
        x = self.evaluate(samples)
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertGreater(counts_by_indices[1], 0)
if __name__ == "__main__":
  test.main()
