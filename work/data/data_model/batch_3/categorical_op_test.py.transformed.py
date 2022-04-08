
import collections
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
class CategoricalTest(xla_test.XLATestCase):
  def output_dtypes(self):
    return set(self.int_types).intersection([np.int32, np.int64])
  def _chi2(self, expected, actual):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = actual - expected
    chi2 = np.sum(diff * diff / expected)
    return chi2
  def _do_sampling(self, logits, num_samples):
    with self.session(), self.test_scope():
      random_seed.set_random_seed(1618)
      op = random_ops.multinomial(logits, num_samples,
                                  output_dtype=dtypes.int32)
      d = self.evaluate(op)
    batch_size, num_classes = logits.shape
    freqs_mat = []
    for i in range(batch_size):
      cnts = dict(collections.Counter(d[i, :]))
      self.assertLess(max(cnts.keys()), num_classes)
      self.assertGreaterEqual(min(cnts.keys()), 0)
      freqs = [(cnts[k] * 1. / num_samples if k in cnts else 0)
               for k in range(num_classes)]
      freqs_mat.append(freqs)
    return freqs_mat
  def _testRngIsNotConstant(self, rng, dtype, output_dtype):
    with self.session():
      with self.test_scope():
        x = rng(dtype, output_dtype)
      y = self.evaluate(x)
      z = self.evaluate(x)
      w = self.evaluate(x)
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or
                      (not np.array_equal(y, w)))
  def testCategoricalIsNotConstant(self):
    def rng(dtype, output_dtype):
      return random_ops.multinomial(np.array([[1., 1., 1.]], dtype=dtype), 10,
                                    output_dtype=output_dtype)
    dtype = np.float32
    for output_dtype in self.output_dtypes():
      self._testRngIsNotConstant(rng, dtype, output_dtype)
  @test.disable_with_predicate(
      pred=test.is_built_with_rocm, skip_message="Test fails on ROCm.")
  def testCategoricalIsInRange(self):
    for dtype in self.float_types:
      for output_dtype in self.output_dtypes():
        with self.session():
          with self.test_scope():
            x = random_ops.multinomial(
                array_ops.ones(shape=[1, 20], dtype=dtype), 1000,
                output_dtype=output_dtype)
          y = self.evaluate(x)
          self.assertTrue((y >= 0).sum() == 1000)
          self.assertTrue((y < 20).sum() == 1000)
  def testSamplingCorrectness(self):
    num_samples = 40000
    rand_probs = np.random.dirichlet([1., 1., 2., 3.])
    for probs in [[.5, .5], [.85, .05, .1], rand_probs, rand_probs2]:
      probs = np.asarray(probs)
      if len(probs.shape) == 1:
      logits = np.log(probs).astype(np.float32)
      freqs = self._do_sampling(logits, num_samples)
      chi2 = self._chi2(probs, freqs)
      self.assertLess(chi2, 1e-3)
  def testStatelessMultinomialIsInRange(self):
    for dtype in self.float_types.intersection(
        [dtypes.float32, dtypes.bfloat16]):
      for output_dtype in self.output_dtypes():
        with self.session() as sess:
          with self.test_scope():
            seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
            x = stateless_random_ops.stateless_multinomial(
                array_ops.ones(shape=[1, 20], dtype=dtype),
                1000,
                seed_t,
                output_dtype=output_dtype)
          y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
          self.assertTrue((y >= 0).sum() == 1000)
          self.assertTrue((y < 20).sum() == 1000)
  def testDeterminismMultinomial(self):
    num_samples = 10
    with self.session(), self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      seeds = [(x, y) for x in range(5) for y in range(5)] * 3
      for logits in ([[0.1, 0.25, 0.5, 0.15]], [[0.5, 0.5], [0.8, 0.2],
                                                [0.25, 0.75]]):
        pure = stateless_random_ops.stateless_multinomial(
            logits, num_samples, seed=seed_t)
        values = [(seed, pure.eval(feed_dict={seed_t: seed})) for seed in seeds]
        for s0, v0 in values:
          for s1, v1 in values:
            self.assertEqual(s0 == s1, np.all(v0 == v1))
  def testEmpty(self):
    with self.session():
      with self.test_scope():
        x = random_ops.multinomial(
            array_ops.zeros([42, 40]), 0, output_dtype=dtypes.int32)
        y = self.evaluate(x)
        self.assertEqual(y.shape, (42, 0))
  def testEmptyStateless(self):
    with self.session() as sess:
      with self.test_scope():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless_random_ops.stateless_multinomial(
            array_ops.zeros([42, 40]),
            0,
            seed=seed_t,
            output_dtype=dtypes.int32)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
        self.assertEqual(y.shape, (42, 0))
if __name__ == '__main__':
  googletest.main()
