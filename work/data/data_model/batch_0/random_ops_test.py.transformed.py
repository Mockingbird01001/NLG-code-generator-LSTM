
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
class RandomOpTestCommon(test.TestCase):
  def _testSingleSessionNotConstant(self,
                                    rng_func,
                                    num,
                                    dtype,
                                    min_or_mean,
                                    max_or_stddev,
                                    use_gpu,
                                    op_seed=None,
                                    graph_seed=None):
    with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
      if graph_seed is not None:
        random_seed.set_random_seed(graph_seed)
      x = rng_func([num], min_or_mean, max_or_stddev, dtype=dtype, seed=op_seed)
      y = self.evaluate(x)
      z = self.evaluate(x)
      w = self.evaluate(x)
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or (not np.array_equal(y, w)))
class RandomNormalTest(RandomOpTestCommon):
  def _Sampler(self, num, mu, sigma, dtype, use_gpu, seed=None):
    def func():
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_normal(
            [num], mean=mu, stddev=sigma, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in range(10):
          ret[i, :] = self.evaluate(rng)
      return ret
    return func
  def testDistinct(self):
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True)
      x = sampler()
      y = sampler()
      count = (x == y).sum()
      if count >= 10:
        print("x = ", x)
        print("y = ", y)
        print("count = ", count)
      self.assertTrue(count < 10)
  @test_util.run_deprecated_v1
  def testCPUGPUMatch(self):
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(
            1000000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      if dt == dtypes.float16:
        self.assertAllClose(results[False], results[True], rtol=1e-3, atol=1e-3)
      else:
        self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)
  @test_util.run_deprecated_v1
  def testSeed(self):
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sx = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True, seed=345)
      sy = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True, seed=345)
      self.assertAllEqual(sx(), sy())
  @test_util.run_deprecated_v1
  def testNoCSE(self):
    for use_gpu in [False, True]:
      with self.session(use_gpu=use_gpu):
        shape = [2, 3, 4]
        rnd1 = random_ops.random_normal(shape, 0.0, 1.0, dtypes.float32)
        rnd2 = random_ops.random_normal(shape, 0.0, 1.0, dtypes.float32)
        diff = rnd2 - rnd1
        self.assertTrue(np.linalg.norm(diff.eval()) > 0.1)
  @test_util.run_deprecated_v1
  def testSingleSessionNotConstant(self):
    for use_gpu in [False, True]:
      for dt in dtypes.float16, dtypes.float32, dtypes.float64:
        self._testSingleSessionNotConstant(
            random_ops.random_normal, 100, dt, 0.0, 1.0, use_gpu=use_gpu)
  @test_util.run_deprecated_v1
  def testSingleSessionOpSeedNotConstant(self):
    for use_gpu in [False, True]:
      for dt in dtypes.float16, dtypes.float32, dtypes.float64:
        self._testSingleSessionNotConstant(
            random_ops.random_normal,
            100,
            dt,
            0.0,
            1.0,
            use_gpu=use_gpu,
            op_seed=1345)
  @test_util.run_deprecated_v1
  def testSingleSessionGraphSeedNotConstant(self):
    for use_gpu in [False, True]:
      for dt in dtypes.float16, dtypes.float32, dtypes.float64:
        self._testSingleSessionNotConstant(
            random_ops.random_normal,
            100,
            dt,
            0.0,
            1.0,
            use_gpu=use_gpu,
            graph_seed=965)
@test_util.with_eager_op_as_function
class TruncatedNormalTest(test.TestCase):
  def _Sampler(self, num, mu, sigma, dtype, use_gpu, seed=None):
    def func():
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.truncated_normal(
            [num], mean=mu, stddev=sigma, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in range(10):
          ret[i, :] = self.evaluate(rng)
      return ret
    return func
  def testDistinct(self):
    if not test.is_gpu_available():
      for dt in dtypes.float16, dtypes.float32, dtypes.float64:
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=False)
        x = sampler()
        y = sampler()
        count = (x == y).sum()
        if count >= 10:
          print("x = ", x)
          print("y = ", y)
          print("count = ", count)
        self.assertTrue(count < 10)
  @test_util.run_deprecated_v1
  def testCPUGPUMatch(self):
    if not test.is_gpu_available():
      return
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(
            1000000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      if dt == dtypes.float16:
        self.assertAllClose(results[False], results[True], rtol=1e-3, atol=1e-3)
      else:
        self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)
  @test_util.run_deprecated_v1
  def testSeed(self):
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sx = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True, seed=345)
      sy = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True, seed=345)
      self.assertAllEqual(sx(), sy())
  def testStdDev(self):
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      stddev = 3.0
      sampler = self._Sampler(100000, 0.0, stddev, dt, use_gpu=True)
      x = sampler()
      print("std(x)", np.std(x), abs(np.std(x) / stddev - 0.85))
      self.assertLess(abs(np.std(x) / stddev - 0.85), 0.04)
  def testSuccessAfterError(self):
    config.enable_op_determinism()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "When determinism is enabled, random ops must have a seed specified"):
      self.evaluate(gen_random_ops.truncated_normal((1,), dtypes.float32))
    config.disable_op_determinism()
    self.testStdDev()
  @test_util.run_deprecated_v1
  def testLargeShape(self):
    with self.session():
      v = variables.Variable(
          array_ops.zeros(dtype=dtypes.float32, shape=[2**33, 1]))
      n = random_ops.truncated_normal(v.shape)
      self.assertEqual([8589934592, 1], n.shape.as_list())
  @test_util.run_deprecated_v1
  def testNoCSE(self):
    with self.session():
      shape = [2, 3, 4]
      rnd1 = random_ops.truncated_normal(shape, 0.0, 1.0, dtypes.float32)
      rnd2 = random_ops.truncated_normal(shape, 0.0, 1.0, dtypes.float32)
      diff = rnd2 - rnd1
      self.assertTrue(np.linalg.norm(diff.eval()) > 0.1)
  def testEagerSeed(self):
    with context.eager_mode():
      random_ops.random_normal([])
      context.set_global_seed(42)
      rnd1 = random_ops.random_normal([])
      context.set_global_seed(42)
      rnd2 = random_ops.random_normal([])
      self.assertAllEqual(rnd1, rnd2)
@test_util.with_eager_op_as_function
@test_util.for_all_test_methods(test_util.disable_xla,
                                "This never passed on XLA")
class RandomUniformTest(RandomOpTestCommon):
  def _Sampler(self, num, minv, maxv, dtype, use_gpu, seed=None):
    def func():
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_uniform(
            [num], minval=minv, maxval=maxv, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in range(10):
          ret[i, :] = self.evaluate(rng)
      return ret
    return func
  def testRange(self):
    for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
               dtypes.int64):
      sampler = self._Sampler(1000, minv=-2, maxv=8, dtype=dt, use_gpu=True)
      x = sampler()
      self.assertTrue(-2 <= np.min(x))
      self.assertTrue(np.max(x) < 8)
  def testDistinct(self):
    for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
               dtypes.int64):
      maxv = 1.0 if dt.is_floating else 1 << 30
      sampler = self._Sampler(1000, minv=0, maxv=maxv, dtype=dt, use_gpu=True)
      x = sampler()
      y = sampler()
      count = (x == y).sum()
      count_limit = 50 if dt == dtypes.float16 else 10
      if count >= count_limit:
        print("x = ", x)
        print("y = ", y)
        print("count = ", count)
      self.assertTrue(count < count_limit)
  @test_util.run_deprecated_v1
  def testUniformIntsWithInvalidShape(self):
    for dtype in dtypes.int32, dtypes.int64:
      with self.assertRaisesRegex(
          ValueError, "minval must be a scalar; got a tensor of shape"):
        random_ops.random_uniform(
            [1000], minval=[1, 2], maxval=3, dtype=dtype)
      with self.assertRaisesRegex(
          ValueError, "maxval must be a scalar; got a tensor of shape"):
        random_ops.random_uniform(
            [1000], minval=1, maxval=[2, 3], dtype=dtype)
  @test_util.run_deprecated_v1
  def testUniformInts(self):
    minv = -2
    maxv = 15
    n = 100000
    p = 1 / (maxv - minv)
    mean = p * n
    std = np.sqrt(n * p * (1 - p))
    for dt in dtypes.int32, dtypes.int64:
      sampler = self._Sampler(
          n // 10, minv=minv, maxv=maxv, dtype=dt, use_gpu=True, seed=17)
      x = sampler().ravel()
      self.assertEqual(x.shape, (n,))
      counts, _ = np.histogram(x, bins=maxv - minv)
      self.assertEqual(counts.shape, (maxv - minv,))
      self.assertEqual(counts.sum(), n)
      error = np.abs(counts - mean)
      self.assertLess(error.max(), 5 * std)
  def testUniformIntsDegenerate(self):
    for dt in dtypes.int32, dtypes.int64:
      def sample(n):
        return self._Sampler(n, minv=0, maxv=0, dtype=dt, use_gpu=True)()
      self.assertEqual(sample(0).shape, (10, 0))
      with self.assertRaisesOpError('Need minval < maxval, got 0 >= 0'):
        sample(1)
  @test_util.run_deprecated_v1
  def testCPUGPUMatch(self):
    for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
               dtypes.int64):
      maxv = 1.0 if dt.is_floating else 17
      results = {}
      for use_gpu in False, True:
        sampler = self._Sampler(
            1000000, minv=0, maxv=maxv, dtype=dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      self.assertAllEqual(results[False], results[True])
  @test_util.run_deprecated_v1
  def testSeed(self):
    for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
               dtypes.int64):
      for seed in [345, 2**100, -2**100]:
        sx = self._Sampler(1000, 0, 17, dtype=dt, use_gpu=True, seed=seed)
        sy = self._Sampler(1000, 0, 17, dtype=dt, use_gpu=True, seed=seed)
        self.assertAllEqual(sx(), sy())
  @test_util.run_deprecated_v1
  def testNoCSE(self):
    shape = [2, 3, 4]
    for dtype in dtypes.float16, dtypes.float32, dtypes.int32:
      with self.session():
        rnd1 = random_ops.random_uniform(shape, 0, 17, dtype=dtype)
        rnd2 = random_ops.random_uniform(shape, 0, 17, dtype=dtype)
        diff = (rnd2 - rnd1).eval()
        self.assertTrue(np.linalg.norm(diff) > 0.1)
  @test_util.run_deprecated_v1
  def testSingleSessionNotConstant(self):
    for use_gpu in [False, True]:
      for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
                 dtypes.int64):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform, 100, dt, 0, 17, use_gpu=use_gpu)
  @test_util.run_deprecated_v1
  def testSingleSessionOpSeedNotConstant(self):
    for use_gpu in [False, True]:
      for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
                 dtypes.int64):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform,
            100,
            dt,
            10,
            20,
            use_gpu=use_gpu,
            op_seed=1345)
  @test_util.run_deprecated_v1
  def testSingleSessionGraphSeedNotConstant(self):
    for use_gpu in [False, True]:
      for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
                 dtypes.int64):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform,
            100,
            dt,
            20,
            200,
            use_gpu=use_gpu,
            graph_seed=965)
class RandomShapeTest(test.TestCase):
  @test_util.run_deprecated_v1
  def testTruncatedNormal(self):
    rnd1 = random_ops.truncated_normal([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    rnd2 = random_ops.truncated_normal(
        array_ops.placeholder(dtypes.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    rnd3 = random_ops.truncated_normal(array_ops.placeholder(dtypes.int32))
    self.assertIs(None, rnd3.get_shape().ndims)
  @test_util.run_deprecated_v1
  def testRandomNormal(self):
    rnd1 = random_ops.random_normal([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    rnd2 = random_ops.random_normal(
        array_ops.placeholder(dtypes.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    rnd3 = random_ops.random_normal(array_ops.placeholder(dtypes.int32))
    self.assertIs(None, rnd3.get_shape().ndims)
  @test_util.run_deprecated_v1
  def testRandomUniform(self):
    rnd1 = random_ops.random_uniform([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    rnd2 = random_ops.random_uniform(
        array_ops.placeholder(dtypes.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    rnd3 = random_ops.random_uniform(array_ops.placeholder(dtypes.int32))
    self.assertIs(None, rnd3.get_shape().ndims)
class DeterministicOpsTest(test.TestCase):
  def setUp(self):
    super().setUp()
    random_seed.set_random_seed(None)
    config.enable_op_determinism()
  def tearDown(self):
    super().tearDown()
    config.disable_op_determinism()
  def testDeterministicOpsErrors(self):
    with self.assertRaisesRegex(
        RuntimeError,
        "Random ops require a seed to be set when determinism is enabled."):
      random_ops.random_normal((1,))
    with self.assertRaisesRegex(
        RuntimeError,
        "Random ops require a seed to be set when determinism is enabled."):
      random_ops.truncated_normal((1,))
    with self.assertRaisesRegex(
        RuntimeError,
        "Random ops require a seed to be set when determinism is enabled."):
      random_ops.random_uniform((1,))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "When determinism is enabled, random ops must have a seed specified"):
      self.evaluate(gen_random_ops.random_standard_normal((1,), dtypes.float32))
  def testErrorNotThrownWithSeed(self):
    random_ops.random_normal((1,), seed=0)
    random_seed.set_random_seed(0)
    random_ops.random_normal((1,))
    self.evaluate(gen_random_ops.random_standard_normal((1,), dtypes.float32,
                                                        seed=1))
    self.evaluate(gen_random_ops.random_standard_normal((1,), dtypes.float32,
                                                        seed2=1))
if __name__ == "__main__":
  test.main()
