
import numpy as np
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
class DilationTest(test.TestCase):
  def _VerifyValues(self, image, kernel, strides, rates, padding, out, use_gpu):
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]
    with self.cached_session(use_gpu=use_gpu):
      out_tensor = nn_ops.dilation2d(
          constant_op.constant(image),
          constant_op.constant(kernel),
          strides=strides,
          rates=rates,
          padding=padding,
          name="dilation2d")
      self.assertAllClose(out, self.evaluate(out_tensor))
  def _testDilationValidPadding(self, use_gpu):
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    out = [[[[.5]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)
  def _testDilationSamePadding(self, use_gpu):
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    out = [[[[.5], [.6]], [[.7], [.8]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)
  def _testDilationSamePaddingDepth(self, use_gpu):
    image = [[[[.1, .2, .0], [.2, .3, .1]], [[.3, .4, .2], [.4, .5, .3]]]]
    kernel = [[[.4, .5, .3], [.3, .4, .2]], [[.1, .2, .0], [.0, .1, -.1]]]
    out = [[[[.5, .7, .3], [.6, .8, .4]], [[.7, .9, .5], [.8, 1., .6]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)
  def _testDilationSamePaddingBatch(self, use_gpu):
    image = [[[[.1], [.2]], [[.3], [.4]]], [[[.2], [.3]], [[.4], [.5]]]]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    out = [[[[.5], [.6]], [[.7], [.8]]], [[[.6], [.7]], [[.8], [.9]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)
  def _testDilationValidPaddingNonSquareWindow(self, use_gpu):
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    kernel = [[[.4], [.3]]]
    out = [[[[.5]], [[.7]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)
  def _testDilationSamePaddingRate(self, use_gpu):
    image = [[[[.1], [.2], [.3]], [[.4], [.5], [.6]], [[.7], [.8], [.9]]]]
    kernel = [[[.4], [.3]], [[.1], [.2]]]
    out = [[[[.7], [.8], [.6]], [[1.0], [1.1], [.9]], [[.8], [.9], [.9]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[2, 2],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)
  def _testDilationValidPaddingUnevenStride(self, use_gpu):
    image = [[[[.1], [.2], [.3], [.4]], [[.5], [.6], [.7], [.8]],
              [[.9], [1.0], [1.1], [1.2]]]]
    kernel = [[[.4], [.3]], [[.1], [.2]]]
    out = [[[[.8], [1.0]], [[1.2], [1.4]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 2],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)
  def testDilation(self):
    for use_gpu in True, False:
      self._testDilationValidPadding(use_gpu)
      self._testDilationSamePadding(use_gpu)
      self._testDilationSamePaddingDepth(use_gpu)
      self._testDilationSamePaddingBatch(use_gpu)
      self._testDilationValidPaddingNonSquareWindow(use_gpu)
      self._testDilationSamePaddingRate(use_gpu)
      self._testDilationValidPaddingUnevenStride(use_gpu)
  def _ConstructAndTestGradient(self, image_shape, kernel_shape, strides, rates,
                                padding, use_gpu):
    assert image_shape[3] == kernel_shape[2]
    image = np.random.random_sample(image_shape).astype(np.float32)
    kernel = np.random.random_sample(kernel_shape).astype(np.float32)
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]
    image_tensor = constant_op.constant(image, shape=image_shape, name="input")
    kernel_tensor = constant_op.constant(
        kernel, shape=kernel_shape, name="filter")
    def compute_dilation2d(image_tensor, kernel_tensor):
      return nn_ops.dilation2d(
          image_tensor,
          kernel_tensor,
          strides=strides,
          rates=rates,
          padding=padding,
          name="dilation2d")
    with test_util.device(use_gpu=use_gpu):
      with self.cached_session():
        err1 = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: compute_dilation2d(x, kernel_tensor), [image_tensor]))
        err2 = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: compute_dilation2d(image_tensor, x), [kernel_tensor]))
        err = max(err1, err2)
    print("Dilation gradient error = %f" % err)
    self.assertLess(err, 1e-4)
  def _testDilationGradValidPadding_1x1x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[1, 1, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        use_gpu=use_gpu)
  def _testDilationGradDeterminismError(self, use_gpu):
    if use_gpu and test.is_gpu_available(cuda_only=True):
      try:
        config.enable_op_determinism()
        with self.assertRaisesRegexp(
            errors_impl.UnimplementedError, "Determinism is not yet supported "
            "for Dilation2DBackpropInput."):
          self._ConstructAndTestGradient(
              image_shape=[1, 3, 3, 1],
              kernel_shape=[1, 1, 1],
              strides=[1, 1],
              rates=[1, 1],
              padding="VALID",
              use_gpu=use_gpu)
      finally:
        config.disable_op_determinism()
    else:
      try:
        config.enable_op_determinism()
        self._ConstructAndTestGradient(
            image_shape=[1, 3, 3, 1],
            kernel_shape=[1, 1, 1],
            strides=[1, 1],
            rates=[1, 1],
            padding="VALID",
            use_gpu=use_gpu)
      finally:
        config.disable_op_determinism()
  def _testDilationGradSamePadding_1x1x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[1, 1, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def _testDilationGradSamePadding_1x1x2(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 2],
        kernel_shape=[1, 1, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def _testDilationGradValidPadding_2x2x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        use_gpu=use_gpu)
  def _testDilationGradSamePadding_2x2x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def _testDilationGradSamePaddingBatch_2x2x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[4, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def _testDilationGradSamePadding_2x2x4(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 4],
        kernel_shape=[2, 2, 4],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def testDilationGrad(self):
    for use_gpu in True, False:
      self._testDilationGradDeterminismError(use_gpu)
      self._testDilationGradValidPadding_1x1x1(use_gpu)
      self._testDilationGradSamePadding_1x1x1(use_gpu)
      self._testDilationGradSamePadding_1x1x2(use_gpu)
      self._testDilationGradValidPadding_2x2x1(use_gpu)
      self._testDilationGradSamePadding_2x2x1(use_gpu)
      self._testDilationGradSamePaddingBatch_2x2x1(use_gpu)
      self._testDilationGradSamePadding_2x2x4(use_gpu)
class ErosionTest(test.TestCase):
  def _VerifyValues(self, image, kernel, strides, rates, padding, out, use_gpu):
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]
    with self.cached_session(use_gpu=use_gpu):
      out_tensor = nn_ops.erosion2d(
          constant_op.constant(image),
          constant_op.constant(kernel),
          strides=strides,
          rates=rates,
          padding=padding,
          name="erosion2d")
      self.assertAllClose(out, self.evaluate(out_tensor))
  def _testErosionValidPadding(self, use_gpu):
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    out = [[[[.0]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)
  def _testErosionSamePadding(self, use_gpu):
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    out = [[[[.0], [.1]], [[.3], [.4]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)
  def _testErosionSamePaddingDepth(self, use_gpu):
    image = [[[[.1, .2, .0], [.2, .3, .1]], [[.3, .4, .2], [.4, .5, .3]]]]
    kernel = [[[.4, .5, .3], [.3, .4, .2]], [[.1, .2, .0], [.0, .1, -.1]]]
    out = [[[[.0, .0, .0], [.1, .1, .1]], [[.3, .3, .3], [.4, .4, .4]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)
  def _testErosionSamePaddingBatch(self, use_gpu):
    image = [[[[.1], [.2]], [[.3], [.4]]], [[[.2], [.3]], [[.4], [.5]]]]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    out = [[[[.0], [.1]], [[.3], [.4]]], [[[.1], [.2]], [[.4], [.5]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)
  def _testErosionValidPaddingNonSquareWindow(self, use_gpu):
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    kernel = [[[.4], [.3]]]
    out = [[[[-.2]], [[.0]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)
  def _testErosionSamePaddingRate(self, use_gpu):
    image = [[[[.1], [.2], [.3]], [[.4], [.5], [.6]], [[.7], [.8], [.9]]]]
    kernel = [[[.4], [.3]], [[.1], [.2]]]
    out = [[[[.1], [.1], [.2]], [[0.1], [-.1], [.0]], [[.4], [.2], [.3]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[2, 2],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)
  def _testErosionValidPaddingUnevenStride(self, use_gpu):
    image = [[[[.1], [.2], [.3], [.4]], [[.5], [.6], [.7], [.8]],
              [[.9], [1.0], [1.1], [1.2]]]]
    kernel = [[[.4], [.3]], [[.1], [.2]]]
    out = [[[[-.1], [.1]], [[.3], [.5]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 2],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)
  def testErosion(self):
    for use_gpu in True, False:
      self._testErosionValidPadding(use_gpu)
      self._testErosionSamePadding(use_gpu)
      self._testErosionSamePaddingDepth(use_gpu)
      self._testErosionSamePaddingBatch(use_gpu)
      self._testErosionValidPaddingNonSquareWindow(use_gpu)
      self._testErosionSamePaddingRate(use_gpu)
      self._testErosionValidPaddingUnevenStride(use_gpu)
  def _ConstructAndTestGradient(self, image_shape, kernel_shape, strides, rates,
                                padding, use_gpu):
    assert image_shape[3] == kernel_shape[2]
    image = np.random.random_sample(image_shape).astype(np.float32)
    kernel = np.random.random_sample(kernel_shape).astype(np.float32)
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]
    image_tensor = constant_op.constant(image, shape=image_shape, name="input")
    kernel_tensor = constant_op.constant(
        kernel, shape=kernel_shape, name="filter")
    def compute_erosion2d(image_tensor, kernel_tensor):
      return nn_ops.erosion2d(
          image_tensor,
          kernel_tensor,
          strides=strides,
          rates=rates,
          padding=padding,
          name="erosion2d")
    with test_util.device(use_gpu=use_gpu):
      with self.cached_session():
        err1 = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: compute_erosion2d(x, kernel_tensor), [image_tensor]))
        err2 = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: compute_erosion2d(image_tensor, x), [kernel_tensor]))
        err = max(err1, err2)
    print("Erosion gradient error = %f" % err)
    self.assertLess(err, 1e-4)
  def _testErosionGradValidPadding_1x1x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[1, 1, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        use_gpu=use_gpu)
  def _testErosionGradSamePadding_1x1x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[1, 1, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def _testErosionGradSamePadding_1x1x2(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 2],
        kernel_shape=[1, 1, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def _testErosionGradValidPadding_2x2x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        use_gpu=use_gpu)
  def _testErosionGradSamePadding_2x2x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def _testErosionGradSamePaddingBatch_2x2x1(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[4, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def _testErosionGradSamePadding_2x2x4(self, use_gpu):
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 4],
        kernel_shape=[2, 2, 4],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)
  def testErosionGrad(self):
    for use_gpu in True, False:
      self._testErosionGradValidPadding_1x1x1(use_gpu)
      self._testErosionGradSamePadding_1x1x1(use_gpu)
      self._testErosionGradSamePadding_1x1x2(use_gpu)
      self._testErosionGradValidPadding_2x2x1(use_gpu)
      self._testErosionGradSamePadding_2x2x1(use_gpu)
      self._testErosionGradSamePaddingBatch_2x2x1(use_gpu)
      self._testErosionGradSamePadding_2x2x4(use_gpu)
if __name__ == "__main__":
  test.main()
