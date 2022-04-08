
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
class ExtractImagePatches(test.TestCase):
  def _VerifyValues(self, image, ksizes, strides, rates, padding, patches):
    ksizes = [1] + ksizes + [1]
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]
    out_tensor = array_ops.extract_image_patches(
        constant_op.constant(image),
        ksizes=ksizes,
        strides=strides,
        rates=rates,
        padding=padding,
        name="im2col")
    self.assertAllClose(patches, self.evaluate(out_tensor))
  def testKsize1x1Stride1x1Rate1x1(self):
    image = np.reshape(range(120), [2, 3, 4, 5])
    patches = np.reshape(range(120), [2, 3, 4, 5])
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1],
          strides=[1, 1],
          rates=[1, 1],
          padding=padding,
          patches=patches)
  def testKsize1x1Stride2x3Rate1x1(self):
    image = np.reshape(range(120), [2, 4, 5, 3])
    patches = image[:, ::2, ::3, :]
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1],
          strides=[2, 3],
          rates=[1, 1],
          padding=padding,
          patches=patches)
  def testKsize2x2Stride1x1Rate1x1Valid(self):
    image = [[[[1], [2]], [[3], [4]]]]
    patches = [[[[1, 2, 3, 4]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        patches=patches)
  def testKsize2x2Stride1x1Rate1x1Same(self):
    image = [[[[1], [2]], [[3], [4]]]]
    patches = [[[[1, 2, 3, 4], [2, 0, 4, 0]], [[3, 4, 0, 0], [4, 0, 0, 0]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        patches=patches)
  def testKsize2x2Stride1x1Rate2x2Valid(self):
    image = np.arange(16).reshape(1, 4, 4, 1).astype(np.float32)
    patches = [[[[0, 2, 8, 10], [1, 3, 9, 11]],
                [[4, 6, 12, 14], [5, 7, 13, 15]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[2, 2],
        padding="VALID",
        patches=patches)
  def testComplexDataTypes(self):
    for dtype in [np.complex64, np.complex128]:
      image = (
          np.reshape(range(120), [2, 3, 4, 5]).astype(dtype) +
          np.reshape(range(120, 240), [2, 3, 4, 5]).astype(dtype) * 1j)
      patches = (
          np.reshape(range(120), [2, 3, 4, 5]).astype(dtype) +
          np.reshape(range(120, 240), [2, 3, 4, 5]).astype(dtype) * 1j)
      for padding in ["VALID", "SAME"]:
        self._VerifyValues(
            image,
            ksizes=[1, 1],
            strides=[1, 1],
            rates=[1, 1],
            padding=padding,
            patches=patches)
if __name__ == "__main__":
  test.main()
