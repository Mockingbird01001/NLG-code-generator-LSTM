
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
class DrawBoundingBoxOpTest(test.TestCase):
  def _fillBorder(self, image, color):
    height, width, depth = image.shape
    if depth != color.shape[0]:
      raise ValueError("Image (%d) and color (%d) depths must match." %
                       (depth, color.shape[0]))
    image[0:height, 0, 0:depth] = color
    image[0:height, width - 1, 0:depth] = color
    image[0, 0:width, 0:depth] = color
    image[height - 1, 0:width, 0:depth] = color
    return image
  def _testDrawBoundingBoxColorCycling(self, img, colors=None):
    color_table = colors
    if colors is None:
      color_table = np.asarray([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1],
                                [0, 1, 0, 1], [0.5, 0, 0.5,
                                               1], [0.5, 0.5, 0, 1],
                                [0.5, 0, 0, 1], [0, 0, 0.5, 1], [0, 1, 1, 1],
                                [1, 0, 1, 1]])
    assert len(img.shape) == 3
    depth = img.shape[2]
    assert depth <= color_table.shape[1]
    assert depth == 1 or depth == 3 or depth == 4
    if depth == 1:
      color_table[:, 0] = 1
    num_colors = color_table.shape[0]
    for num_boxes in range(1, num_colors + 2):
      image = np.copy(img)
      color = color_table[(num_boxes - 1) % num_colors, 0:depth]
      test_drawn_image = self._fillBorder(image, color)
      bboxes = np.asarray([0, 0, 1, 1])
      bboxes = np.vstack([bboxes for _ in range(num_boxes)])
      bboxes = math_ops.cast(bboxes, dtypes.float32)
      bboxes = array_ops.expand_dims(bboxes, 0)
      image = ops.convert_to_tensor(image)
      image = image_ops_impl.convert_image_dtype(image, dtypes.float32)
      image = array_ops.expand_dims(image, 0)
      image = image_ops.draw_bounding_boxes(image, bboxes, colors=colors)
      with self.cached_session(use_gpu=False) as sess:
        op_drawn_image = np.squeeze(sess.run(image), 0)
        self.assertAllEqual(test_drawn_image, op_drawn_image)
  def testDrawBoundingBoxRGBColorCycling(self):
    image = np.zeros([10, 10, 3], "float32")
    self._testDrawBoundingBoxColorCycling(image)
  def testDrawBoundingBoxRGBAColorCycling(self):
    image = np.zeros([10, 10, 4], "float32")
    self._testDrawBoundingBoxColorCycling(image)
  def testDrawBoundingBoxGRY(self):
    image = np.zeros([4, 4, 1], "float32")
    self._testDrawBoundingBoxColorCycling(image)
  def testDrawBoundingBoxRGBColorCyclingWithColors(self):
    image = np.zeros([10, 10, 3], "float32")
    colors = np.asarray([[1, 1, 0, 1], [0, 0, 1, 1], [0.5, 0, 0.5, 1],
                         [0.5, 0.5, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]])
    self._testDrawBoundingBoxColorCycling(image, colors=colors)
  def testDrawBoundingBoxRGBAColorCyclingWithColors(self):
    image = np.zeros([10, 10, 4], "float32")
    colors = np.asarray([[0.5, 0, 0.5, 1], [0.5, 0.5, 0, 1], [0.5, 0, 0, 1],
                         [0, 0, 0.5, 1]])
    self._testDrawBoundingBoxColorCycling(image, colors=colors)
if __name__ == "__main__":
  test.main()
