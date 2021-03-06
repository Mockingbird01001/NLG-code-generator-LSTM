
import os
import time
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
class DecodeJpegBenchmark(test.Benchmark):
  def _evalDecodeJpeg(self,
                      image_name,
                      parallelism,
                      num_iters,
                      crop_during_decode=None,
                      crop_window=None,
                      tile=None):
    """Evaluate DecodeJpegOp for the given image.
    TODO(tanmingxing): add decoding+cropping as well.
    Args:
      image_name: a string of image file name (without suffix).
      parallelism: the number of concurrent decode_jpeg ops to be run.
      num_iters: number of iterations for evaluation.
      crop_during_decode: If true, use fused DecodeAndCropJpeg instead of
          separate decode and crop ops. It is ignored if crop_window is None.
      crop_window: if not None, crop the decoded image. Depending on
          crop_during_decode, cropping could happen during or after decoding.
      tile: if not None, tile the image to composite a larger fake image.
    Returns:
      The duration of the run in seconds.
    """
    ops.reset_default_graph()
    image_file_path = resource_loader.get_path_to_datafile(
        os.path.join('core', 'lib', 'jpeg', 'testdata', image_name))
    if not os.path.exists(image_file_path):
      image_file_path = resource_loader.get_path_to_datafile(
          os.path.join(
              '..', '..', 'core', 'lib', 'jpeg', 'testdata', image_name))
    if tile is None:
      image_content = variable_scope.get_variable(
          'image_%s' % image_name,
          initializer=io_ops.read_file(image_file_path))
    else:
      single_image = image_ops.decode_jpeg(
          io_ops.read_file(image_file_path), channels=3, name='single_image')
      tiled_image = array_ops.tile(single_image, tile)
      image_content = variable_scope.get_variable(
          'tiled_image_%s' % image_name,
          initializer=image_ops.encode_jpeg(tiled_image))
    with session.Session() as sess:
      self.evaluate(variables.global_variables_initializer())
      images = []
      for _ in range(parallelism):
        if crop_window is None:
          image = image_ops.decode_jpeg(image_content, channels=3)
        elif crop_during_decode:
          image = image_ops.decode_and_crop_jpeg(
              image_content, crop_window, channels=3)
        else:
          image = image_ops.decode_jpeg(image_content, channels=3)
          image = image_ops.crop_to_bounding_box(
              image,
              offset_height=crop_window[0],
              offset_width=crop_window[1],
              target_height=crop_window[2],
              target_width=crop_window[3])
        images.append(image)
      r = control_flow_ops.group(*images)
      for _ in range(3):
        self.evaluate(r)
      start_time = time.time()
      for _ in range(num_iters):
        self.evaluate(r)
      end_time = time.time()
    return end_time - start_time
  def benchmarkDecodeJpegSmall(self):
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    for parallelism in [1, 100]:
      duration_decode = self._evalDecodeJpeg('small.jpg', parallelism,
                                             num_iters)
      duration_decode_crop = self._evalDecodeJpeg('small.jpg', parallelism,
                                                  num_iters, False, crop_window)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'small.jpg', parallelism, num_iters, True, crop_window)
      self.report_benchmark(
          name='decode_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)
  def benchmarkDecodeJpegMedium(self):
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    for parallelism in [1, 100]:
      duration_decode = self._evalDecodeJpeg('medium.jpg', parallelism,
                                             num_iters)
      duration_decode_crop = self._evalDecodeJpeg('medium.jpg', parallelism,
                                                  num_iters, False, crop_window)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, True, crop_window)
      self.report_benchmark(
          name='decode_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)
  def benchmarkDecodeJpegLarge(self):
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    tile = [4, 4, 1]
    for parallelism in [1, 100]:
      duration_decode = self._evalDecodeJpeg('medium.jpg', parallelism,
                                             num_iters, tile)
      duration_decode_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, False, crop_window, tile)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, True, crop_window, tile)
      self.report_benchmark(
          name='decode_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)
if __name__ == '__main__':
  test.main()
