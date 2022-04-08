
import sys
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
FLAGS = flags.FLAGS
flags.DEFINE_multi_string("image_file_names", None,
                          "List of paths to the input images.")
flags.DEFINE_integer("width", 96, "Width to scale images to.")
flags.DEFINE_integer("height", 96, "Height to scale images to.")
flags.DEFINE_boolean("want_grayscale", False,
                     "Whether to convert the image to monochrome.")
def get_image(width, height, want_grayscale, filepath):
  """Returns an image loaded into an np.ndarray with dims [height, width, (3 or 1)].
  Args:
    width: Width to rescale the image to.
    height: Height to rescale the image to.
    want_grayscale: Whether the result should be converted to grayscale.
    filepath: Path of the image file..
  Returns:
    np.ndarray of shape (height, width, channels) where channels is 1 if
      want_grayscale is true, otherwise 3.
  """
  with ops.Graph().as_default():
    with session.Session():
      file_data = io_ops.read_file(filepath)
      channels = 1 if want_grayscale else 3
      image_tensor = image_ops.decode_image(file_data, channels=channels).eval()
      resized_tensor = image_ops.resize_images_v2(image_tensor,
                                                  (height, width)).eval()
  return resized_tensor
def array_to_int_csv(array_data):
  flattened_array = array_data.flatten()
  array_as_strings = [item.astype(int).astype(str) for item in flattened_array]
  return ",".join(array_as_strings)
def main(_):
  for image_file_name in FLAGS.image_file_names:
    try:
      image_data = get_image(FLAGS.width, FLAGS.height, FLAGS.want_grayscale,
                             image_file_name)
      print(array_to_int_csv(image_data))
    except NotFoundError:
      sys.stderr.write("Image file not found at {0}\n".format(image_file_name))
      sys.exit(1)
if __name__ == "__main__":
  app.run(main)
