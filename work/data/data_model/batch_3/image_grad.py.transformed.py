
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
@ops.RegisterGradient("ResizeNearestNeighbor")
def _ResizeNearestNeighborGrad(op, grad):
  image = op.inputs[0]
  if image.get_shape()[1:3].is_fully_defined():
    image_shape = image.get_shape()[1:3]
  else:
    image_shape = array_ops.shape(image)[1:3]
  grads = gen_image_ops.resize_nearest_neighbor_grad(
      grad,
      image_shape,
      align_corners=op.get_attr("align_corners"),
      half_pixel_centers=op.get_attr("half_pixel_centers"))
  return [grads, None]
@ops.RegisterGradient("ResizeBilinear")
def _ResizeBilinearGrad(op, grad):
  grad0 = gen_image_ops.resize_bilinear_grad(
      grad,
      op.inputs[0],
      align_corners=op.get_attr("align_corners"),
      half_pixel_centers=op.get_attr("half_pixel_centers"))
  return [grad0, None]
@ops.RegisterGradient("ScaleAndTranslate")
def _ScaleAndTranslateGrad(op, grad):
  grad0 = gen_image_ops.scale_and_translate_grad(
      grad,
      op.inputs[0],
      op.inputs[2],
      op.inputs[3],
      kernel_type=op.get_attr("kernel_type"),
      antialias=op.get_attr("antialias"))
  return [grad0, None, None, None]
@ops.RegisterGradient("ResizeBicubic")
def _ResizeBicubicGrad(op, grad):
  allowed_types = [dtypes.float32, dtypes.float64]
  grad0 = None
  if op.inputs[0].dtype in allowed_types:
    grad0 = gen_image_ops.resize_bicubic_grad(
        grad,
        op.inputs[0],
        align_corners=op.get_attr("align_corners"),
        half_pixel_centers=op.get_attr("half_pixel_centers"))
  return [grad0, None]
@ops.RegisterGradient("CropAndResize")
def _CropAndResizeGrad(op, grad):
  image = op.inputs[0]
  if image.get_shape().is_fully_defined():
    image_shape = image.get_shape().as_list()
  else:
    image_shape = array_ops.shape(image)
  allowed_types = [dtypes.float16, dtypes.float32, dtypes.float64]
  if op.inputs[0].dtype in allowed_types:
    grad0 = gen_image_ops.crop_and_resize_grad_image(
        grad, op.inputs[1], op.inputs[2], image_shape, T=op.get_attr("T"),
        method=op.get_attr("method"))
  else:
    grad0 = None
  grad1 = gen_image_ops.crop_and_resize_grad_boxes(
      grad, op.inputs[0], op.inputs[1], op.inputs[2])
  return [grad0, grad1, None, None]
def _CustomReciprocal(x):
  """Wrapper function around `math_ops.div_no_nan()` to perform a "safe" reciprocal incase the input is zero. Avoids divide by zero and NaNs.
  Input:
    x -> input tensor to be reciprocat-ed.
  Returns:
    x_reciprocal -> reciprocal of x without NaNs.
  """
  return math_ops.div_no_nan(1.0, x)
@ops.RegisterGradient("RGBToHSV")
def _RGBToHSVGrad(op, grad):
  reds = op.inputs[0][..., 0]
  greens = op.inputs[0][..., 1]
  blues = op.inputs[0][..., 2]
  saturation = op.outputs[0][..., 1]
  value = op.outputs[0][..., 2]
  red_biggest = math_ops.cast((reds >= blues) & \
                 (reds >= greens), dtypes.float32)
  green_biggest = math_ops.cast((greens > reds) & \
                   (greens >= blues), dtypes.float32)
  blue_biggest = math_ops.cast((blues > reds) & \
                  (blues > greens), dtypes.float32)
  red_smallest = math_ops.cast((reds < blues) & \
                  (reds < greens), dtypes.float32)
  green_smallest = math_ops.cast((greens <= reds) & \
                    (greens < blues), dtypes.float32)
  blue_smallest = math_ops.cast((blues <= reds) & \
                   (blues <= greens), dtypes.float32)
  dv_dr = red_biggest
  dv_dg = green_biggest
  dv_db = blue_biggest
  ds_dr = math_ops.cast(reds > 0, dtypes.float32) * \
               math_ops.add(red_biggest * \
               math_ops.add(green_smallest * greens, blue_smallest * blues) * \
               _CustomReciprocal(math_ops.square(reds)),\
               red_smallest * -1 * _CustomReciprocal((green_biggest * \
               greens) + (blue_biggest * blues)))
  ds_dg = math_ops.cast(greens > 0, dtypes.float32) * \
               math_ops.add(green_biggest * \
               math_ops.add(red_smallest * reds, blue_smallest * blues) * \
               _CustomReciprocal(math_ops.square(greens)),\
               green_smallest * -1 * _CustomReciprocal((red_biggest * \
               reds) + (blue_biggest * blues)))
  ds_db = math_ops.cast(blues > 0, dtypes.float32) * \
               math_ops.add(blue_biggest * \
               math_ops.add(green_smallest * greens, red_smallest * reds) * \
               _CustomReciprocal(math_ops.square(blues)),\
               blue_smallest * -1 * _CustomReciprocal((green_biggest * \
               greens) + (red_biggest * reds)))
  dh_dr_1 = 60 * (math_ops.cast(reds > 0, dtypes.float32) * red_biggest * \
                  -1  * \
                  (greens - blues) * \
                  _CustomReciprocal(math_ops.square(saturation)) *\
                  _CustomReciprocal(math_ops.square(value)))
  dh_dr_2 = 60 * (math_ops.cast(greens > 0, dtypes.float32) * green_biggest * \
                  red_smallest  * (blues - greens) * \
                  _CustomReciprocal(math_ops.square(reds - greens)))
  dh_dr_3 = 60 * (math_ops.cast(greens > 0, dtypes.float32) * green_biggest * \
                  blue_smallest * -1 * _CustomReciprocal(greens - blues))
  dh_dr_4 = 60 * (math_ops.cast(blues > 0, dtypes.float32) * blue_biggest * \
                  red_smallest * (blues - greens) * \
                  _CustomReciprocal(math_ops.square(blues - reds)))
  dh_dr_5 = 60 * (math_ops.cast(blues > 0, dtypes.float32) * blue_biggest * \
                  green_smallest * _CustomReciprocal(blues - greens))
  dh_dr = dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5
  dh_dr = dh_dr / 360
  dh_dg_1 = 60 * (math_ops.cast(greens > 0, dtypes.float32) * green_biggest * \
                  -1 * (blues - reds) * \
                  _CustomReciprocal(math_ops.square(saturation))\
                  * _CustomReciprocal(math_ops.square(value)))
  dh_dg_2 = 60 * (math_ops.cast(reds > 0, dtypes.float32) * red_biggest * \
                  green_smallest * (reds - blues) * \
                  _CustomReciprocal(math_ops.square(reds - greens)))
  dh_dg_3 = 60 * (math_ops.cast(reds > 0, dtypes.float32) * red_biggest * \
                  blue_smallest * _CustomReciprocal(reds - blues))
  dh_dg_4 = 60 * (math_ops.cast(blues > 0, dtypes.float32) * blue_biggest * \
                  green_smallest * (reds - blues) * \
                  _CustomReciprocal(math_ops.square(blues - greens)))
  dh_dg_5 = 60 * (math_ops.cast(blues > 0, dtypes.float32) * blue_biggest * \
                  red_smallest * -1 * _CustomReciprocal(blues - reds))
  dh_dg = dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5
  dh_dg = dh_dg / 360
  dh_db_1 = 60 * (math_ops.cast(blues > 0, dtypes.float32) * blue_biggest * \
                   -1 * \
                  (reds - greens) * \
                  _CustomReciprocal(math_ops.square(saturation)) * \
                  _CustomReciprocal(math_ops.square(value)))
  dh_db_2 = 60 * (math_ops.cast(reds > 0, dtypes.float32) * red_biggest *\
                  blue_smallest * (greens - reds) * \
                  _CustomReciprocal(math_ops.square(reds - blues)))
  dh_db_3 = 60 * (math_ops.cast(reds > 0, dtypes.float32) * red_biggest * \
                  green_smallest * -1 * _CustomReciprocal(reds - greens))
  dh_db_4 = 60 * (math_ops.cast(greens > 0, dtypes.float32) * green_biggest * \
                  blue_smallest * (greens - reds) * \
                  _CustomReciprocal(math_ops.square(greens - blues)))
  dh_db_5 = 60 * (math_ops.cast(greens > 0, dtypes.float32) * green_biggest * \
                  red_smallest * _CustomReciprocal(greens - reds))
  dh_db = dh_db_1 + dh_db_2 + dh_db_3 + dh_db_4 + dh_db_5
  dh_db = dh_db / 360
  dv_drgb = array_ops.stack(
      [grad[..., 2] * dv_dr, grad[..., 2] * dv_dg, grad[..., 2] * dv_db],
      axis=-1)
  ds_drgb = array_ops.stack(
      [grad[..., 1] * ds_dr, grad[..., 1] * ds_dg, grad[..., 1] * ds_db],
      axis=-1)
  dh_drgb = array_ops.stack(
      [grad[..., 0] * dh_dr, grad[..., 0] * dh_dg, grad[..., 0] * dh_db],
      axis=-1)
  gradient_input = math_ops.add(math_ops.add(dv_drgb, ds_drgb), dh_drgb)
  return gradient_input
