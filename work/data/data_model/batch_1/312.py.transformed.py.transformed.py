
import math
from typing import List, Optional, Tuple, Union
import cv2
import dataclasses
import numpy as np
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2
PRESENCE_THRESHOLD = 0.5
RGB_CHANNELS = 3
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
VISIBILITY_THRESHOLD = 0.5
@dataclasses.dataclass
class DrawingSpec:
  color: Tuple[int, int, int] = (0, 255, 0)
  thickness: int = 2
  circle_radius: int = 2
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))
  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px
def draw_detection(
    image: np.ndarray,
    detection: detection_pb2.Detection,
    keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    bbox_drawing_spec: DrawingSpec = DrawingSpec()):
  if not detection.location_data:
    return
  if image.shape[2] != RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  location = detection.location_data
  if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
    raise ValueError(
        'LocationData must be relative for this drawing funtion to work.')
  for keypoint in location.relative_keypoints:
    keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                   image_cols, image_rows)
    cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
               keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
  if not location.HasField('relative_bounding_box'):
    return
  relative_bounding_box = location.relative_bounding_box
  rect_start_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
      image_rows)
  rect_end_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin + relative_bounding_box.width,
      relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
      image_rows)
  cv2.rectangle(image, rect_start_point, rect_end_point,
                bbox_drawing_spec.color, bbox_drawing_spec.thickness)
def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: DrawingSpec = DrawingSpec()):
  if not landmark_list:
    return
  if image.shape[2] != RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  if connections:
    num_landmarks = len(landmark_list.landmark)
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], connection_drawing_spec.color,
                 connection_drawing_spec.thickness)
  for landmark_px in idx_to_coordinates.values():
    cv2.circle(image, landmark_px, landmark_drawing_spec.circle_radius,
               landmark_drawing_spec.color, landmark_drawing_spec.thickness)
def draw_axis(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    focal_length: Tuple[float, float] = (1.0, 1.0),
    principal_point: Tuple[float, float] = (0.0, 0.0),
    axis_length: float = 0.1,
    axis_drawing_spec: DrawingSpec = DrawingSpec()):
  if image.shape[2] != RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
  axis_cam = np.matmul(rotation, axis_length*axis_world.T).T + translation
  x = axis_cam[..., 0]
  y = axis_cam[..., 1]
  z = axis_cam[..., 2]
  fx, fy = focal_length
  px, py = principal_point
  x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
  y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
  x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
  y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
  origin = (x_im[0], y_im[0])
  x_axis = (x_im[1], y_im[1])
  y_axis = (x_im[2], y_im[2])
  z_axis = (x_im[3], y_im[3])
  cv2.arrowedLine(image, origin, x_axis, RED_COLOR,
                  axis_drawing_spec.thickness)
  cv2.arrowedLine(image, origin, y_axis, GREEN_COLOR,
                  axis_drawing_spec.thickness)
  cv2.arrowedLine(image, origin, z_axis, BLUE_COLOR,
                  axis_drawing_spec.thickness)
