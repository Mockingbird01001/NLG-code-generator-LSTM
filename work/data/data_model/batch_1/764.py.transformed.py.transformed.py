
import enum
from typing import List, Tuple, NamedTuple, Optional
import attr
import numpy as np
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_floats_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_landmarks_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import association_calculator_pb2
from mediapipe.calculators.util import collection_has_min_size_calculator_pb2
from mediapipe.calculators.util import detection_label_id_to_text_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import landmark_projection_calculator_pb2
from mediapipe.calculators.util import local_file_contents_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.modules.objectron.calculators import annotation_data_pb2
from mediapipe.modules.objectron.calculators import frame_annotation_to_rect_calculator_pb2
from mediapipe.modules.objectron.calculators import lift_2d_frame_annotation_to_3d_calculator_pb2
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions import download_utils
class BoxLandmark(enum.IntEnum):
  CENTER = 0
  BACK_BOTTOM_LEFT = 1
  FRONT_BOTTOM_LEFT = 2
  BACK_TOP_LEFT = 3
  FRONT_TOP_LEFT = 4
  BACK_BOTTOM_RIGHT = 5
  FRONT_BOTTOM_RIGHT = 6
  BACK_TOP_RIGHT = 7
  FRONT_TOP_RIGHT = 8
BINARYPB_FILE_PATH = 'mediapipe/modules/objectron/objectron_cpu.binarypb'
BOX_CONNECTIONS = frozenset([
    (BoxLandmark.BACK_BOTTOM_LEFT, BoxLandmark.FRONT_BOTTOM_LEFT),
    (BoxLandmark.BACK_BOTTOM_LEFT, BoxLandmark.BACK_TOP_LEFT),
    (BoxLandmark.BACK_BOTTOM_LEFT, BoxLandmark.BACK_BOTTOM_RIGHT),
    (BoxLandmark.FRONT_BOTTOM_LEFT, BoxLandmark.FRONT_TOP_LEFT),
    (BoxLandmark.FRONT_BOTTOM_LEFT, BoxLandmark.FRONT_BOTTOM_RIGHT),
    (BoxLandmark.BACK_TOP_LEFT, BoxLandmark.FRONT_TOP_LEFT),
    (BoxLandmark.BACK_TOP_LEFT, BoxLandmark.BACK_TOP_RIGHT),
    (BoxLandmark.FRONT_TOP_LEFT, BoxLandmark.FRONT_TOP_RIGHT),
    (BoxLandmark.BACK_BOTTOM_RIGHT, BoxLandmark.FRONT_BOTTOM_RIGHT),
    (BoxLandmark.BACK_BOTTOM_RIGHT, BoxLandmark.BACK_TOP_RIGHT),
    (BoxLandmark.FRONT_BOTTOM_RIGHT, BoxLandmark.FRONT_TOP_RIGHT),
    (BoxLandmark.BACK_TOP_RIGHT, BoxLandmark.FRONT_TOP_RIGHT),
])
@attr.s(auto_attribs=True)
class ObjectronModel(object):
  model_path: str
  label_name: str
@attr.s(auto_attribs=True, frozen=True)
class ShoeModel(ObjectronModel):
  model_path: str = ('mediapipe/modules/objectron/'
                     'object_detection_3d_sneakers.tflite')
  label_name: str = 'Footwear'
@attr.s(auto_attribs=True, frozen=True)
class ChairModel(ObjectronModel):
  model_path: str = ('mediapipe/modules/objectron/'
                     'object_detection_3d_chair.tflite')
  label_name: str = 'Chair'
@attr.s(auto_attribs=True, frozen=True)
class CameraModel(ObjectronModel):
  model_path: str = ('mediapipe/modules/objectron/'
                     'object_detection_3d_camera.tflite')
  label_name: str = 'Camera'
@attr.s(auto_attribs=True, frozen=True)
class CupModel(ObjectronModel):
  model_path: str = ('mediapipe/modules/objectron/'
                     'object_detection_3d_cup.tflite')
  label_name: str = 'Coffee cup, Mug'
_MODEL_DICT = {
    'Shoe': ShoeModel(),
    'Chair': ChairModel(),
    'Cup': CupModel(),
    'Camera': CameraModel()
}
def _download_oss_objectron_models(objectron_model: str):
  download_utils.download_oss_model(
      'mediapipe/modules/objectron/object_detection_ssd_mobilenetv2_oidv4_fp16.tflite'
  )
  download_utils.download_oss_model(objectron_model)
def get_model_by_name(name: str) -> ObjectronModel:
  if name not in _MODEL_DICT:
    raise ValueError(f'{name} is not a valid model name for Objectron.')
  _download_oss_objectron_models(_MODEL_DICT[name].model_path)
  return _MODEL_DICT[name]
@attr.s(auto_attribs=True)
class ObjectronOutputs(object):
  landmarks_2d: landmark_pb2.NormalizedLandmarkList
  landmarks_3d: landmark_pb2.LandmarkList
  rotation: np.ndarray
  translation: np.ndarray
  scale: np.ndarray
class Objectron(SolutionBase):
  def __init__(self,
               static_image_mode: bool = False,
               max_num_objects: int = 5,
               min_detection_confidence: float = 0.5,
               min_tracking_confidence: float = 0.99,
               model_name: str = 'Shoe',
               focal_length: Tuple[float, float] = (1.0, 1.0),
               principal_point: Tuple[float, float] = (0.0, 0.0),
               image_size: Optional[Tuple[int, int]] = None,
               ):
    fx, fy = focal_length
    px, py = principal_point
    if image_size is not None:
      half_width = image_size[0] / 2.0
      half_height = image_size[1] / 2.0
      fx = fx / half_width
      fy = fy / half_height
      px = - (px - half_width) / half_width
      py = - (py - half_height) / half_height
    model = get_model_by_name(model_name)
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        side_inputs={
            'box_landmark_model_path': model.model_path,
            'allowed_labels': model.label_name,
            'max_num_objects': max_num_objects,
        },
        calculator_params={
            'ConstantSidePacketCalculator.packet': [
                constant_side_packet_calculator_pb2
                .ConstantSidePacketCalculatorOptions.ConstantSidePacket(
                    bool_value=not static_image_mode)
            ],
            ('objectdetectionoidv4subgraph'
             '__TensorsToDetectionsCalculator.min_score_thresh'):
                min_detection_confidence,
            ('boxlandmarksubgraph__ThresholdingCalculator'
             '.threshold'):
                min_tracking_confidence,
            ('Lift2DFrameAnnotationTo3DCalculator'
             '.normalized_focal_x'): fx,
            ('Lift2DFrameAnnotationTo3DCalculator'
             '.normalized_focal_y'): fy,
            ('Lift2DFrameAnnotationTo3DCalculator'
             '.normalized_principal_point_x'): px,
            ('Lift2DFrameAnnotationTo3DCalculator'
             '.normalized_principal_point_y'): py,
        },
        outputs=['detected_objects'])
  def process(self, image: np.ndarray) -> NamedTuple:
    results = super().process(input_data={'image': image})
    if results.detected_objects:
      results.detected_objects = self._convert_format(results.detected_objects)
    else:
      results.detected_objects = None
    return results
  def _convert_format(
      self,
      inputs: annotation_data_pb2.FrameAnnotation) -> List[ObjectronOutputs]:
    new_outputs = list()
    for annotation in inputs.annotations:
      rotation = np.reshape(np.array(annotation.rotation), (3, 3))
      translation = np.array(annotation.translation)
      scale = np.array(annotation.scale)
      landmarks_2d = landmark_pb2.NormalizedLandmarkList()
      landmarks_3d = landmark_pb2.LandmarkList()
      for keypoint in annotation.keypoints:
        point_2d = keypoint.point_2d
        landmarks_2d.landmark.add(x=point_2d.x, y=point_2d.y)
        point_3d = keypoint.point_3d
        landmarks_3d.landmark.add(x=point_3d.x, y=point_3d.y, z=point_3d.z)
      new_outputs.append(ObjectronOutputs(landmarks_2d, landmarks_3d,
                                          rotation, translation, scale=scale))
    return new_outputs
