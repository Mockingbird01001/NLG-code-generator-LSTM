
import enum
from typing import NamedTuple, Union
import numpy as np
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.python.solution_base import SolutionBase
BINARYPB_FILE_PATH = 'mediapipe/modules/face_detection/face_detection_front_cpu.binarypb'
def get_key_point(
    detection: detection_pb2.Detection, key_point_enum: 'FaceKeyPoint'
) -> Union[None, location_data_pb2.LocationData.RelativeKeypoint]:
  if not detection or not detection.location_data:
    return None
  return detection.location_data.relative_keypoints[key_point_enum]
class FaceKeyPoint(enum.IntEnum):
  RIGHT_EYE = 0
  LEFT_EYE = 1
  NOSE_TIP = 2
  MOUTH_CENTER = 3
  RIGHT_EAR_TRAGION = 4
  LEFT_EAR_TRAGION = 5
class FaceDetection(SolutionBase):
  def __init__(self, min_detection_confidence=0.5):
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        calculator_params={
            'facedetectionfrontcommon__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
        },
        outputs=['detections'])
  def process(self, image: np.ndarray) -> NamedTuple:
    return super().process(input_data={'image': image})
