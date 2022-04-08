
from typing import NamedTuple
import numpy as np
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_classification_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_floats_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_landmarks_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import landmark_projection_calculator_pb2
from mediapipe.calculators.util import local_file_contents_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.framework.tool import switch_container_pb2
from mediapipe.modules.holistic_landmark.calculators import roi_tracking_calculator_pb2
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions import download_utils
from mediapipe.python.solutions.face_mesh import FACE_CONNECTIONS
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import POSE_CONNECTIONS
from mediapipe.python.solutions.pose import PoseLandmark
BINARYPB_FILE_PATH = 'mediapipe/modules/holistic_landmark/holistic_landmark_cpu.binarypb'
def _download_oss_pose_landmark_model(model_complexity):
  if model_complexity == 0:
    download_utils.download_oss_model(
        'mediapipe/modules/pose_landmark/pose_landmark_lite.tflite')
  elif model_complexity == 2:
    download_utils.download_oss_model(
        'mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite')
class Holistic(SolutionBase):
  def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
    _download_oss_pose_landmark_model(model_complexity)
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        side_inputs={
            'model_complexity': model_complexity,
            'smooth_landmarks': smooth_landmarks and not static_image_mode,
        },
        calculator_params={
            'poselandmarkcpu__ConstantSidePacketCalculator.packet': [
                constant_side_packet_calculator_pb2
                .ConstantSidePacketCalculatorOptions.ConstantSidePacket(
                    bool_value=not static_image_mode)
            ],
            'poselandmarkcpu__posedetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'poselandmarkcpu__poselandmarkbyroicpu__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=[
            'pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks',
            'face_landmarks'
        ])
  def process(self, image: np.ndarray) -> NamedTuple:
    results = super().process(input_data={'image': image})
    if results.pose_landmarks:
      for landmark in results.pose_landmarks.landmark:
        landmark.ClearField('presence')
    return results
