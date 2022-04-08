
from typing import NamedTuple
import numpy as np
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_segmentation_calculator_pb2
from mediapipe.calculators.util import local_file_contents_calculator_pb2
from mediapipe.framework.tool import switch_container_pb2
from mediapipe.python.solution_base import SolutionBase
BINARYPB_FILE_PATH = 'mediapipe/modules/selfie_segmentation/selfie_segmentation_cpu.binarypb'
class SelfieSegmentation(SolutionBase):
  def __init__(self, model_selection=0):
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        side_inputs={
            'model_selection': model_selection,
        },
        outputs=['segmentation_mask'])
  def process(self, image: np.ndarray) -> NamedTuple:
    return super().process(input_data={'image': image})
