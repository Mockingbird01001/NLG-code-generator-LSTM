
import enum
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
@tf_export("tpu.experimental.HardwareFeature")
class HardwareFeature(object):
  def __init__(self, tpu_hardware_feature_proto):
    self.tpu_hardware_feature_proto = tpu_hardware_feature_proto
  class EmbeddingFeature(enum.Enum):
    UNSUPPORTED = "UNSUPPORTED"
    V1 = "V1"
    V2 = "V2"
  @classmethod
  def _embedding_feature_proto_to_string(cls, embedding_feature_proto):
    embedding_feature_proto_to_string_map = {
        topology_pb2.TPUHardwareFeature.EmbeddingFeature.UNSUPPORTED:
            HardwareFeature.EmbeddingFeature.UNSUPPORTED,
        topology_pb2.TPUHardwareFeature.EmbeddingFeature.V1:
            HardwareFeature.EmbeddingFeature.V1,
        topology_pb2.TPUHardwareFeature.EmbeddingFeature.V2:
            HardwareFeature.EmbeddingFeature.V2
    }
    return embedding_feature_proto_to_string_map.get(
        embedding_feature_proto, HardwareFeature.EmbeddingFeature.UNSUPPORTED)
  @property
  def embedding_feature(self):
    return HardwareFeature._embedding_feature_proto_to_string(
        self.tpu_hardware_feature_proto.embedding_feature)
