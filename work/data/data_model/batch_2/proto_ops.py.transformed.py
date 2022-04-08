
from tensorflow.python.framework import ops
from tensorflow.python.ops.gen_decode_proto_ops import decode_proto_v2 as decode_proto
from tensorflow.python.ops.gen_encode_proto_ops import encode_proto
ops.NotDifferentiable("DecodeProtoV2")
ops.NotDifferentiable("EncodeProto")
