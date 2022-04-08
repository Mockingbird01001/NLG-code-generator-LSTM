
from tensorflow.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
FLOAT = dtypes.float32
FLOAT16 = dtypes.float16
INT32 = dtypes.int32
INT64 = dtypes.int64
STRING = dtypes.string
QUANTIZED_UINT8 = dtypes.uint8
INT8 = dtypes.int8
INT16 = dtypes.int16
COMPLEX64 = dtypes.complex64
TENSORFLOW_GRAPHDEF = _toco_flags_pb2.TENSORFLOW_GRAPHDEF
TFLITE = _toco_flags_pb2.TFLITE
GRAPHVIZ_DOT = _toco_flags_pb2.GRAPHVIZ_DOT
_tf_export(v1=["lite.constants.FLOAT"]).export_constant(__name__, "FLOAT")
_tf_export(v1=["lite.constants.FLOAT16"]).export_constant(__name__, "FLOAT16")
_tf_export(v1=["lite.constants.INT32"]).export_constant(__name__, "INT32")
_tf_export(v1=["lite.constants.INT64"]).export_constant(__name__, "INT64")
_tf_export(v1=["lite.constants.STRING"]).export_constant(__name__, "STRING")
_tf_export(v1=["lite.constants.QUANTIZED_UINT8"]).export_constant(
    __name__, "QUANTIZED_UINT8")
_tf_export(v1=["lite.constants.INT8"]).export_constant(__name__, "INT8")
_tf_export(v1=["lite.constants.INT16"]).export_constant(__name__, "INT16")
_tf_export(v1=["lite.constants.TFLITE"]).export_constant(__name__, "TFLITE")
_tf_export(v1=["lite.constants.GRAPHVIZ_DOT"]).export_constant(
    __name__, "GRAPHVIZ_DOT")
EXPERIMENTAL_USE_TOCO_API_DIRECTLY = False
_allowed_symbols = [
    "FLOAT",
    "FLOAT16",
    "INT32",
    "INT64",
    "STRING",
    "QUANTIZED_UINT8",
    "INT8",
    "INT16",
    "COMPLEX64",
    "TENSORFLOW_GRAPHDEF",
    "TFLITE",
    "GRAPHVIZ_DOT",
    "EXPERIMENTAL_USE_TOCO_API_DIRECTLY",
]
remove_undocumented(__name__, _allowed_symbols)
