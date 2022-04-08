
from tensorflow.python.framework import dtypes
STR_TO_TFLITE_TYPES = {
    'INT8': dtypes.int8,
    'UINT8': dtypes.uint8,
    'INT16': dtypes.int16,
}
TFLITE_TO_STR_TYPES = {v: k for k, v in STR_TO_TFLITE_TYPES.items()}
STR_TYPES = STR_TO_TFLITE_TYPES.keys()
TFLITE_TYPES = STR_TO_TFLITE_TYPES.values()
DEFAULT_STR_TYPE = 'INT8'
DEFAULT_TFLITE_TYPE = STR_TO_TFLITE_TYPES[DEFAULT_STR_TYPE]
