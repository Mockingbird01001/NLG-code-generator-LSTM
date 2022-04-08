
import json
import tempfile
from tensorflow.lite.schema import upgrade_schema as upgrade_schema_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
EMPTY_TEST_SCHEMA_V1 = {
    "version": 1,
    "operator_codes": [],
    "subgraphs": [],
}
EMPTY_TEST_SCHEMA_V3 = {
    "version": 3,
    "operator_codes": [],
    "subgraphs": [],
    "buffers": [{
        "data": []
    }]
}
TEST_SCHEMA_V0 = {
    "operator_codes": [],
    "tensors": [],
    "inputs": [],
    "outputs": [],
    "operators": [],
    "version": 0
}
TEST_SCHEMA_V3 = {
    "operator_codes": [],
    "buffers": [{
        "data": []
    }],
    "subgraphs": [{
        "tensors": [],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "version":
        3
}
FULL_TEST_SCHEMA_V1 = {
    "version":
        1,
    "operator_codes": [
        {
            "builtin_code": "CONVOLUTION"
        },
        {
            "builtin_code": "DEPTHWISE_CONVOLUTION"
        },
        {
            "builtin_code": "AVERAGE_POOL"
        },
        {
            "builtin_code": "MAX_POOL"
        },
        {
            "builtin_code": "L2_POOL"
        },
        {
            "builtin_code": "SIGMOID"
        },
        {
            "builtin_code": "L2NORM"
        },
        {
            "builtin_code": "LOCAL_RESPONSE_NORM"
        },
        {
            "builtin_code": "ADD"
        },
        {
            "builtin_code": "Basic_RNN"
        },
    ],
    "subgraphs": [{
        "operators": [
            {
                "builtin_options_type": "PoolOptions"
            },
            {
                "builtin_options_type": "DepthwiseConvolutionOptions"
            },
            {
                "builtin_options_type": "ConvolutionOptions"
            },
            {
                "builtin_options_type": "LocalResponseNormOptions"
            },
            {
                "builtin_options_type": "BasicRNNOptions"
            },
        ],
    }],
    "description":
        "",
}
FULL_TEST_SCHEMA_V3 = {
    "version":
        3,
    "operator_codes": [
        {
            "builtin_code": "CONV_2D"
        },
        {
            "builtin_code": "DEPTHWISE_CONV_2D"
        },
        {
            "builtin_code": "AVERAGE_POOL_2D"
        },
        {
            "builtin_code": "MAX_POOL_2D"
        },
        {
            "builtin_code": "L2_POOL_2D"
        },
        {
            "builtin_code": "LOGISTIC"
        },
        {
            "builtin_code": "L2_NORMALIZATION"
        },
        {
            "builtin_code": "LOCAL_RESPONSE_NORMALIZATION"
        },
        {
            "builtin_code": "ADD"
        },
        {
            "builtin_code": "RNN"
        },
    ],
    "subgraphs": [{
        "operators": [
            {
                "builtin_options_type": "Pool2DOptions"
            },
            {
                "builtin_options_type": "DepthwiseConv2DOptions"
            },
            {
                "builtin_options_type": "Conv2DOptions"
            },
            {
                "builtin_options_type": "LocalResponseNormalizationOptions"
            },
            {
                "builtin_options_type": "RNNOptions"
            },
        ],
    }],
    "description":
        "",
    "buffers": [{
        "data": []
    }]
}
BUFFER_TEST_V2 = {
    "operator_codes": [],
    "buffers": [],
    "subgraphs": [{
        "tensors": [
            {
                "data_buffer": [1, 2, 3, 4]
            },
            {
                "data_buffer": [1, 2, 3, 4, 5, 6, 7, 8]
            },
            {
                "data_buffer": []
            },
        ],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "version":
        2
}
BUFFER_TEST_V3 = {
    "operator_codes": [],
    "subgraphs": [{
        "tensors": [
            {
                "buffer": 1
            },
            {
                "buffer": 2
            },
            {
                "buffer": 0
            },
        ],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "buffers": [
        {
            "data": []
        },
        {
            "data": [1, 2, 3, 4]
        },
        {
            "data": [1, 2, 3, 4, 5, 6, 7, 8]
        },
    ],
    "version":
        3
}
def JsonDumpAndFlush(data, fp):
  """Write the dictionary `data` to a JSON file `fp` (and flush).
  Args:
    data: in a dictionary that is JSON serializable.
    fp: File-like object
  """
  json.dump(data, fp)
  fp.flush()
class TestSchemaUpgrade(test_util.TensorFlowTestCase):
  def testNonExistentFile(self):
    converter = upgrade_schema_lib.Converter()
    with self.assertRaisesRegex(IOError, "No such file or directory"):
      converter.Convert(non_existent, non_existent)
  def testInvalidExtension(self):
    converter = upgrade_schema_lib.Converter()
    with self.assertRaisesRegex(ValueError, "Invalid extension on input"):
      converter.Convert(invalid_extension, invalid_extension)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_json:
      JsonDumpAndFlush(EMPTY_TEST_SCHEMA_V1, in_json)
      with self.assertRaisesRegex(ValueError, "Invalid extension on output"):
        converter.Convert(in_json.name, invalid_extension)
  def CheckConversion(self, data_old, data_expected):
    """Given a data dictionary, test upgrading to current version.
    Args:
        data_old: TFLite model as a dictionary (arbitrary version).
        data_expected: TFLite model as a dictionary (upgraded).
    """
    converter = upgrade_schema_lib.Converter()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_json, \
            tempfile.NamedTemporaryFile(
                suffix=".json", mode="w+") as out_json, \
            tempfile.NamedTemporaryFile(
                suffix=".bin", mode="w+b") as out_bin, \
            tempfile.NamedTemporaryFile(
                suffix=".tflite", mode="w+b") as out_tflite:
      JsonDumpAndFlush(data_old, in_json)
      converter.Convert(in_json.name, out_json.name)
      converter.Convert(in_json.name, out_tflite.name)
      converter.Convert(out_tflite.name, out_bin.name)
      self.assertEqual(
          open(out_bin.name, "rb").read(),
          open(out_tflite.name, "rb").read())
      converted_schema = json.load(out_json)
      self.assertEqual(converted_schema, data_expected)
  def testAlreadyUpgraded(self):
    self.CheckConversion(EMPTY_TEST_SCHEMA_V3, EMPTY_TEST_SCHEMA_V3)
    self.CheckConversion(TEST_SCHEMA_V3, TEST_SCHEMA_V3)
    self.CheckConversion(BUFFER_TEST_V3, BUFFER_TEST_V3)
  def testV1Upgrade_RenameOps(self):
    self.CheckConversion(EMPTY_TEST_SCHEMA_V1, EMPTY_TEST_SCHEMA_V3)
    self.CheckConversion(FULL_TEST_SCHEMA_V1, FULL_TEST_SCHEMA_V3)
  def testV2Upgrade_CreateBuffers(self):
    self.CheckConversion(BUFFER_TEST_V2, BUFFER_TEST_V3)
if __name__ == "__main__":
  test_lib.main()
