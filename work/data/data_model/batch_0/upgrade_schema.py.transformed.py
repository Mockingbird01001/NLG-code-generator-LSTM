
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tensorflow as tf
from tensorflow.python.platform import resource_loader
parser = argparse.ArgumentParser(
    description="Script to move TFLite models from pre-release schema to "
    "new schema.")
parser.add_argument(
    "input",
    type=str,
    help="Input TensorFlow lite file in `.json`, `.bin` or `.tflite` format.")
parser.add_argument(
    "output",
    type=str,
    help="Output json or bin TensorFlow lite model compliant with "
    "the new schema. Extension must be `.json`, `.bin` or `.tflite`.")
@contextlib.contextmanager
def TemporaryDirectoryResource():
  temporary = tempfile.mkdtemp()
  try:
    yield temporary
  finally:
    shutil.rmtree(temporary)
class Converter(object):
  """Converts TensorFlow flatbuffer models from old to new version of schema.
  This can convert between any version to the latest version. It uses
  an incremental upgrade strategy to go from version to version.
  Usage:
    converter = Converter()
    converter.Convert("a.tflite", "a.json")
    converter.Convert("b.json", "b.tflite")
  """
  def __init__(self):
    paths_to_try = [
    ]
    for p in paths_to_try:
      self._flatc_path = resource_loader.get_path_to_datafile(p)
      if os.path.exists(self._flatc_path): break
    def FindSchema(base_name):
      return resource_loader.get_path_to_datafile("%s" % base_name)
    self._schemas = [
        (0, FindSchema("schema_v0.fbs"), True, self._Upgrade0To1),
        (1, FindSchema("schema_v1.fbs"), True, self._Upgrade1To2),
        (2, FindSchema("schema_v2.fbs"), True, self._Upgrade2To3),
    ]
    self._schemas.sort()
    self._new_version, self._new_schema = self._schemas[-1][:2]
    self._upgrade_dispatch = {
        version: dispatch
        for version, unused1, unused2, dispatch in self._schemas}
  def _Read(self, input_file, schema, raw_binary=False):
    """Read a tflite model assuming the given flatbuffer schema.
    If `input_file` is in bin, then we must use flatc to convert the schema
    from binary to json.
    Args:
      input_file: a binary (flatbuffer) or json file to read from. Extension
        must  be `.tflite`, `.bin`, or `.json` for FlatBuffer Binary or
        FlatBuffer JSON.
      schema: which schema to use for reading
      raw_binary: whether to assume raw_binary (versions previous to v3)
        that lacked file_identifier require this.
    Raises:
      RuntimeError: 1. When flatc cannot be invoked.
                    2. When json file does not exists.
      ValueError: When the extension is not json or bin.
    Returns:
      A dictionary representing the read tflite model.
    """
    raw_binary = ["--raw-binary"] if raw_binary else []
    with TemporaryDirectoryResource() as tempdir:
      basename = os.path.basename(input_file)
      basename_no_extension, extension = os.path.splitext(basename)
      if extension in [".bin", ".tflite"]:
        returncode = subprocess.call([
            self._flatc_path,
            "-t",
            "--strict-json",
            "--defaults-json",
        ] + raw_binary + ["-o", tempdir, schema, "--", input_file])
        if returncode != 0:
          raise RuntimeError("flatc failed to convert from binary to json.")
        json_file = os.path.join(tempdir, basename_no_extension + ".json")
        if not os.path.exists(json_file):
          raise RuntimeError("Could not find %r" % json_file)
      elif extension == ".json":
        json_file = input_file
      else:
        raise ValueError("Invalid extension on input file %r" % input_file)
      return json.load(open(json_file))
  def _Write(self, data, output_file):
    """Output a json or bin version of the flatbuffer model.
    Args:
      data: Dict representing the TensorFlow Lite model to write.
      output_file: filename to write the converted flatbuffer to. (json,
        tflite, or bin extension is required).
    Raises:
      ValueError: When the extension is not json or bin
      RuntimeError: When flatc fails to convert json data to binary.
    """
    _, extension = os.path.splitext(output_file)
    with TemporaryDirectoryResource() as tempdir:
      if extension == ".json":
        json.dump(data, open(output_file, "w"), sort_keys=True, indent=2)
      elif extension in [".tflite", ".bin"]:
        input_json = os.path.join(tempdir, "temp.json")
        with open(input_json, "w") as fp:
          json.dump(data, fp, sort_keys=True, indent=2)
        returncode = subprocess.call([
            self._flatc_path, "-b", "--defaults-json", "--strict-json", "-o",
            tempdir, self._new_schema, input_json
        ])
        if returncode != 0:
          raise RuntimeError("flatc failed to convert upgraded json to binary.")
        shutil.copy(os.path.join(tempdir, "temp.tflite"), output_file)
      else:
        raise ValueError("Invalid extension on output file %r" % output_file)
  def _Upgrade0To1(self, data):
    """Upgrade data from Version 0 to Version 1.
    Changes: Added subgraphs (which contains a subset of formally global
    entries).
    Args:
      data: Dictionary representing the TensorFlow lite data to be upgraded.
        This will be modified in-place to be an upgraded version.
    """
    subgraph = {}
    for key_to_promote in ["tensors", "operators", "inputs", "outputs"]:
      subgraph[key_to_promote] = data[key_to_promote]
      del data[key_to_promote]
    data["subgraphs"] = [subgraph]
  def _Upgrade1To2(self, data):
    def RemapOperator(opcode_name):
      """Go from old schema op name to new schema op name.
      Args:
        opcode_name: String representing the ops (see :schema.fbs).
      Returns:
        Converted opcode_name from V1 to V2.
      """
      old_name_to_new_name = {
          "CONVOLUTION": "CONV_2D",
          "DEPTHWISE_CONVOLUTION": "DEPTHWISE_CONV_2D",
          "AVERAGE_POOL": "AVERAGE_POOL_2D",
          "MAX_POOL": "MAX_POOL_2D",
          "L2_POOL": "L2_POOL_2D",
          "SIGMOID": "LOGISTIC",
          "L2NORM": "L2_NORMALIZATION",
          "LOCAL_RESPONSE_NORM": "LOCAL_RESPONSE_NORMALIZATION",
          "Basic_RNN": "RNN",
      }
      return (old_name_to_new_name[opcode_name]
              if opcode_name in old_name_to_new_name else opcode_name)
    def RemapOperatorType(operator_type):
      """Remap operator structs from old names to new names.
      Args:
        operator_type: String representing the builtin operator data type
          string.
        (see :schema.fbs).
      Raises:
        ValueError: When the model has consistency problems.
      Returns:
        Upgraded builtin operator data type as a string.
      """
      old_to_new = {
          "PoolOptions": "Pool2DOptions",
          "DepthwiseConvolutionOptions": "DepthwiseConv2DOptions",
          "ConvolutionOptions": "Conv2DOptions",
          "LocalResponseNormOptions": "LocalResponseNormalizationOptions",
          "BasicRNNOptions": "RNNOptions",
      }
      return (old_to_new[operator_type]
              if operator_type in old_to_new else operator_type)
    for subgraph in data["subgraphs"]:
      for ops in subgraph["operators"]:
        ops["builtin_options_type"] = RemapOperatorType(
            ops["builtin_options_type"])
    for operator_code in data["operator_codes"]:
      if not isinstance(operator_code["builtin_code"], type(u"")):
        raise ValueError("builtin_code %r is non-string. this usually means "
                         "your model has consistency problems." %
                         (operator_code["builtin_code"]))
      operator_code["builtin_code"] = (RemapOperator(
          operator_code["builtin_code"]))
  def _Upgrade2To3(self, data):
    for subgraph in data["subgraphs"]:
      if "tensors" not in subgraph:
        continue
      for tensor in subgraph["tensors"]:
        if "data_buffer" not in tensor:
          tensor["buffer"] = 0
        else:
          if tensor["data_buffer"]:
            tensor[u"buffer"] = len(buffers)
            buffers.append({"data": tensor["data_buffer"]})
          else:
            tensor["buffer"] = 0
          del tensor["data_buffer"]
    data["buffers"] = buffers
  def _PerformUpgrade(self, data):
    """Manipulate the `data` (parsed JSON) based on changes in format.
    This incrementally will upgrade from version to version within data.
    Args:
      data: Dictionary representing the TensorFlow data. This will be upgraded
        in place.
    """
    while data["version"] < self._new_version:
      self._upgrade_dispatch[data["version"]](data)
      data["version"] += 1
  def Convert(self, input_file, output_file):
    for version, schema, raw_binary, _ in self._schemas:
      try:
        data_candidate = self._Read(input_file, schema, raw_binary)
      except RuntimeError:
        data_candidate["version"] = 1
        data_candidate["version"] = 1
      if data_candidate["version"] == version:
        self._PerformUpgrade(data_candidate)
        self._Write(data_candidate, output_file)
        return
    raise RuntimeError("No schema that the converter understands worked with "
                       "the data file you provided.")
def main(argv):
  del argv
  Converter().Convert(FLAGS.input, FLAGS.output)
if __name__ == "__main__":
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
