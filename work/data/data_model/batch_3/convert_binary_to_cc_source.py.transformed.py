
import argparse
import datetime
import sys
def _convert_bytes_to_cc_source(data,
                                array_name,
                                max_line_width=80,
                                include_guard=None,
                                include_path=None,
                                use_tensorflow_license=False):
  starting_pad = "   "
  array_lines = []
  array_line = starting_pad
  for value in bytearray(data):
    if (len(array_line) + 4) > max_line_width:
      array_lines.append(array_line + "\n")
      array_line = starting_pad
    array_line += " 0x%02x," % value
  if len(array_line) > len(starting_pad):
    array_lines.append(array_line + "\n")
  array_values = "".join(array_lines)
  if include_guard is None:
    include_guard = "TENSORFLOW_LITE_UTIL_" + array_name.upper() + "_DATA_H_"
  if include_path is not None:
        include_path=include_path)
  else:
    include_line = ""
  if use_tensorflow_license:
    license_text = """
/* Copyright {year} The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
""".format(year=datetime.date.today().year)
  else:
    license_text = ""
  source_template =
  source_text = source_template.format(
      array_name=array_name,
      array_length=len(data),
      array_values=array_values,
      license_text=license_text,
      include_line=include_line)
  header_template =
  header_text = header_template.format(
      array_name=array_name,
      include_guard=include_guard,
      license_text=license_text)
  return source_text, header_text
def main():
  parser = argparse.ArgumentParser(
      description=("Binary to C++ source converter"))
  parser.add_argument(
      "--input_binary_file",
      type=str,
      help="Full filepath of input binary.",
      required=True)
  parser.add_argument(
      "--output_header_file",
      type=str,
      help="Full filepath of output header.",
      required=True)
  parser.add_argument(
      "--array_variable_name",
      type=str,
      help="Full filepath of output source.",
      required=True)
  parser.add_argument(
      "--output_source_file",
      type=str,
      help="Name of global variable that will contain the binary data.",
      required=True)
  flags, _ = parser.parse_known_args(args=sys.argv[1:])
  with open(flags.input_binary_file, "rb") as input_handle:
    input_data = input_handle.read()
  source, header = _convert_bytes_to_cc_source(
      data=input_data,
      array_name=flags.array_variable_name,
      use_tensorflow_license=True)
  with open(flags.output_source_file, "w") as source_handle:
    source_handle.write(source)
  with open(flags.output_header_file, "w") as header_handle:
    header_handle.write(header)
if __name__ == "__main__":
  main()
