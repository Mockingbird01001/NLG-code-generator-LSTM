
import argparse
import six
try:
except ImportError:
  cuda_config = None
try:
except ImportError:
  tensorrt_config = None
def write_build_info(filename, key_value_list):
  build_info = {}
  if cuda_config:
    build_info.update(cuda_config.config)
  if tensorrt_config:
    build_info.update(tensorrt_config.config)
  for arg in key_value_list:
    key, value = six.ensure_str(arg).split("=")
    if value.lower() == "true":
      build_info[key] = True
    elif value.lower() == "false":
      build_info[key] = False
    else:
      build_info[key] = value.format(**build_info)
  sorted_build_info_pairs = sorted(build_info.items())
  contents = """
\"\"\"Auto-generated module providing information about the build.\"\"\"
import collections
build_info = collections.OrderedDict(%s)
""" % sorted_build_info_pairs
  open(filename, "w").write(contents)
parser = argparse.ArgumentParser(
    description=
)
parser.add_argument("--raw_generate", type=str, help="Generate build_info.py")
parser.add_argument(
    "--key_value", type=str, nargs="*", help="List of key=value pairs.")
args = parser.parse_args()
if args.raw_generate:
  write_build_info(args.raw_generate, args.key_value)
else:
  raise RuntimeError("--raw_generate must be used.")
