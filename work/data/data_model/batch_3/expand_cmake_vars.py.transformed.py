
import re
import sys
_CMAKE_VAR_REGEX = re.compile(r"\${([A-Za-z_0-9]*)}")
_CMAKE_ATVAR_REGEX = re.compile(r"@([A-Za-z_0-9]*)@")
def _parse_args(argv):
  result = {}
  for arg in argv:
    k, v = arg.split("=")
    result[k] = v
  return result
def _expand_variables(input_str, cmake_vars):
  def replace(match):
    if match.group(1) in cmake_vars:
      return cmake_vars[match.group(1)]
    return ""
  return _CMAKE_ATVAR_REGEX.sub(replace,_CMAKE_VAR_REGEX.sub(replace, input_str))
def _expand_cmakedefines(line, cmake_vars):
  match = _CMAKE_DEFINE_REGEX.match(line)
  if match:
    name = match.group(1)
    suffix = match.group(2) or ""
    if name in cmake_vars:
                                     _expand_variables(suffix, cmake_vars))
    else:
  match = _CMAKE_DEFINE01_REGEX.match(line)
  if match:
    name = match.group(1)
    value = cmake_vars.get(name, "0")
  return _expand_variables(line, cmake_vars)
def main():
  cmake_vars = _parse_args(sys.argv[1:])
  for line in sys.stdin:
    sys.stdout.write(_expand_cmakedefines(line, cmake_vars))
if __name__ == "__main__":
  main()
