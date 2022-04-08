
import re
import shutil
import sys
import tempfile
from typing import Dict
class FileCheckVarReplacer:
  _counter: int
  _replacement_cache: Dict[str, str]
  _check_instruction_matcher: re.Pattern = re.compile(r"^[^:]*CHECK[^:]*:.*=")
  _instr_name_matcher: re.Pattern = re.compile(r"%[\w-]+(\.\d+)?")
  def __init__(self):
    self._counter = -1
    self._replacement_cache = {}
  def replace_instruction_names_for_line(self, line: str) -> str:
    if not self._check_instruction_matcher.match(line):
      self._counter = -1
      self._replacement_cache = {}
      return line
    return re.sub(self._instr_name_matcher, self._replacer, line)
  def _replacer(self, m: re.Match) -> str:
    instr_name = m.group(0)
    if instr_name in self._replacement_cache:
      return self._replacement_cache[instr_name]
    replacement_instr = self._generate_unique_varname()
    self._replacement_cache[instr_name] = f"[[{replacement_instr}]]"
    return "".join([f"[[{replacement_instr}:", r"%[^ ]+", "]]"])
  def _generate_unique_varname(self) -> str:
    self._counter += 1
    return f"INSTR_{self._counter}"
def replace_instruction_names(t: str) -> str:
  f = FileCheckVarReplacer()
  out = []
  for line in t.split("\n"):
    out.append(f.replace_instruction_names_for_line(line))
  return "\n".join(out)
def main() -> None:
  argv = sys.argv
  if len(argv) != 2:
    raise Exception("Expecting exactly one filename argument (or -)")
  r = FileCheckVarReplacer()
  input_filename = argv[1]
  if input_filename == "-":
    for line in sys.stdin:
      sys.stdout.write(r.replace_instruction_names_for_line(line))
    return 0
  with open(input_filename) as f:
    fd, fname = tempfile.mkstemp()
    with open(fd, "w") as out_f:
      for line in f:
        out_f.write(r.replace_instruction_names_for_line(line))
  shutil.move(fname, input_filename)
if __name__ == "__main__":
  main()
