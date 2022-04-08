
import os
import importlib
def get_parent_dir(module):
  return os.path.abspath(os.path.join(os.path.dirname(module.__file__), ".."))
def get_parent_dir_for_name(module_name):
  name_split = module_name.split(".")
  if not name_split:
    return None
  try:
    spec = importlib.util.find_spec(name_split[0])
  except ValueError:
    return None
  if not spec or not spec.origin:
    return None
  base_path = os.path.dirname(spec.origin)
  return os.path.join(base_path, *name_split[1:-1])
