
from tensorflow.python.util.compat import as_str_any
def path_to_str(path):
  if hasattr(path, "__fspath__"):
    path = as_str_any(path.__fspath__())
  return path
