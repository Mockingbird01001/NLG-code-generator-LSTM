
import os as _os
import sys as _sys
from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.tf_export import tf_export
try:
  from rules_python.python.runfiles import runfiles
except ImportError:
  runfiles = None
@tf_export(v1=['resource_loader.load_resource'])
def load_resource(path):
  with open(get_path_to_datafile(path), 'rb') as f:
    return f.read()
@tf_export(v1=['resource_loader.get_data_files_path'])
def get_data_files_path():
  return _os.path.dirname(_inspect.getfile(_sys._getframe(1)))
@tf_export(v1=['resource_loader.get_root_dir_with_all_resources'])
def get_root_dir_with_all_resources():
  script_dir = get_data_files_path()
  directories = [script_dir]
  data_files_dir = ''
  while True:
    candidate_dir = directories[-1]
    current_directory = _os.path.basename(candidate_dir)
    if '.runfiles' in current_directory:
      if len(directories) > 1:
        data_files_dir = directories[-2]
      break
    else:
      new_candidate_dir = _os.path.dirname(candidate_dir)
      if new_candidate_dir == candidate_dir:
        break
      else:
        directories.append(new_candidate_dir)
  return data_files_dir or script_dir
@tf_export(v1=['resource_loader.get_path_to_datafile'])
def get_path_to_datafile(path):
  if runfiles:
    r = runfiles.Create()
    new_fpath = r.Rlocation(
        _os.path.abspath(_os.path.join('tensorflow', path)))
    if new_fpath is not None and _os.path.exists(new_fpath):
      return new_fpath
  old_filepath = _os.path.join(
      _os.path.dirname(_inspect.getfile(_sys._getframe(1))), path)
  return old_filepath
@tf_export(v1=['resource_loader.readahead_file_path'])
  return path
