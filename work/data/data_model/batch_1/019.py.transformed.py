from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
try:
  from absl.testing import _bazel_selected_py3
except ImportError:
  _bazel_selected_py3 = None
FLAGS = flags.FLAGS
def get_executable_path(py_binary_name):
  root, ext = os.path.splitext(py_binary_name)
  suffix = 'py3' if _bazel_selected_py3 else 'py2'
  py_binary_name = '{}_{}{}'.format(root, suffix, ext)
  if os.name == 'nt':
    py_binary_name += '.exe'
    manifest_file = os.path.join(FLAGS.test_srcdir, 'MANIFEST')
    workspace_name = os.environ['TEST_WORKSPACE']
    manifest_entry = '{}/{}'.format(workspace_name, py_binary_name)
    with open(manifest_file, 'r') as manifest_fd:
      for line in manifest_fd:
        tokens = line.strip().split(' ')
        if len(tokens) != 2:
          continue
        if manifest_entry == tokens[0]:
          return tokens[1]
    raise RuntimeError(
        'Cannot locate executable path for {}, MANIFEST file: {}.'.format(
            py_binary_name, manifest_file))
  else:
    path = __file__
    for _ in range(__name__.count('.') + 1):
      path = os.path.dirname(path)
    root_directory = path
    return os.path.join(root_directory, py_binary_name)
