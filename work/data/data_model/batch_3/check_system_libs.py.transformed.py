
import glob
import os
import sys
tf_source_path = sys.argv[1]
syslibs_configure_path = os.path.join(tf_source_path, 'third_party',
                                      'systemlibs', 'syslibs_configure.bzl')
workspace0_path = os.path.join(tf_source_path, 'tensorflow', 'workspace0.bzl')
workspace_glob = os.path.join(tf_source_path, 'tensorflow', 'workspace*.bzl')
third_party_path = os.path.join(tf_source_path, 'third_party')
third_party_glob = os.path.join(third_party_path, '*', 'workspace.bzl')
if not os.path.isdir(tf_source_path):
  raise ValueError('The path to the TensorFlow source must be passed as'
                   ' the first argument')
if not os.path.isfile(syslibs_configure_path):
  raise ValueError('Could not find syslibs_configure.bzl at %s' %
                   syslibs_configure_path)
if not os.path.isfile(workspace0_path):
  raise ValueError('Could not find workspace0.bzl at %s' % workspace0_path)
def extract_valid_libs(filepath):
    del kwargs
  with open(filepath, 'r') as f:
    f_globals = {'repository_rule': repository_rule}
    f_locals = {}
  return set(f_locals['VALID_LIBS'])
def extract_system_builds(filepath):
  lib_names = []
  system_build_files = []
  current_name = None
  with open(filepath, 'r') as f:
    for line in f:
      line = line.strip()
      if line.startswith('name = '):
        current_name = line[7:-1].strip('"')
      elif line.startswith('system_build_file = '):
        lib_names.append(current_name)
        system_build_spec = line.split('=')[-1].split('"')[1]
        assert system_build_spec.startswith('//')
        system_build_files.append(system_build_spec[2:].replace(':', os.sep))
  return lib_names, system_build_files
syslibs = extract_valid_libs(syslibs_configure_path)
syslibs_from_workspace = set()
system_build_files_from_workspace = []
for current_path in glob.glob(workspace_glob) + glob.glob(third_party_glob):
  cur_lib_names, build_files = extract_system_builds(current_path)
  syslibs_from_workspace.update(cur_lib_names)
  system_build_files_from_workspace.extend(build_files)
missing_build_files = [
    file for file in system_build_files_from_workspace
    if not os.path.isfile(os.path.join(tf_source_path, file))
]
has_error = False
if missing_build_files:
  has_error = True
  print('Missing system build files: ' + ', '.join(missing_build_files))
if syslibs != syslibs_from_workspace:
  has_error = True
  missing_syslibs = syslibs_from_workspace - syslibs
  if missing_syslibs:
    libs = ', '.join(sorted(missing_syslibs))
    print('Libs missing from syslibs_configure: ' + libs)
  additional_syslibs = syslibs - syslibs_from_workspace
  if additional_syslibs:
    libs = ', '.join(sorted(additional_syslibs))
    print('Libs missing in workspace (or superfluous in syslibs_configure): ' +
          libs)
sys.exit(1 if has_error else 0)
