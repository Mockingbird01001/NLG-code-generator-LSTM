
import errno
import hashlib
import importlib
import os
import platform
import sys
from tensorflow.python.client import pywrap_tf_session as py_tf
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_op_gen
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('load_op_library')
def load_op_library(library_filename):
  lib_handle = py_tf.TF_LoadLibrary(library_filename)
  try:
    wrappers = _pywrap_python_op_gen.GetPythonWrappers(
        py_tf.TF_GetOpList(lib_handle))
  finally:
    py_tf.TF_DeleteLibraryHandle(lib_handle)
  module_name = hashlib.sha1(wrappers).hexdigest()
  if module_name in sys.modules:
    return sys.modules[module_name]
  module_spec = importlib.machinery.ModuleSpec(module_name, None)
  module = importlib.util.module_from_spec(module_spec)
  exec(wrappers, module.__dict__)
  setattr(module, '_IS_TENSORFLOW_PLUGIN', True)
  sys.modules[module_name] = module
  return module
@deprecation.deprecated(date=None,
                        instructions='Use `tf.load_library` instead.')
@tf_export(v1=['load_file_system_library'])
def load_file_system_library(library_filename):
  py_tf.TF_LoadLibrary(library_filename)
def _is_shared_object(filename):
  if platform.system() == 'Linux':
    if filename.endswith('.so'):
      return True
    else:
      index = filename.rfind('.so.')
      if index == -1:
        return False
      else:
        return filename[index + 4].isdecimal()
  elif platform.system() == 'Darwin':
    return filename.endswith('.dylib')
  elif platform.system() == 'Windows':
    return filename.endswith('.dll')
  else:
    return False
@tf_export('load_library')
def load_library(library_location):
  if os.path.exists(library_location):
    if os.path.isdir(library_location):
      directory_contents = os.listdir(library_location)
      kernel_libraries = [
          os.path.join(library_location, f) for f in directory_contents
          if _is_shared_object(f)]
    else:
      kernel_libraries = [library_location]
    for lib in kernel_libraries:
      py_tf.TF_LoadLibrary(lib)
  else:
    raise OSError(
        errno.ENOENT,
        'The file or folder to load kernel libraries from does not exist.',
        library_location)
def load_pluggable_device_library(library_location):
  if os.path.exists(library_location):
    if os.path.isdir(library_location):
      directory_contents = os.listdir(library_location)
      pluggable_device_libraries = [
          os.path.join(library_location, f)
          for f in directory_contents
          if _is_shared_object(f)
      ]
    else:
      pluggable_device_libraries = [library_location]
    for lib in pluggable_device_libraries:
      py_tf.TF_LoadPluggableDeviceLibrary(lib)
    context.context().reinitialize_physical_devices()
  else:
    raise OSError(
        errno.ENOENT,
        'The file or folder to load pluggable device libraries from does not '
        'exist.', library_location)
@tf_export('experimental.register_filesystem_plugin')
def register_filesystem_plugin(plugin_location):
  if os.path.exists(plugin_location):
    py_tf.TF_RegisterFilesystemPlugin(plugin_location)
  else:
    raise OSError(errno.ENOENT,
                  'The file to load file system plugin from does not exist.',
                  plugin_location)
