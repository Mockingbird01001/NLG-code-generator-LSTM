
import os.path as _os_path
import platform as _platform
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as _CXX11_ABI_FLAG
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as _MONOLITHIC_BUILD
from tensorflow.python.framework.versions import VERSION as _VERSION
from tensorflow.python.platform import build_info
from tensorflow.python.util.tf_export import tf_export
@tf_export('sysconfig.get_include')
def get_include():
  import tensorflow as tf
  return _os_path.join(_os_path.dirname(tf.__file__), 'include')
@tf_export('sysconfig.get_lib')
def get_lib():
  import tensorflow as tf
  return _os_path.join(_os_path.dirname(tf.__file__))
@tf_export('sysconfig.get_compile_flags')
def get_compile_flags():
  flags = []
  flags.append('-I%s' % get_include())
  flags.append('-D_GLIBCXX_USE_CXX11_ABI=%d' % _CXX11_ABI_FLAG)
  flags.append('-DEIGEN_MAX_ALIGN_BYTES=%d' %
               pywrap_tf_session.get_eigen_max_align_bytes())
  return flags
@tf_export('sysconfig.get_link_flags')
def get_link_flags():
  is_mac = _platform.system() == 'Darwin'
  ver = _VERSION.split('.')[0]
  flags = []
  if not _MONOLITHIC_BUILD:
    flags.append('-L%s' % get_lib())
    if is_mac:
      flags.append('-ltensorflow_framework.%s' % ver)
    else:
      flags.append('-l:libtensorflow_framework.so.%s' % ver)
  return flags
@tf_export('sysconfig.get_build_info')
def get_build_info():
  return build_info.build_info
