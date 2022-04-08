
import ctypes
import sys
import traceback
from tensorflow.python.platform import self_check
self_check.preload_check()
try:
  from tensorflow.python import pywrap_dlopen_global_flags
  _use_dlopen_global_flags = True
except ImportError:
  _use_dlopen_global_flags = False
_can_set_rtld_local = (
    hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'))
if _can_set_rtld_local:
  _default_dlopen_flags = sys.getdlopenflags()
try:
  if _use_dlopen_global_flags:
    pywrap_dlopen_global_flags.set_dlopen_flags()
  elif _can_set_rtld_local:
    sys.setdlopenflags(_default_dlopen_flags | ctypes.RTLD_LOCAL)
  try:
    ModuleNotFoundError
  except NameError:
  try:
    from tensorflow.python._pywrap_tensorflow_internal import *
  except ModuleNotFoundError:
    pass
  if _use_dlopen_global_flags:
    pywrap_dlopen_global_flags.reset_dlopen_flags()
  elif _can_set_rtld_local:
    sys.setdlopenflags(_default_dlopen_flags)
except ImportError:
  raise ImportError(
      f'{traceback.format_exc()}'
      f'\n\nFailed to load the native TensorFlow runtime.\n'
      f'See https://www.tensorflow.org/install/errors '
      f'for some common causes and solutions.\n'
      f'If you need help, create an issue '
      f'at https://github.com/tensorflow/tensorflow/issues '
      f'and include the entire stack trace above this error message.')
