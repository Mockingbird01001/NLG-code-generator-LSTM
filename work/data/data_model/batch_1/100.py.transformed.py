
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
def make_process_name_useful():
  set_kernel_process_name(os.path.basename(sys.argv[0]))
def set_kernel_process_name(name):
  if not isinstance(name, bytes):
    name = name.encode('ascii', 'replace')
  try:
    with open('/proc/self/comm', 'wb') as proc_comm:
      proc_comm.write(name[:15])
  except EnvironmentError:
    try:
      import ctypes
    except ImportError:
      return
    try:
      libc = ctypes.CDLL('libc.so.6')
    except EnvironmentError:
      return
    pr_set_name = ctypes.c_ulong(15)
    zero = ctypes.c_ulong(0)
    try:
      libc.prctl(pr_set_name, name, zero, zero, zero)
    except AttributeError:
      return
