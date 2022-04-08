
from tensorflow.python.util import tf_export
def public_name(np_fun_name):
  return "experimental.numpy." + np_fun_name
def np_export(np_fun_name):
  return tf_export.tf_export(public_name(np_fun_name), v1=[])
def np_export_constant(module_name, name, value):
  np_export(name).export_constant(module_name, name)
  return value
