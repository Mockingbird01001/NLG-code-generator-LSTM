
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['saved_model.main_op.main_op'])
@deprecation.deprecated(
    None,
    'This function will only be available through the v1 compatibility '
    'library as tf.compat.v1.saved_model.main_op.main_op.')
def main_op():
  init = variables.global_variables_initializer()
  init_local = variables.local_variables_initializer()
  init_tables = lookup_ops.tables_initializer()
  return control_flow_ops.group(init, init_local, init_tables)
@tf_export(v1=['saved_model.main_op_with_restore',
               'saved_model.main_op.main_op_with_restore'])
@deprecation.deprecated(
    None,
    'This function will only be available through the v1 compatibility '
    'library as tf.compat.v1.saved_model.main_op_with_restore or '
    'tf.compat.v1.saved_model.main_op.main_op_with_restore.')
def main_op_with_restore(restore_op_name):
  with ops.control_dependencies([main_op()]):
    main_op_with_restore = control_flow_ops.group(restore_op_name)
  return main_op_with_restore
