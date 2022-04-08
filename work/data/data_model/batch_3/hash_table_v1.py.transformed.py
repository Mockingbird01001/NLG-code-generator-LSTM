
import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1
def Test():
  z = tf.compat.v1.get_variable(
      name='y',
      shape=(),
      initializer=tf.random_normal_initializer(),
      trainable=True)
  table_initializer = tf.lookup.KeyValueTensorInitializer(
      keys=[1, 2, 3, 4],
      values=[5, 6, 7, 8],
      key_dtype=tf.int32,
      value_dtype=tf.float32)
  table = tf.lookup.StaticHashTable(
      table_initializer, default_value=tf.constant(0.0))
  x = tf.placeholder(tf.int32, shape=(), name='input')
  y = table.lookup(x)
  r = tf.add(y, z)
  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)
  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name='some_function'))
  }, tf.tables_initializer(), None
if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test, canonicalize=True)
