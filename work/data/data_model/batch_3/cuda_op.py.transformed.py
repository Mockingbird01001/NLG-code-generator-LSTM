
import os.path
import tensorflow as tf
if tf.test.is_built_with_cuda():
  _cuda_op_module = tf.load_op_library(os.path.join(
      tf.compat.v1.resource_loader.get_data_files_path(), 'cuda_op_kernel.so'))
  add_one = _cuda_op_module.add_one
